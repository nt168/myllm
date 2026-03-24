/**
 * @file sampler.c
 * @brief 采样器实现
 *
 * 实现各种采样策略:
 * - Temperature scaling
 * - Top-K sampling
 * - Top-P (Nucleus) sampling
 * - Repetition penalty
 * - Softmax and random sampling
 */

#include "sampler/sampler.h"
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <float.h>

/* ============================================================================
 * 内部数据结构
 * ============================================================================ */

/**
 * @brief 采样器结构
 */
struct Sampler {
    SamplerConfig config;
    uint64_t rng_state;
};

/* ============================================================================
 * 随机数生成 (PCG-XSH-RR)
 * ============================================================================ */

uint64_t sampler_pcg_random(uint64_t* state) {
    if (!state) return 0;

    uint64_t oldstate = *state;
    *state = oldstate * 6364136223846793005ULL + 1442695040888963407ULL;

    uint32_t xorshifted = ((oldstate >> 18u) ^ oldstate) >> 27u;
    uint32_t rot = oldstate >> 59u;

    return (xorshifted >> rot) | (xorshifted << ((-rot) & 31));
}

float sampler_random_float(uint64_t* state) {
    if (!state) return 0.0f;

    uint64_t r = sampler_pcg_random(state);
    return (float)(r >> 40) / (float)(1ULL << 24);
}

/* ============================================================================
 * 生命周期管理
 * ============================================================================ */

SamplerConfig sampler_default_config(void) {
    SamplerConfig config;
    memset(&config, 0, sizeof(SamplerConfig));

    config.temperature = SAMPLER_DEFAULT_TEMPERATURE;
    config.top_k = SAMPLER_DEFAULT_TOP_K;
    config.top_p = SAMPLER_DEFAULT_TOP_P;
    config.repetition_penalty = SAMPLER_DEFAULT_REPETITION_PENALTY;
    config.penalty_tokens = NULL;
    config.num_penalty_tokens = 0;
    config.seed = 0xDEADBEEFCAFEBABEULL;

    return config;
}

Sampler* sampler_new(void) {
    return sampler_new_with_config(NULL);
}

Sampler* sampler_new_with_config(const SamplerConfig* config) {
    Sampler* sampler = (Sampler*)malloc(sizeof(Sampler));
    if (!sampler) return NULL;

    if (config) {
        memcpy(&sampler->config, config, sizeof(SamplerConfig));
    } else {
        sampler->config = sampler_default_config();
    }

    sampler->rng_state = sampler->config.seed;

    if (sampler->config.temperature < SAMPLER_MIN_TEMPERATURE) {
        sampler->config.temperature = SAMPLER_MIN_TEMPERATURE;
    }

    return sampler;
}

void sampler_free(Sampler* sampler) {
    if (!sampler) return;
    free(sampler);
}

/* ============================================================================
 * 配置设置
 * ============================================================================ */

void sampler_set_temperature(Sampler* sampler, float temperature) {
    if (!sampler) return;
    sampler->config.temperature = (temperature < SAMPLER_MIN_TEMPERATURE)
        ? SAMPLER_MIN_TEMPERATURE : temperature;
}

void sampler_set_top_k(Sampler* sampler, int32_t top_k) {
    if (!sampler) return;
    sampler->config.top_k = top_k;
}

void sampler_set_top_p(Sampler* sampler, float top_p) {
    if (!sampler) return;
    if (top_p < 0.0f) top_p = 0.0f;
    if (top_p > 1.0f) top_p = 1.0f;
    sampler->config.top_p = top_p;
}

void sampler_set_repetition_penalty(Sampler* sampler, float penalty) {
    if (!sampler) return;
    sampler->config.repetition_penalty = penalty;
}

void sampler_set_seed(Sampler* sampler, uint64_t seed) {
    if (!sampler) return;
    sampler->rng_state = seed;
    sampler->config.seed = seed;
}

/* ============================================================================
 * 工具函数
 * ============================================================================ */

float sampler_max_logit(const float* logits, size_t size) {
    if (!logits || size == 0) return -FLT_MAX;

    float max_val = logits[0];
    for (size_t i = 1; i < size; i++) {
        if (logits[i] > max_val) {
            max_val = logits[i];
        }
    }
    return max_val;
}

int32_t sampler_argmax(const float* logits, size_t size) {
    if (!logits || size == 0) return -1;

    int32_t max_idx = 0;
    float max_val = logits[0];

    for (size_t i = 1; i < size; i++) {
        if (logits[i] > max_val) {
            max_val = logits[i];
            max_idx = (int32_t)i;
        }
    }
    return max_idx;
}

float sampler_logsumexp(const float* logits, size_t size) {
    if (!logits || size == 0) return 0.0f;

    float max_val = sampler_max_logit(logits, size);
    if (max_val == -FLT_MAX) return -FLT_MAX;

    float sum = 0.0f;
    for (size_t i = 0; i < size; i++) {
        if (logits[i] > -FLT_MAX / 2) {
            sum += expf(logits[i] - max_val);
        }
    }
    return max_val + logf(sum);
}

/* ============================================================================
 * Softmax
 * ============================================================================ */

void sampler_softmax(float* logits, size_t size) {
    if (!logits || size == 0) return;

    float max_val = sampler_max_logit(logits, size);

    float sum = 0.0f;
    for (size_t i = 0; i < size; i++) {
        logits[i] = expf(logits[i] - max_val);
        sum += logits[i];
    }

    if (sum > 0.0f) {
        for (size_t i = 0; i < size; i++) {
            logits[i] /= sum;
        }
    }
}

/* ============================================================================
 * Temperature
 * ============================================================================ */

void sampler_apply_temperature(float* logits, size_t size, float temperature) {
    if (!logits || size == 0) return;

    if (temperature < SAMPLER_MIN_TEMPERATURE) {
        temperature = SAMPLER_MIN_TEMPERATURE;
    }

    float inv_temp = 1.0f / temperature;
    for (size_t i = 0; i < size; i++) {
        logits[i] *= inv_temp;
    }
}

/* ============================================================================
 * 重复惩罚
 * ============================================================================ */

void sampler_apply_repetition_penalty(
    float* logits,
    size_t size,
    const int32_t* token_ids,
    size_t num_tokens,
    float penalty
) {
    if (!logits || !token_ids || num_tokens == 0 || penalty == 1.0f) return;

    for (size_t i = 0; i < num_tokens; i++) {
        int32_t id = token_ids[i];
        if (id < 0 || id >= (int32_t)size) continue;

        float logit = logits[id];
        if (logit > 0.0f) {
            logits[id] = logit / penalty;
        } else {
            logits[id] = logit * penalty;
        }
    }
}

/* ============================================================================
 * Top-K 过滤
 * ============================================================================ */

/* 迭代式快速排序，避免栈溢出 */
#define QSORT_STACK_SIZE 64

static void quicksort_descending_iterative(int32_t* indices, const float* values, int n) {
    if (n <= 1) return;

    int stack[QSORT_STACK_SIZE];
    int top = -1;

    stack[++top] = 0;
    stack[++top] = n - 1;

    while (top >= 1) {
        int high = stack[top--];
        int low = stack[top--];

        if (low >= high) continue;

        /* 使用三数取中选择基准，避免最坏情况 */
        int mid = low + (high - low) / 2;

        /* 对 low, mid, high 位置的值排序 */
        if (values[indices[mid]] > values[indices[low]]) {
            int32_t tmp = indices[low];
            indices[low] = indices[mid];
            indices[mid] = tmp;
        }
        if (values[indices[high]] > values[indices[low]]) {
            int32_t tmp = indices[low];
            indices[low] = indices[high];
            indices[high] = tmp;
        }
        if (values[indices[mid]] > values[indices[high]]) {
            int32_t tmp = indices[mid];
            indices[mid] = indices[high];
            indices[high] = tmp;
        }

        /* 将基准放到 high-1 位置 */
        float pivot = values[indices[high]];
        int i = low - 1;

        for (int j = low; j < high; j++) {
            if (values[indices[j]] >= pivot) {
                i++;
                int32_t tmp = indices[i];
                indices[i] = indices[j];
                indices[j] = tmp;
            }
        }

        int32_t tmp = indices[i + 1];
        indices[i + 1] = indices[high];
        indices[high] = tmp;

        int pi = i + 1;

        /* 先处理较小的分区，保证栈深度为 O(log n) */
        int left_size = pi - 1 - low;
        int right_size = high - pi - 1;

        if (left_size < right_size) {
            if (pi + 1 < high) {
                stack[++top] = pi + 1;
                stack[++top] = high;
            }
            if (low < pi - 1) {
                stack[++top] = low;
                stack[++top] = pi - 1;
            }
        } else {
            if (low < pi - 1) {
                stack[++top] = low;
                stack[++top] = pi - 1;
            }
            if (pi + 1 < high) {
                stack[++top] = pi + 1;
                stack[++top] = high;
            }
        }
    }
}

void sampler_apply_top_k(float* logits, size_t size, int32_t k) {
    if (!logits || size == 0 || k <= 0 || k >= (int32_t)size) return;

    int32_t* indices = (int32_t*)malloc(size * sizeof(int32_t));
    if (!indices) return;

    for (size_t i = 0; i < size; i++) {
        indices[i] = (int32_t)i;
    }

    quicksort_descending_iterative(indices, logits, (int)size);

    for (size_t i = k; i < size; i++) {
        logits[indices[i]] = -FLT_MAX;
    }

    free(indices);
}

/* ============================================================================
 * Top-P 过滤
 * ============================================================================ */

void sampler_apply_top_p(float* logits, size_t size, float p) {
    if (!logits || size == 0 || p >= 1.0f || p <= 0.0f) return;

    float* probs = (float*)malloc(size * sizeof(float));
    if (!probs) return;

    float max_val = sampler_max_logit(logits, size);
    float sum = 0.0f;

    for (size_t i = 0; i < size; i++) {
        probs[i] = expf(logits[i] - max_val);
        sum += probs[i];
    }

    if (sum <= 0.0f) {
        free(probs);
        return;
    }

    int32_t* indices = (int32_t*)malloc(size * sizeof(int32_t));
    if (!indices) {
        free(probs);
        return;
    }

    for (size_t i = 0; i < size; i++) {
        indices[i] = (int32_t)i;
        probs[i] /= sum;
    }

    quicksort_descending_iterative(indices, probs, (int)size);

    float cumsum = 0.0f;
    int32_t cutoff = (int32_t)size;

    for (size_t i = 0; i < size; i++) {
        cumsum += probs[indices[i]];
        if (cumsum > p) {
            cutoff = (int32_t)i + 1;
            break;
        }
    }

    for (size_t i = cutoff; i < size; i++) {
        logits[indices[i]] = -FLT_MAX;
    }

    free(indices);
    free(probs);
}

/* ============================================================================
 * 随机采样
 * ============================================================================ */

int32_t sampler_random_sample(const float* probs, size_t size, uint64_t* rng_state) {
    if (!probs || size == 0 || !rng_state) return -1;

    float r = sampler_random_float(rng_state);

    float cumsum = 0.0f;
    for (size_t i = 0; i < size; i++) {
        cumsum += probs[i];
        if (r < cumsum) {
            return (int32_t)i;
        }
    }

    return (int32_t)(size - 1);
}

/* ============================================================================
 * 贪婪采样
 * ============================================================================ */

int32_t sampler_sample_greedy(const float* logits, size_t vocab_size) {
    return sampler_argmax(logits, vocab_size);
}

/* ============================================================================
 * 温度采样
 * ============================================================================ */

int32_t sampler_sample_temperature(
    Sampler* sampler,
    const float* logits,
    size_t vocab_size
) {
    if (!sampler || !logits || vocab_size == 0) return -1;

    float* probs = (float*)malloc(vocab_size * sizeof(float));
    if (!probs) return -1;

    memcpy(probs, logits, vocab_size * sizeof(float));

    sampler_apply_temperature(probs, vocab_size, sampler->config.temperature);
    sampler_softmax(probs, vocab_size);

    int32_t token = sampler_random_sample(probs, vocab_size, &sampler->rng_state);

    free(probs);
    return token;
}

/* ============================================================================
 * 完整采样流程
 * ============================================================================ */

int32_t sampler_sample(Sampler* sampler, const float* logits, size_t vocab_size) {
    if (!sampler || !logits || vocab_size == 0) return -1;

    float* probs = (float*)malloc(vocab_size * sizeof(float));
    if (!probs) return -1;

    memcpy(probs, logits, vocab_size * sizeof(float));

    /* 1. 应用温度 */
    if (sampler->config.temperature != 1.0f) {
        sampler_apply_temperature(probs, vocab_size, sampler->config.temperature);
    }

    /* 2. 应用重复惩罚 */
    if (sampler->config.repetition_penalty != 1.0f &&
        sampler->config.penalty_tokens &&
        sampler->config.num_penalty_tokens > 0) {
        sampler_apply_repetition_penalty(
            probs, vocab_size,
            sampler->config.penalty_tokens,
            sampler->config.num_penalty_tokens,
            sampler->config.repetition_penalty
        );
    }

    /* 3. Top-K 过滤 */
    if (sampler->config.top_k > 0 && sampler->config.top_k < (int32_t)vocab_size) {
        sampler_apply_top_k(probs, vocab_size, sampler->config.top_k);
    }

    /* 4. Top-P 过滤 */
    if (sampler->config.top_p > 0.0f && sampler->config.top_p < 1.0f) {
        sampler_apply_top_p(probs, vocab_size, sampler->config.top_p);
    }

    /* 5. Softmax */
    sampler_softmax(probs, vocab_size);

    /* 6. 随机采样 */
    int32_t token = sampler_random_sample(probs, vocab_size, &sampler->rng_state);

    free(probs);
    return token;
}
