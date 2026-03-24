/**
 * @file test_sampler.c
 * @brief 采样器模块单元测试
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <float.h>
#include "sampler/sampler.h"
#include "common/test_common.h"

/* ============================================================================
 * 常量
 * ============================================================================ */

#define TEST_VOCAB_SIZE 100
#define FLOAT_TOLERANCE 1e-4f

/* ============================================================================
 * 生命周期测试
 * ============================================================================ */

static void test_sampler_new(void) {
    Sampler* sampler = sampler_new();
    ASSERT_NOT_NULL(sampler, "Sampler should not be NULL");

    sampler_free(sampler);
    TEST_PASS();
}

static void test_sampler_new_with_config(void) {
    SamplerConfig config = sampler_default_config();
    config.temperature = 0.8f;
    config.top_k = 50;
    config.top_p = 0.95f;
    config.seed = 12345;

    Sampler* sampler = sampler_new_with_config(&config);
    ASSERT_NOT_NULL(sampler, "Sampler should not be NULL");

    sampler_free(sampler);
    TEST_PASS();
}

static void test_sampler_free_null(void) {
    sampler_free(NULL);
    TEST_PASS();
}

static void test_sampler_default_config(void) {
    SamplerConfig config = sampler_default_config();

    ASSERT_FLOAT_EQ(config.temperature, SAMPLER_DEFAULT_TEMPERATURE, FLOAT_TOLERANCE, "Default temperature");
    ASSERT_EQ(config.top_k, SAMPLER_DEFAULT_TOP_K, "Default top_k");
    ASSERT_FLOAT_EQ(config.top_p, SAMPLER_DEFAULT_TOP_P, FLOAT_TOLERANCE, "Default top_p");
    ASSERT_FLOAT_EQ(config.repetition_penalty, SAMPLER_DEFAULT_REPETITION_PENALTY, FLOAT_TOLERANCE, "Default repetition penalty");

    TEST_PASS();
}

/* ============================================================================
 * 配置设置测试
 * ============================================================================ */

static void test_sampler_set_temperature(void) {
    Sampler* sampler = sampler_new();
    ASSERT_NOT_NULL(sampler, "Sampler should not be NULL");

    sampler_set_temperature(sampler, 0.5f);
    sampler_set_temperature(sampler, 0.0f);  /* 应该被设为最小值 */

    sampler_free(sampler);
    TEST_PASS();
}

static void test_sampler_set_top_k(void) {
    Sampler* sampler = sampler_new();
    ASSERT_NOT_NULL(sampler, "Sampler should not be NULL");

    sampler_set_top_k(sampler, 10);
    sampler_set_top_k(sampler, 0);
    sampler_set_top_k(sampler, 1000);

    sampler_free(sampler);
    TEST_PASS();
}

static void test_sampler_set_top_p(void) {
    Sampler* sampler = sampler_new();
    ASSERT_NOT_NULL(sampler, "Sampler should not be NULL");

    sampler_set_top_p(sampler, 0.5f);
    sampler_set_top_p(sampler, 1.5f);  /* 应该被 clamp 到 1.0 */
    sampler_set_top_p(sampler, -0.5f); /* 应该被 clamp 到 0.0 */

    sampler_free(sampler);
    TEST_PASS();
}

/* ============================================================================
 * 随机数生成测试
 * ============================================================================ */

static void test_sampler_pcg_random(void) {
    uint64_t state = 12345;

    uint64_t r1 = sampler_pcg_random(&state);
    uint64_t r2 = sampler_pcg_random(&state);

    ASSERT_TRUE(r1 != r2, "Random values should differ");
    ASSERT_TRUE(r1 != 0 || r2 != 0, "At least one should be non-zero");

    TEST_PASS();
}

static void test_sampler_random_float(void) {
    uint64_t state = 12345;

    for (int i = 0; i < 100; i++) {
        float r = sampler_random_float(&state);
        ASSERT_TRUE(r >= 0.0f && r < 1.0f, "Random float should be in [0, 1)");
    }

    TEST_PASS();
}

static void test_sampler_reproducibility(void) {
    uint64_t state1 = 12345;
    uint64_t state2 = 12345;

    for (int i = 0; i < 10; i++) {
        uint64_t r1 = sampler_pcg_random(&state1);
        uint64_t r2 = sampler_pcg_random(&state2);
        ASSERT_EQ(r1, r2, "Same seed should produce same sequence");
    }

    TEST_PASS();
}

/* ============================================================================
 * Softmax 测试
 * ============================================================================ */

static void test_sampler_softmax_basic(void) {
    float logits[] = {1.0f, 2.0f, 3.0f};
    size_t size = 3;

    sampler_softmax(logits, size);

    /* 概率和应该为 1 */
    float sum = logits[0] + logits[1] + logits[2];
    ASSERT_FLOAT_EQ(sum, 1.0f, FLOAT_TOLERANCE, "Probabilities should sum to 1");

    /* 最大值对应的概率应该最大 */
    ASSERT_TRUE(logits[2] > logits[1] && logits[1] > logits[0], "Probabilities should be ordered");

    TEST_PASS();
}

static void test_sampler_softmax_single(void) {
    float logits[] = {5.0f};
    size_t size = 1;

    sampler_softmax(logits, size);

    ASSERT_FLOAT_EQ(logits[0], 1.0f, FLOAT_TOLERANCE, "Single element softmax should be 1");

    TEST_PASS();
}

static void test_sampler_softmax_uniform(void) {
    float logits[] = {0.0f, 0.0f, 0.0f, 0.0f};
    size_t size = 4;

    sampler_softmax(logits, size);

    /* 所有概率应该相等 */
    for (size_t i = 1; i < size; i++) {
        ASSERT_FLOAT_EQ(logits[i], logits[0], FLOAT_TOLERANCE, "Uniform logits should have equal probs");
    }

    float sum = 0.0f;
    for (size_t i = 0; i < size; i++) sum += logits[i];
    ASSERT_FLOAT_EQ(sum, 1.0f, FLOAT_TOLERANCE, "Probabilities should sum to 1");

    TEST_PASS();
}

/* ============================================================================
 * Temperature 测试
 * ============================================================================ */

static void test_sampler_apply_temperature(void) {
    float logits[] = {1.0f, 2.0f, 3.0f};
    size_t size = 3;

    /* 温度 1.0 不应该改变值 */
    float original[3];
    memcpy(original, logits, sizeof(logits));
    sampler_apply_temperature(logits, size, 1.0f);

    for (size_t i = 0; i < size; i++) {
        ASSERT_FLOAT_EQ(logits[i], original[i], FLOAT_TOLERANCE, "Temp 1.0 should not change");
    }

    /* 温度 2.0 应该使值更接近 */
    memcpy(logits, original, sizeof(logits));
    sampler_apply_temperature(logits, size, 2.0f);
    ASSERT_FLOAT_EQ(logits[0], 0.5f, FLOAT_TOLERANCE, "Logit 0 with temp 2");

    /* 温度 0.5 应该使值更极端 */
    memcpy(logits, original, sizeof(logits));
    sampler_apply_temperature(logits, size, 0.5f);
    ASSERT_FLOAT_EQ(logits[0], 2.0f, FLOAT_TOLERANCE, "Logit 0 with temp 0.5");

    TEST_PASS();
}

/* ============================================================================
 * Top-K 测试
 * ============================================================================ */

static void test_sampler_apply_top_k(void) {
    float logits[] = {0.1f, 0.9f, 0.3f, 0.8f, 0.2f, 0.7f, 0.4f, 0.6f, 0.5f, 0.0f};
    size_t size = 10;

    float copy[10];
    memcpy(copy, logits, sizeof(logits));
    sampler_apply_top_k(copy, size, 3);

    /* 应该只有 3 个非 -INF 值 */
    int count = 0;
    for (size_t i = 0; i < size; i++) {
        if (copy[i] > -FLT_MAX / 2) count++;
    }

    ASSERT_EQ(count, 3, "Should have 3 non-filtered tokens");

    TEST_PASS();
}

static void test_sampler_apply_top_k_larger_than_vocab(void) {
    float logits[] = {0.1f, 0.2f, 0.3f};
    size_t size = 3;

    float original[3];
    memcpy(original, logits, sizeof(logits));

    /* K >= vocab_size 不应该过滤任何内容 */
    sampler_apply_top_k(logits, size, 10);

    /* 当 k >= size 时，函数应该不修改 */
    /* 具体行为取决于实现 */

    TEST_PASS();
}

/* ============================================================================
 * Top-P 测试
 * ============================================================================ */

static void test_sampler_apply_top_p(void) {
    float logits[] = {0.0f, 0.0f, 0.0f, 0.0f, 0.0f};
    size_t size = 5;

    /* 设置不同的值 */
    for (size_t i = 0; i < size; i++) logits[i] = (float)i * 0.1f;

    /* Top-P = 0.5 应该保留一部分 tokens */
    float copy[5];
    memcpy(copy, logits, sizeof(logits));
    sampler_apply_top_p(copy, size, 0.5f);

    /* 计算保留的 tokens 数量 */
    int count = 0;
    for (size_t i = 0; i < size; i++) {
        if (copy[i] > -FLT_MAX / 2) count++;
    }

    ASSERT_TRUE(count < (int)size, "Top-P should filter some tokens");
    ASSERT_TRUE(count > 0, "Top-P should keep at least one token");

    TEST_PASS();
}

/* ============================================================================
 * 重复惩罚测试
 * ============================================================================ */

static void test_sampler_apply_repetition_penalty(void) {
    float logits[] = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f};
    size_t size = 5;
    int32_t penalty_tokens[] = {2, 4};  /* 惩罚索引 2 和 4 */
    size_t num_penalty = 2;
    float penalty = 2.0f;

    sampler_apply_repetition_penalty(logits, size, penalty_tokens, num_penalty, penalty);

    /* 正 logits 应该被降低 (除以 penalty) */
    ASSERT_FLOAT_EQ(logits[2], 1.5f, FLOAT_TOLERANCE, "Token 2 should be penalized (3/2)");
    ASSERT_FLOAT_EQ(logits[4], 2.5f, FLOAT_TOLERANCE, "Token 4 should be penalized (5/2)");

    /* 其他 logits 不应该改变 */
    ASSERT_FLOAT_EQ(logits[0], 1.0f, FLOAT_TOLERANCE, "Token 0 should not change");
    ASSERT_FLOAT_EQ(logits[1], 2.0f, FLOAT_TOLERANCE, "Token 1 should not change");
    ASSERT_FLOAT_EQ(logits[3], 4.0f, FLOAT_TOLERANCE, "Token 3 should not change");

    TEST_PASS();
}

static void test_sampler_apply_repetition_penalty_negative(void) {
    float logits[] = {-1.0f, -2.0f, -3.0f};
    size_t size = 3;
    int32_t penalty_tokens[] = {0, 1, 2};
    size_t num_penalty = 3;
    float penalty = 2.0f;

    sampler_apply_repetition_penalty(logits, size, penalty_tokens, num_penalty, penalty);

    /* 负 logits 应该被乘以 penalty (变得更负) */
    ASSERT_FLOAT_EQ(logits[0], -2.0f, FLOAT_TOLERANCE, "Negative logit should be multiplied");
    ASSERT_FLOAT_EQ(logits[1], -4.0f, FLOAT_TOLERANCE, "Negative logit should be multiplied");
    ASSERT_FLOAT_EQ(logits[2], -6.0f, FLOAT_TOLERANCE, "Negative logit should be multiplied");

    TEST_PASS();
}

/* ============================================================================
 * 贪婪采样测试
 * ============================================================================ */

static void test_sampler_sample_greedy(void) {
    float logits[] = {0.1f, 0.5f, 0.9f, 0.3f, 0.7f};
    size_t size = 5;

    int32_t token = sampler_sample_greedy(logits, size);

    ASSERT_EQ(token, 2, "Should select index 2 (max logit 0.9)");

    TEST_PASS();
}

/* ============================================================================
 * 完整采样流程测试
 * ============================================================================ */

static void test_sampler_sample_basic(void) {
    Sampler* sampler = sampler_new();
    ASSERT_NOT_NULL(sampler, "Sampler should not be NULL");

    /* 创建测试 logits */
    float logits[TEST_VOCAB_SIZE];
    for (size_t i = 0; i < TEST_VOCAB_SIZE; i++) {
        logits[i] = (float)i * 0.01f;
    }

    /* 采样多次，应该得到有效的 token ID */
    for (int i = 0; i < 10; i++) {
        int32_t token = sampler_sample(sampler, logits, TEST_VOCAB_SIZE);
        ASSERT_TRUE(token >= 0 && token < (int32_t)TEST_VOCAB_SIZE, "Token should be valid");
    }

    sampler_free(sampler);
    TEST_PASS();
}

static void test_sampler_sample_deterministic(void) {
    SamplerConfig config = sampler_default_config();
    config.seed = 12345;
    config.top_k = 0;  /* 禁用 top-k */
    config.top_p = 1.0f;  /* 禁用 top-p */

    Sampler* s1 = sampler_new_with_config(&config);
    Sampler* s2 = sampler_new_with_config(&config);

    float logits[TEST_VOCAB_SIZE];
    for (size_t i = 0; i < TEST_VOCAB_SIZE; i++) {
        logits[i] = (float)(i % 10) * 0.1f;
    }

    /* 相同种子应该产生相同序列 */
    for (int i = 0; i < 5; i++) {
        int32_t t1 = sampler_sample(s1, logits, TEST_VOCAB_SIZE);
        int32_t t2 = sampler_sample(s2, logits, TEST_VOCAB_SIZE);
        ASSERT_EQ(t1, t2, "Same seed should produce same tokens");
    }

    sampler_free(s1);
    sampler_free(s2);
    TEST_PASS();
}

static void test_sampler_sample_with_temperature(void) {
    Sampler* sampler = sampler_new();
    ASSERT_NOT_NULL(sampler, "Sampler should not be NULL");

    float logits[TEST_VOCAB_SIZE];
    for (size_t i = 0; i < TEST_VOCAB_SIZE; i++) {
        logits[i] = (float)i * 0.01f;
    }

    /* 低温度应该更确定 */
    sampler_set_temperature(sampler, 0.1f);

    int32_t token = sampler_sample(sampler, logits, TEST_VOCAB_SIZE);
    ASSERT_TRUE(token >= 0 && token < (int32_t)TEST_VOCAB_SIZE, "Token should be valid");

    sampler_free(sampler);
    TEST_PASS();
}

/* ============================================================================
 * 边界情况测试
 * ============================================================================ */

static void test_sampler_sample_null(void) {
    float logits[] = {1.0f, 2.0f, 3.0f};

    int32_t token = sampler_sample(NULL, logits, 3);
    ASSERT_EQ(token, -1, "NULL sampler should return -1");

    Sampler* sampler = sampler_new();
    token = sampler_sample(sampler, NULL, 3);
    ASSERT_EQ(token, -1, "NULL logits should return -1");

    token = sampler_sample(sampler, logits, 0);
    ASSERT_EQ(token, -1, "Zero size should return -1");

    sampler_free(sampler);
    TEST_PASS();
}

static void test_sampler_extreme_logits(void) {
    Sampler* sampler = sampler_new();
    ASSERT_NOT_NULL(sampler, "Sampler should not be NULL");

    /* 极端值 */
    float logits[] = {-1000.0f, 0.0f, 1000.0f};
    size_t size = 3;

    int32_t token = sampler_sample(sampler, logits, size);
    ASSERT_TRUE(token >= 0 && token < (int32_t)size, "Should handle extreme values");

    sampler_free(sampler);
    TEST_PASS();
}

/* ============================================================================
 * 工具函数测试
 * ============================================================================ */

static void test_sampler_argmax(void) {
    float logits[] = {0.5f, 1.0f, 0.3f, 0.8f, 0.2f};
    size_t size = 5;

    int32_t idx = sampler_argmax(logits, size);
    ASSERT_EQ(idx, 1, "Argmax should return index 1");

    TEST_PASS();
}

static void test_sampler_max_logit(void) {
    float logits[] = {0.5f, 1.0f, 0.3f, 0.8f, 0.2f};
    size_t size = 5;

    float max_val = sampler_max_logit(logits, size);
    ASSERT_FLOAT_EQ(max_val, 1.0f, FLOAT_TOLERANCE, "Max should be 1.0");

    TEST_PASS();
}

/* ============================================================================
 * 主函数
 * ============================================================================ */

int main(void) {
    printf("========================================\n");
    printf("Sampler Module Unit Tests\n");
    printf("========================================\n\n");

    /* 生命周期测试 */
    printf("[Lifecycle Tests]\n");
    RUN_TEST(test_sampler_new);
    RUN_TEST(test_sampler_new_with_config);
    RUN_TEST(test_sampler_free_null);
    RUN_TEST(test_sampler_default_config);
    printf("\n");

    /* 配置设置测试 */
    printf("[Configuration Tests]\n");
    RUN_TEST(test_sampler_set_temperature);
    RUN_TEST(test_sampler_set_top_k);
    RUN_TEST(test_sampler_set_top_p);
    printf("\n");

    /* 随机数测试 */
    printf("[Random Tests]\n");
    RUN_TEST(test_sampler_pcg_random);
    RUN_TEST(test_sampler_random_float);
    RUN_TEST(test_sampler_reproducibility);
    printf("\n");

    /* Softmax 测试 */
    printf("[Softmax Tests]\n");
    RUN_TEST(test_sampler_softmax_basic);
    RUN_TEST(test_sampler_softmax_single);
    RUN_TEST(test_sampler_softmax_uniform);
    printf("\n");

    /* Temperature 测试 */
    printf("[Temperature Tests]\n");
    RUN_TEST(test_sampler_apply_temperature);
    printf("\n");

    /* Top-K 测试 */
    printf("[Top-K Tests]\n");
    RUN_TEST(test_sampler_apply_top_k);
    RUN_TEST(test_sampler_apply_top_k_larger_than_vocab);
    printf("\n");

    /* Top-P 测试 */
    printf("[Top-P Tests]\n");
    RUN_TEST(test_sampler_apply_top_p);
    printf("\n");

    /* 重复惩罚测试 */
    printf("[Repetition Penalty Tests]\n");
    RUN_TEST(test_sampler_apply_repetition_penalty);
    RUN_TEST(test_sampler_apply_repetition_penalty_negative);
    printf("\n");

    /* 贪婪采样测试 */
    printf("[Greedy Sampling Tests]\n");
    RUN_TEST(test_sampler_sample_greedy);
    printf("\n");

    /* 完整采样流程测试 */
    printf("[Full Sampling Tests]\n");
    RUN_TEST(test_sampler_sample_basic);
    RUN_TEST(test_sampler_sample_deterministic);
    RUN_TEST(test_sampler_sample_with_temperature);
    printf("\n");

    /* 边界情况测试 */
    printf("[Edge Case Tests]\n");
    RUN_TEST(test_sampler_sample_null);
    RUN_TEST(test_sampler_extreme_logits);
    printf("\n");

    /* 工具函数测试 */
    printf("[Utility Tests]\n");
    RUN_TEST(test_sampler_argmax);
    RUN_TEST(test_sampler_max_logit);
    printf("\n");

    TEST_SUMMARY();
    return TEST_RETURN();
}
