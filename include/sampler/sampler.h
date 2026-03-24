/**
 * @file sampler.h
 * @brief 采样器模块
 *
 * 实现各种采样策略用于文本生成:
 * - Temperature (温度缩放)
 * - Top-K 采样
 * - Top-P (Nucleus) 采样
 * - Repetition Penalty (重复惩罚)
 * - Softmax 计算和随机采样
 */

#ifndef MYLLM_SAMPLER_H
#define MYLLM_SAMPLER_H

#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>
#include <float.h>

#ifdef __cplusplus
extern "C" {
#endif

/* ============================================================================
 * 常量定义
 * ============================================================================ */

/** 默认温度 */
#define SAMPLER_DEFAULT_TEMPERATURE 1.0f

/** 默认 Top-K */
#define SAMPLER_DEFAULT_TOP_K 40

/** 默认 Top-P */
#define SAMPLER_DEFAULT_TOP_P 0.9f

/** 默认重复惩罚 */
#define SAMPLER_DEFAULT_REPETITION_PENALTY 1.0f

/** 最小温度值 (避免除零) */
#define SAMPLER_MIN_TEMPERATURE 1e-7f

/* ============================================================================
 * 采样器配置
 * ============================================================================ */

/**
 * @brief 采样器配置
 */
typedef struct SamplerConfig {
    float temperature;              /**< 温度参数 (>0, 默认 1.0) */
    int32_t top_k;                  /**< Top-K 采样 (0=禁用, 默认 40) */
    float top_p;                    /**< Top-P 采样 (0-1, 默认 0.9) */
    float repetition_penalty;       /**< 重复惩罚 (1.0=禁用, 默认 1.0) */
    int32_t* penalty_tokens;        /**< 惩罚 token 列表 */
    size_t num_penalty_tokens;      /**< 惩罚 token 数量 */
    uint64_t seed;                  /**< 随机种子 */
} SamplerConfig;

/* ============================================================================
 * 采样器结构
 * ============================================================================ */

/**
 * @brief 采样器结构 (不透明指针)
 */
typedef struct Sampler Sampler;

/* ============================================================================
 * 生命周期管理
 * ============================================================================ */

/**
 * @brief 创建默认配置的采样器
 */
Sampler* sampler_new(void);

/**
 * @brief 使用指定配置创建采样器
 */
Sampler* sampler_new_with_config(const SamplerConfig* config);

/**
 * @brief 释放采样器
 */
void sampler_free(Sampler* sampler);

/* ============================================================================
 * 配置
 * ============================================================================ */

/**
 * @brief 获取默认配置
 */
SamplerConfig sampler_default_config(void);

/**
 * @brief 设置温度
 */
void sampler_set_temperature(Sampler* sampler, float temperature);

/**
 * @brief 设置 Top-K
 */
void sampler_set_top_k(Sampler* sampler, int32_t top_k);

/**
 * @brief 设置 Top-P
 */
void sampler_set_top_p(Sampler* sampler, float top_p);

/**
 * @brief 设置重复惩罚
 */
void sampler_set_repetition_penalty(Sampler* sampler, float penalty);

/**
 * @brief 设置随机种子
 */
void sampler_set_seed(Sampler* sampler, uint64_t seed);

/* ============================================================================
 * 采样函数
 * ============================================================================ */

/**
 * @brief 从 logits 中采样一个 token
 *
 * 执行完整采样流程:
 * 1. 应用温度缩放
 * 2. 应用重复惩罚 (如果有)
 * 3. Top-K 过滤
 * 4. Top-P 过滤
 * 5. Softmax
 * 6. 随机采样
 *
 * @param sampler 采样器
 * @param logits logits 数组 [vocab_size]
 * @param vocab_size 词表大小
 * @return 采样得到的 token ID，失败返回 -1
 */
int32_t sampler_sample(Sampler* sampler, const float* logits, size_t vocab_size);

/**
 * @brief 贪婪解码 - 选择概率最高的 token
 *
 * @param logits logits 数组
 * @param vocab_size 词表大小
 * @return 概率最高的 token ID
 */
int32_t sampler_sample_greedy(const float* logits, size_t vocab_size);

/**
 * @brief 带温度的采样 (无 Top-K/Top-P)
 *
 * @param sampler 采样器
 * @param logits logits 数组
 * @param vocab_size 词表大小
 * @return 采样得到的 token ID
 */
int32_t sampler_sample_temperature(
    Sampler* sampler,
    const float* logits,
    size_t vocab_size
);

/* ============================================================================
 * 采样组件 (可单独使用)
 * ============================================================================ */

/**
 * @brief 应用温度缩放到 logits (原地修改)
 *
 * scaled_logits = logits / temperature
 *
 * @param logits logits 数组 (会被修改)
 * @param size 数组大小
 * @param temperature 温度值
 */
void sampler_apply_temperature(float* logits, size_t size, float temperature);

/**
 * @brief 应用重复惩罚 (原地修改)
 *
 * 降低已出现 token 的概率
 *
 * @param logits logits 数组 (会被修改)
 * @param size 数组大小
 * @param token_ids 要惩罚的 token ID 数组
 * @param num_tokens 惩罚 token 数量
 * @param penalty 惩罚系数 (>1 降低概率, <1 增加概率)
 */
void sampler_apply_repetition_penalty(
    float* logits,
    size_t size,
    const int32_t* token_ids,
    size_t num_tokens,
    float penalty
);

/**
 * @brief 应用 Top-K 过滤 (原地修改)
 *
 * 保留概率最高的 K 个 token，其余设为 -INF
 *
 * @param logits logits 数组 (会被修改)
 * @param size 数组大小
 * @param k 保留的 token 数量
 */
void sampler_apply_top_k(float* logits, size_t size, int32_t k);

/**
 * @brief 应用 Top-P (Nucleus) 过滤 (原地修改)
 *
 * 保留累积概率达到 P 的最小 token 集合
 *
 * @param logits logits 数组 (会被修改)
 * @param size 数组大小
 * @param p 累积概率阈值 (0-1)
 */
void sampler_apply_top_p(float* logits, size_t size, float p);

/**
 * @brief 计算 Softmax (原地修改)
 *
 * @param logits logits 数组 (会被修改为概率)
 * @param size 数组大小
 */
void sampler_softmax(float* logits, size_t size);

/**
 * @brief 从概率分布中随机采样
 *
 * @param probs 概率数组 (必须和为 1)
 * @param size 数组大小
 * @param rng_state 随机状态指针 (用于确定性采样)
 * @return 采样得到的索引
 */
int32_t sampler_random_sample(const float* probs, size_t size, uint64_t* rng_state);

/* ============================================================================
 * 工具函数
 * ============================================================================ */

/**
 * @brief 获取 logits 中的最大值
 */
float sampler_max_logit(const float* logits, size_t size);

/**
 * @brief 获取 logits 中最大值的索引 (Argmax)
 */
int32_t sampler_argmax(const float* logits, size_t size);

/**
 * @brief 计算 logits 的 log-sum-exp
 */
float sampler_logsumexp(const float* logits, size_t size);

/**
 * @brief 简单的 PCG 随机数生成器
 */
uint64_t sampler_pcg_random(uint64_t* state);

/**
 * @brief 生成 [0, 1) 范围的随机浮点数
 */
float sampler_random_float(uint64_t* state);

#ifdef __cplusplus
}
#endif

#endif /* MYLLM_SAMPLER_H */
