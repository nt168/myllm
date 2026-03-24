/**
 * @file model_types.h
 * @brief 模型通用类型定义 - 对应 phyllm/src/models/traits.rs
 */

#ifndef MYLLM_MODEL_TYPES_H
#define MYLLM_MODEL_TYPES_H

#include "tensor/dtype.h"
#include <stdint.h>
#include <stddef.h>
#include <stdbool.h>

#ifdef __cplusplus
extern "C" {
#endif

/* ============================================================================
 * 张量结构 (前向声明)
 * ============================================================================ */

/* 使用前向声明，实际定义在 tensor/tensor.h 中 */
typedef struct Tensor Tensor;

/* ============================================================================
 * 模型配置
 * ============================================================================ */

/**
 * @brief 加载后的模型配置 - 对应 phyllm LoadedConfig
 */
typedef struct {
    char model_type[64];            /**< 模型类型 */
    size_t hidden_dim;              /**< 隐藏维度 */
    size_t intermediate_dim;        /**< FFN中间维度 */
    size_t num_heads;               /**< 注意力头数 */
    size_t num_kv_heads;            /**< KV头数 (GQA) */
    size_t head_dim;                /**< 头维度 */
    size_t num_layers;              /**< 层数 */
    size_t vocab_size;              /**< 词表大小 */
    size_t max_seq_len;             /**< 最大序列长度 */
    size_t max_position_embeddings; /**< 最大位置嵌入 */
    double rope_theta;              /**< RoPE theta */
    float norm_eps;                 /**< 归一化epsilon */
    DType dtype;                    /**< 数据类型 */
    bool tie_word_embeddings;       /**< 是否共享权重 */
} LoadedConfig;

/**
 * @brief 初始化默认配置
 */
static inline void loaded_config_init(LoadedConfig* config) {
    config->model_type[0] = '\0';
    config->hidden_dim = 0;
    config->intermediate_dim = 0;
    config->num_heads = 0;
    config->num_kv_heads = 0;
    config->head_dim = 0;
    config->num_layers = 0;
    config->vocab_size = 0;
    config->max_seq_len = 2048;
    config->max_position_embeddings = 2048;
    config->rope_theta = 10000.0;
    config->norm_eps = 1e-6f;
    config->dtype = DTYPE_F32;
    config->tie_word_embeddings = false;
}

/* ============================================================================
 * 采样参数
 * ============================================================================ */

/**
 * @brief 采样参数 - 对应 phyllm SamplingParams
 */
typedef struct {
    uint32_t max_tokens;            /**< 最大生成token数 */
    float temperature;              /**< 温度 (0=贪婪) */
    float top_p;                    /**< Top-p核采样 */
    uint32_t top_k;                 /**< Top-k采样 */
    float repetition_penalty;       /**< 重复惩罚 */
    float presence_penalty;         /**< 存在惩罚 */
    bool do_sample;                 /**< 是否采样 */
} SamplingParams;

/**
 * @brief 初始化默认采样参数
 */
static inline void sampling_params_init(SamplingParams* params) {
    params->max_tokens = 256;
    params->temperature = 1.0f;
    params->top_p = 0.9f;
    params->top_k = 50;
    params->repetition_penalty = 1.0f;
    params->presence_penalty = 0.0f;
    params->do_sample = true;
}

/* ============================================================================
 * 消息结构
 * ============================================================================ */

/**
 * @brief 消息角色
 */
typedef enum {
    MESSAGE_ROLE_SYSTEM = 0,
    MESSAGE_ROLE_USER = 1,
    MESSAGE_ROLE_ASSISTANT = 2,
} MessageRole;

/**
 * @brief 聊天消息
 */
typedef struct {
    MessageRole role;
    char* content;
} Message;

/* ============================================================================
 * 错误处理
 * ============================================================================ */

/**
 * @brief 错误码
 */
typedef enum {
    MYLLM_OK = 0,
    MYLLM_ERROR_NULL_POINTER = -1,
    MYLLM_ERROR_INVALID_SHAPE = -2,
    MYLLM_ERROR_INVALID_DTYPE = -3,
    MYLLM_ERROR_OUT_OF_MEMORY = -4,
    MYLLM_ERROR_INDEX_OUT_OF_BOUNDS = -5,
    MYLLM_ERROR_CACHE_OVERFLOW = -6,
    MYLLM_ERROR_CACHE_EMPTY = -7,
    MYLLM_ERROR_INVALID_INPUT = -8,
    MYLLM_ERROR_NOT_IMPLEMENTED = -9,
    MYLLM_ERROR_INTERNAL = -10,
} MyLLMError;

/**
 * @brief 获取错误描述
 */
static inline const char* myllm_error_str(MyLLMError err) {
    switch (err) {
        case MYLLM_OK: return "Success";
        case MYLLM_ERROR_NULL_POINTER: return "Null pointer";
        case MYLLM_ERROR_INVALID_SHAPE: return "Invalid shape";
        case MYLLM_ERROR_INVALID_DTYPE: return "Invalid dtype";
        case MYLLM_ERROR_OUT_OF_MEMORY: return "Out of memory";
        case MYLLM_ERROR_INDEX_OUT_OF_BOUNDS: return "Index out of bounds";
        case MYLLM_ERROR_CACHE_OVERFLOW: return "Cache overflow";
        case MYLLM_ERROR_CACHE_EMPTY: return "Cache empty";
        case MYLLM_ERROR_INVALID_INPUT: return "Invalid input";
        case MYLLM_ERROR_NOT_IMPLEMENTED: return "Not implemented";
        case MYLLM_ERROR_INTERNAL: return "Internal error";
        default: return "Unknown error";
    }
}

/* ============================================================================
 * 结果结构
 * ============================================================================ */

/**
 * @brief 生成结果
 */
typedef struct {
    int32_t* tokens;            /**< 生成的token */
    size_t num_tokens;          /**< token数量 */
    char* text;                 /**< 生成的文本 */
    size_t num_generated;       /**< 生成的token数 (不含prompt) */
    double prefill_time_ms;     /**< Prefill时间 (毫秒) */
    double avg_decode_time_ms;  /**< 平均Decode时间 (毫秒) */
    int stopped_by;             /**< 停止原因: 0=EOS, 1=MaxTokens, 2=StopToken */
} GenerationResult;

#ifdef __cplusplus
}
#endif

#endif /* MYLLM_MODEL_TYPES_H */
