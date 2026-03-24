/**
 * @file config.h
 * @brief LLaMA 模型配置 - 对应 phyllm/src/models/llama/config.rs
 */

#ifndef MYLLM_LLAMA_CONFIG_H
#define MYLLM_LLAMA_CONFIG_H

#include "../model_types.h"
#include <string.h>

#ifdef __cplusplus
extern "C" {
#endif

/**
 * @brief LLaMA 模型配置
 */
typedef struct {
    size_t hidden_size;             /**< 隐藏维度 */
    size_t intermediate_size;       /**< FFN中间维度 */
    size_t num_attention_heads;     /**< 注意力头数 */
    size_t num_hidden_layers;       /**< 隐藏层数 */
    size_t vocab_size;              /**< 词表大小 */
    size_t num_key_value_heads;     /**< KV头数 (GQA) */
    size_t head_dim;                /**< 头维度 */
    double rope_theta;              /**< RoPE theta */
    size_t max_position_embeddings; /**< 最大位置嵌入 */
    float rms_norm_eps;             /**< RMSNorm epsilon */
    DType torch_dtype;              /**< 数据类型 */
    bool tie_word_embeddings;       /**< 是否共享权重 */
} LlamaConfig;

/**
 * @brief 初始化 LLaMA 默认配置
 */
static inline void llama_config_init(LlamaConfig* config) {
    config->hidden_size = 4096;
    config->intermediate_size = 11008;
    config->num_attention_heads = 32;
    config->num_hidden_layers = 32;
    config->vocab_size = 32000;
    config->num_key_value_heads = 32;
    config->head_dim = 128;
    config->rope_theta = 10000.0;
    config->max_position_embeddings = 2048;
    config->rms_norm_eps = 1e-6f;
    config->torch_dtype = DTYPE_F16;
    config->tie_word_embeddings = false;
}

/**
 * @brief 设置 LLaMA 配置默认值 (根据已设置的字段推断其他字段)
 */
static inline void llama_config_set_defaults(LlamaConfig* config) {
    if (!config) return;

    /* 如果 num_key_value_heads 未设置, 默认为 num_attention_heads */
    if (config->num_key_value_heads == 0) {
        config->num_key_value_heads = config->num_attention_heads;
    }

    /* 如果 head_dim 未设置, 默认为 hidden_size / num_attention_heads */
    if (config->head_dim == 0 && config->num_attention_heads > 0) {
        config->head_dim = config->hidden_size / config->num_attention_heads;
    }
}

/**
 * @brief 从 LlamaConfig 转换为 LoadedConfig
 */
static inline void llama_config_to_loaded(const LlamaConfig* src, LoadedConfig* dst) {
    dst->model_type[0] = '\0';
    strcpy(dst->model_type, "llama");
    dst->hidden_dim = src->hidden_size;
    dst->intermediate_dim = src->intermediate_size;
    dst->num_heads = src->num_attention_heads;
    dst->num_kv_heads = src->num_key_value_heads;
    dst->head_dim = src->head_dim;
    dst->num_layers = src->num_hidden_layers;
    dst->vocab_size = src->vocab_size;
    dst->max_seq_len = src->max_position_embeddings;
    dst->max_position_embeddings = src->max_position_embeddings;
    dst->rope_theta = src->rope_theta;
    dst->norm_eps = src->rms_norm_eps;
    dst->dtype = src->torch_dtype;
    dst->tie_word_embeddings = src->tie_word_embeddings;
}

/**
 * @brief 从 LoadedConfig 转换为 LlamaConfig
 */
static inline void loaded_config_to_llama(const LoadedConfig* src, LlamaConfig* dst) {
    dst->hidden_size = src->hidden_dim;
    dst->intermediate_size = src->intermediate_dim;
    dst->num_attention_heads = src->num_heads;
    dst->num_key_value_heads = src->num_kv_heads;
    dst->head_dim = src->head_dim;
    dst->num_hidden_layers = src->num_layers;
    dst->vocab_size = src->vocab_size;
    dst->max_position_embeddings = src->max_position_embeddings;
    dst->rope_theta = src->rope_theta;
    dst->rms_norm_eps = src->norm_eps;
    dst->torch_dtype = src->dtype;
    dst->tie_word_embeddings = src->tie_word_embeddings;
}

#ifdef __cplusplus
}
#endif

#endif /* MYLLM_LLAMA_CONFIG_H */
