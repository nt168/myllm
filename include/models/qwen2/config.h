/**
 * @file config.h
 * @brief Qwen2 模型配置 - 对应 phyllm/src/models/qwen2/config.rs
 */

#ifndef MYLLM_QWEN2_CONFIG_H
#define MYLLM_QWEN2_CONFIG_H

#include "../model_types.h"
#include "../llama/config.h"

#ifdef __cplusplus
extern "C" {
#endif

/**
 * @brief Qwen2 模型配置
 *
 * Qwen2 配置类似于 LlamaConfig，但有一些特定字段:
 * - sliding_window: 滑动窗口大小
 * - use_bias: Qwen2 在注意力投影中使用 bias=True
 */
typedef struct {
    LlamaConfig base;                  /**< 基础 Llama 配置 */
    size_t sliding_window;             /**< 滑动窗口大小 (0 表示不使用) */
    bool use_sliding_window;           /**< 是否使用滑动窗口 */
    bool use_bias;                     /**< 注意力投影是否使用 bias (Qwen2 默认 true) */
} Qwen2Config;

/**
 * @brief 初始化 Qwen2 配置
 */
static inline void qwen2_config_init(Qwen2Config* config) {
    if (!config) return;

    llama_config_init(&config->base);

    /* Qwen2 默认值 */
    config->base.rope_theta = 1000000.0;  /* Qwen2 使用较大的 rope_theta */
    config->base.tie_word_embeddings = true;  /* Qwen2 默认共享权重 */
    config->base.torch_dtype = DTYPE_BF16;    /* Qwen2 默认 BF16 */

    config->sliding_window = 0;
    config->use_sliding_window = false;
    config->use_bias = true;  /* Qwen2 特有: 使用 bias */
}

/**
 * @brief 设置 Qwen2 配置默认值
 */
static inline void qwen2_config_set_defaults(Qwen2Config* config) {
    if (!config) return;

    llama_config_set_defaults(&config->base);

    /* Qwen2 特定默认值 */
    if (config->sliding_window == 0) {
        config->use_sliding_window = false;
    }
}

/**
 * @brief 从 LoadedConfig 转换为 Qwen2Config
 */
static inline void loaded_config_to_qwen2(const LoadedConfig* src, Qwen2Config* dst) {
    if (!src || !dst) return;

    loaded_config_to_llama(src, &dst->base);

    /* Qwen2 特定字段 (从 extra 中获取) */
    dst->sliding_window = 0;
    dst->use_sliding_window = false;
    dst->use_bias = true;
}

/**
 * @brief 从 Qwen2Config 转换为 LoadedConfig
 */
static inline void qwen2_config_to_loaded(const Qwen2Config* src, LoadedConfig* dst) {
    if (!src || !dst) return;

    llama_config_to_loaded(&src->base, dst);
    strcpy(dst->model_type, "qwen2");
}

#ifdef __cplusplus
}
#endif

#endif /* MYLLM_QWEN2_CONFIG_H */
