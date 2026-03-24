/**
 * @file config.h
 * @brief Qwen3 模型配置 - 对应 phyllm/src/models/qwen3/config.rs
 */

#ifndef MYLLM_QWEN3_CONFIG_H
#define MYLLM_QWEN3_CONFIG_H

#include "../model_types.h"
#include "../llama/config.h"

#ifdef __cplusplus
extern "C" {
#endif

/**
 * @brief Qwen3 模型配置
 *
 * Qwen3 配置与 Qwen2 类似，默认值略有不同
 */
typedef struct {
    LlamaConfig base;                  /**< 基础 Llama 配置 */
    size_t sliding_window;             /**< 滑动窗口大小 */
    bool use_sliding_window;           /**< 是否使用滑动窗口 */
} Qwen3Config;

/**
 * @brief 初始化 Qwen3 配置
 */
static inline void qwen3_config_init(Qwen3Config* config) {
    if (!config) return;

    llama_config_init(&config->base);

    /* Qwen3 默认值 */
    config->base.rope_theta = 1000000.0;
    config->base.tie_word_embeddings = false;  /* Qwen3 默认不共享权重 */
    config->base.torch_dtype = DTYPE_BF16;
    config->base.max_position_embeddings = 40960;  /* Qwen3 更长的上下文 */

    config->sliding_window = 0;
    config->use_sliding_window = false;
}

/**
 * @brief 从 LoadedConfig 转换为 Qwen3Config
 */
static inline void loaded_config_to_qwen3(const LoadedConfig* src, Qwen3Config* dst) {
    if (!src || !dst) return;

    loaded_config_to_llama(src, &dst->base);

    dst->sliding_window = 0;
    dst->use_sliding_window = false;
}

/**
 * @brief 从 Qwen3Config 转换为 LoadedConfig
 */
static inline void qwen3_config_to_loaded(const Qwen3Config* src, LoadedConfig* dst) {
    if (!src || !dst) return;

    llama_config_to_loaded(&src->base, dst);
    strcpy(dst->model_type, "qwen3");
}

#ifdef __cplusplus
}
#endif

#endif /* MYLLM_QWEN3_CONFIG_H */
