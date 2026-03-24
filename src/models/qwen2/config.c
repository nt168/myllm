/**
 * @file config.c
 * @brief Qwen2 模型配置实现
 */

#include "models/qwen2/config.h"
#include <string.h>
#include <stdlib.h>

void qwen2_config_init(Qwen2Config* config) {
    if (!config) return;

    llama_config_init(&config->base);

    /* Qwen2 默认值 */
    config->base.rope_theta = 1000000.0;
    config->base.tie_word_embeddings = true;
    config->base.torch_dtype = DTYPE_BF16;

    config->sliding_window = 0;
    config->use_sliding_window = false;
    config->use_bias = true;
}

void qwen2_config_set_defaults(Qwen2Config* config) {
    if (!config) return;

    llama_config_set_defaults(&config->base);

    if (config->sliding_window == 0) {
        config->use_sliding_window = false;
    }
}

void loaded_config_to_qwen2(const LoadedConfig* src, Qwen2Config* dst) {
    if (!src || !dst) return;

    loaded_config_to_llama(src, &dst->base);

    dst->sliding_window = 0;
    dst->use_sliding_window = false;
    dst->use_bias = true;
}

void qwen2_config_to_loaded(const Qwen2Config* src, LoadedConfig* dst) {
    if (!src || !dst) return;

    llama_config_to_loaded(&src->base, dst);
    strcpy(dst->model_type, "qwen2");
}
