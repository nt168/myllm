/**
 * @file config.c
 * @brief Qwen3 模型配置实现
 */

#include "models/qwen3/config.h"
#include <string.h>
#include <stdlib.h>

void qwen3_config_init(Qwen3Config* config) {
    if (!config) return;

    llama_config_init(&config->base);

    /* Qwen3 默认值 */
    config->base.rope_theta = 1000000.0;
    config->base.tie_word_embeddings = false;
    config->base.torch_dtype = DTYPE_BF16;
    config->base.max_position_embeddings = 40960;

    config->sliding_window = 0;
    config->use_sliding_window = false;
}

void qwen3_config_set_defaults(Qwen3Config* config) {
    if (!config) return;

    llama_config_set_defaults(&config->base);

    if (config->sliding_window == 0) {
        config->use_sliding_window = false;
    }
}

void loaded_config_to_qwen3(const LoadedConfig* src, Qwen3Config* dst) {
    if (!src || !dst) return;

    loaded_config_to_llama(src, &dst->base);

    dst->sliding_window = 0;
    dst->use_sliding_window = false;
}

void qwen3_config_to_loaded(const Qwen3Config* src, LoadedConfig* dst) {
    if (!src || !dst) return;

    llama_config_to_loaded(&src->base, dst);
    strcpy(dst->model_type, "qwen3");
}
