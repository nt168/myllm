/**
 * @file config.c
 * @brief LLaMA 模型配置实现
 */

#include "models/llama/config.h"
#include <string.h>
#include <stdlib.h>

void llama_config_init(LlamaConfig* config) {
    if (!config) return;

    config->hidden_size = 0;
    config->intermediate_size = 0;
    config->num_attention_heads = 0;
    config->num_key_value_heads = 0;
    config->head_dim = 0;
    config->num_hidden_layers = 0;
    config->vocab_size = 0;
    config->max_position_embeddings = 2048;
    config->rope_theta = 10000.0;
    config->rms_norm_eps = 1e-6f;
    config->torch_dtype = DTYPE_F32;
    config->tie_word_embeddings = false;
}

void llama_config_set_defaults(LlamaConfig* config) {
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

void llama_config_to_loaded(const LlamaConfig* src, LoadedConfig* dst) {
    if (!src || !dst) return;

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

void loaded_config_to_llama(const LoadedConfig* src, LlamaConfig* dst) {
    if (!src || !dst) return;

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
