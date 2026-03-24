/**
 * @file model.c
 * @brief Qwen2 模型实现
 */

#include "models/qwen2/model.h"
#include <stdlib.h>
#include <string.h>
#include <math.h>

/* ============================================================================
 * Qwen2 注意力层实现
 * ============================================================================ */

Qwen2Attention* qwen2_attention_new(
    size_t hidden_dim,
    size_t num_heads,
    size_t num_kv_heads,
    size_t head_dim,
    double rope_theta,
    size_t sliding_window,
    bool use_bias
) {
    Qwen2Attention* attn = (Qwen2Attention*)malloc(sizeof(Qwen2Attention));
    if (!attn) return NULL;

    attn->hidden_dim = hidden_dim;
    attn->num_heads = num_heads;
    attn->num_kv_heads = num_kv_heads;
    attn->head_dim = head_dim;
    attn->scale = 1.0f / sqrtf((float)head_dim);
    attn->rope_theta = rope_theta;
    attn->sliding_window = sliding_window;
    attn->use_sliding_window = (sliding_window > 0);

    /* Qwen2 使用 bias */
    size_t q_dim = num_heads * head_dim;
    size_t kv_dim = num_kv_heads * head_dim;

    attn->q_proj = linear_new(hidden_dim, q_dim, use_bias);
    attn->k_proj = linear_new(hidden_dim, kv_dim, use_bias);
    attn->v_proj = linear_new(hidden_dim, kv_dim, use_bias);
    attn->o_proj = linear_new(q_dim, hidden_dim, use_bias);

    if (!attn->q_proj || !attn->k_proj || !attn->v_proj || !attn->o_proj) {
        if (attn->q_proj) linear_free(attn->q_proj);
        if (attn->k_proj) linear_free(attn->k_proj);
        if (attn->v_proj) linear_free(attn->v_proj);
        if (attn->o_proj) linear_free(attn->o_proj);
        free(attn);
        return NULL;
    }

    return attn;
}

void qwen2_attention_free(Qwen2Attention* attn) {
    if (!attn) return;

    if (attn->q_proj) linear_free(attn->q_proj);
    if (attn->k_proj) linear_free(attn->k_proj);
    if (attn->v_proj) linear_free(attn->v_proj);
    if (attn->o_proj) linear_free(attn->o_proj);
    free(attn);
}

/* ============================================================================
 * Qwen2 Transformer 块实现
 * ============================================================================ */

Qwen2TransformerBlock* qwen2_block_new(const Qwen2Config* config) {
    if (!config) return NULL;

    Qwen2TransformerBlock* block = (Qwen2TransformerBlock*)malloc(sizeof(Qwen2TransformerBlock));
    if (!block) return NULL;

    /* 注意力层 */
    block->attention = qwen2_attention_new(
        config->base.hidden_size,
        config->base.num_attention_heads,
        config->base.num_key_value_heads,
        config->base.head_dim,
        config->base.rope_theta,
        config->sliding_window,
        config->use_bias
    );
    if (!block->attention) {
        free(block);
        return NULL;
    }

    /* MLP */
    block->mlp = mlp_new(config->base.hidden_size, config->base.intermediate_size);
    if (!block->mlp) {
        qwen2_attention_free(block->attention);
        free(block);
        return NULL;
    }

    /* 归一化层 */
    block->input_norm = rmsnorm_new(config->base.hidden_size, config->base.rms_norm_eps);
    block->post_attention_norm = rmsnorm_new(config->base.hidden_size, config->base.rms_norm_eps);
    if (!block->input_norm || !block->post_attention_norm) {
        if (block->input_norm) rmsnorm_free(block->input_norm);
        if (block->post_attention_norm) rmsnorm_free(block->post_attention_norm);
        qwen2_attention_free(block->attention);
        mlp_free(block->mlp);
        free(block);
        return NULL;
    }

    return block;
}

void qwen2_block_free(Qwen2TransformerBlock* block) {
    if (!block) return;

    if (block->attention) qwen2_attention_free(block->attention);
    if (block->mlp) mlp_free(block->mlp);
    if (block->input_norm) rmsnorm_free(block->input_norm);
    if (block->post_attention_norm) rmsnorm_free(block->post_attention_norm);
    free(block);
}

/* ============================================================================
 * Qwen2 模型实现
 * ============================================================================ */

Qwen2Model* qwen2_model_new(const Qwen2Config* config) {
    if (!config) return NULL;

    Qwen2Model* model = (Qwen2Model*)malloc(sizeof(Qwen2Model));
    if (!model) return NULL;

    memcpy(&model->config, config, sizeof(Qwen2Config));
    model->has_cache = false;
    model->kv_caches = NULL;

    /* 嵌入层 */
    model->embed_tokens = embedding_new(config->base.vocab_size, config->base.hidden_size);
    if (!model->embed_tokens) {
        free(model);
        return NULL;
    }

    /* Transformer 层 */
    model->layers = (Qwen2TransformerBlock**)malloc(
        config->base.num_hidden_layers * sizeof(Qwen2TransformerBlock*)
    );
    if (!model->layers) {
        embedding_free(model->embed_tokens);
        free(model);
        return NULL;
    }

    for (size_t i = 0; i < config->base.num_hidden_layers; i++) {
        model->layers[i] = qwen2_block_new(config);
        if (!model->layers[i]) {
            for (size_t j = 0; j < i; j++) {
                qwen2_block_free(model->layers[j]);
            }
            free(model->layers);
            embedding_free(model->embed_tokens);
            free(model);
            return NULL;
        }
    }
    model->num_layers = config->base.num_hidden_layers;

    /* 最终归一化 */
    model->norm = rmsnorm_new(config->base.hidden_size, config->base.rms_norm_eps);
    if (!model->norm) {
        for (size_t i = 0; i < model->num_layers; i++) {
            qwen2_block_free(model->layers[i]);
        }
        free(model->layers);
        embedding_free(model->embed_tokens);
        free(model);
        return NULL;
    }

    /* lm_head (Qwen2 默认共享权重) */
    if (config->base.tie_word_embeddings) {
        model->lm_head = NULL;
    } else {
        model->lm_head = linear_new(config->base.hidden_size, config->base.vocab_size, false);
        if (!model->lm_head) {
            rmsnorm_free(model->norm);
            for (size_t i = 0; i < model->num_layers; i++) {
                qwen2_block_free(model->layers[i]);
            }
            free(model->layers);
            embedding_free(model->embed_tokens);
            free(model);
            return NULL;
        }
    }

    return model;
}

Qwen2Model* qwen2_model_new_with_cache(const Qwen2Config* config, size_t batch_size) {
    Qwen2Model* model = qwen2_model_new(config);
    if (!model) return NULL;

    /* 创建 KV 缓存 */
    model->kv_caches = (KVCache**)malloc(model->num_layers * sizeof(KVCache*));
    if (!model->kv_caches) {
        qwen2_model_free(model);
        return NULL;
    }

    for (size_t i = 0; i < model->num_layers; i++) {
        model->kv_caches[i] = kv_cache_new(
            config->base.max_position_embeddings,
            config->base.num_key_value_heads,
            config->base.head_dim,
            batch_size,
            config->base.torch_dtype
        );
        if (!model->kv_caches[i]) {
            for (size_t j = 0; j < i; j++) {
                kv_cache_free(model->kv_caches[j]);
            }
            free(model->kv_caches);
            qwen2_model_free(model);
            return NULL;
        }
    }
    model->has_cache = true;

    return model;
}

Qwen2Model* qwen2_model_from_loaded_config(const LoadedConfig* config) {
    if (!config) return NULL;

    Qwen2Config qwen2_config;
    loaded_config_to_qwen2(config, &qwen2_config);
    qwen2_config_set_defaults(&qwen2_config);

    return qwen2_model_new(&qwen2_config);
}

Qwen2Model* qwen2_model_from_loaded_config_with_cache(const LoadedConfig* config, size_t batch_size) {
    if (!config) return NULL;

    Qwen2Config qwen2_config;
    loaded_config_to_qwen2(config, &qwen2_config);
    qwen2_config_set_defaults(&qwen2_config);

    return qwen2_model_new_with_cache(&qwen2_config, batch_size);
}

void qwen2_model_free(Qwen2Model* model) {
    if (!model) return;

    if (model->embed_tokens) embedding_free(model->embed_tokens);

    if (model->layers) {
        for (size_t i = 0; i < model->num_layers; i++) {
            if (model->layers[i]) qwen2_block_free(model->layers[i]);
        }
        free(model->layers);
    }

    if (model->norm) rmsnorm_free(model->norm);
    if (model->lm_head) linear_free(model->lm_head);

    if (model->kv_caches) {
        for (size_t i = 0; i < model->num_layers; i++) {
            if (model->kv_caches[i]) kv_cache_free(model->kv_caches[i]);
        }
        free(model->kv_caches);
    }

    free(model);
}

size_t qwen2_model_num_layers(const Qwen2Model* model) {
    return model ? model->num_layers : 0;
}

size_t qwen2_model_vocab_size(const Qwen2Model* model) {
    return model ? model->config.base.vocab_size : 0;
}

bool qwen2_model_has_cache(const Qwen2Model* model) {
    return model ? model->has_cache : false;
}

size_t qwen2_model_cache_len(const Qwen2Model* model) {
    if (!model || !model->has_cache || !model->kv_caches) return 0;
    return kv_cache_len(model->kv_caches[0]);
}

const char* qwen2_model_name(const Qwen2Model* model) {
    (void)model;
    return "Qwen2Model";
}

void qwen2_model_reset_cache(Qwen2Model* model) {
    if (!model || !model->kv_caches) return;

    for (size_t i = 0; i < model->num_layers; i++) {
        if (model->kv_caches[i]) {
            kv_cache_reset(model->kv_caches[i]);
        }
    }
}

/* Prefill 和 Decode 实现与 LLaMA 类似，使用 Qwen2 特定的注意力 */
Tensor* qwen2_model_prefill(Qwen2Model* model, const int32_t* tokens, size_t num_tokens) {
    /* 实现与 llama_model_prefill 类似 */
    (void)model;
    (void)tokens;
    (void)num_tokens;
    return NULL;  /* TODO: 实现 */
}

Tensor* qwen2_model_decode_step(Qwen2Model* model, int32_t token, size_t position) {
    /* 实现与 llama_model_decode_step 类似 */
    (void)model;
    (void)token;
    (void)position;
    return NULL;  /* TODO: 实现 */
}

Tensor* qwen2_model_forward(Qwen2Model* model, const Tensor* input_ids) {
    /* 实现与 llama_model_forward 类似 */
    (void)model;
    (void)input_ids;
    return NULL;  /* TODO: 实现 */
}
