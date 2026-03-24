/**
 * @file model.c
 * @brief Qwen3 模型实现
 *
 * Qwen3 语言模型:
 * - Token 嵌入层
 * - 堆叠的 Transformer 块 (带 Grouped-Query Attention)
 * - 最终 RMSNorm
 * - 可选的 lm_head (可与 embed_tokens 共享权重)
 *
 * Qwen3 架构特点:
 * - Pre-norm Transformer
 * - GQA (Grouped Query Attention)
 * - RoPE 位置编码
 * - SwiGLU 激活
 * - RMSNorm 归一化
 */

#include "models/qwen3/model.h"
#include "ops/rope.h"
#include "tensor/shape.h"
#include <stdlib.h>
#include <string.h>
#include <math.h>

/* ============================================================================
 * Qwen3 注意力层实现
 * ============================================================================ */

Qwen3Attention* qwen3_attention_new(
    size_t hidden_dim,
    size_t num_heads,
    size_t num_kv_heads,
    size_t head_dim,
    double rope_theta,
    bool use_qk_norm
) {
    Qwen3Attention* attn = (Qwen3Attention*)malloc(sizeof(Qwen3Attention));
    if (!attn) return NULL;

    attn->hidden_dim = hidden_dim;
    attn->num_heads = num_heads;
    attn->num_kv_heads = num_kv_heads;
    attn->head_dim = head_dim;
    attn->scale = 1.0f / sqrtf((float)head_dim);
    attn->rope_theta = rope_theta;
    attn->use_qk_norm = use_qk_norm;

    /* 投影层 (Qwen3 默认无 bias) */
    size_t q_dim = num_heads * head_dim;
    size_t kv_dim = num_kv_heads * head_dim;

    attn->q_proj = linear_new(hidden_dim, q_dim, false);
    attn->k_proj = linear_new(hidden_dim, kv_dim, false);
    attn->v_proj = linear_new(hidden_dim, kv_dim, false);
    attn->o_proj = linear_new(q_dim, hidden_dim, false);

    if (!attn->q_proj || !attn->k_proj || !attn->v_proj || !attn->o_proj) {
        if (attn->q_proj) linear_free(attn->q_proj);
        if (attn->k_proj) linear_free(attn->k_proj);
        if (attn->v_proj) linear_free(attn->v_proj);
        if (attn->o_proj) linear_free(attn->o_proj);
        free(attn);
        return NULL;
    }

    /* Q/K Norm (可选) */
    if (use_qk_norm) {
        attn->q_norm = rmsnorm_new(head_dim, 1e-6f);
        attn->k_norm = rmsnorm_new(head_dim, 1e-6f);
        if (!attn->q_norm || !attn->k_norm) {
            if (attn->q_norm) rmsnorm_free(attn->q_norm);
            if (attn->k_norm) rmsnorm_free(attn->k_norm);
            linear_free(attn->q_proj);
            linear_free(attn->k_proj);
            linear_free(attn->v_proj);
            linear_free(attn->o_proj);
            free(attn);
            return NULL;
        }
    } else {
        attn->q_norm = NULL;
        attn->k_norm = NULL;
    }

    return attn;
}

void qwen3_attention_free(Qwen3Attention* attn) {
    if (!attn) return;

    if (attn->q_proj) linear_free(attn->q_proj);
    if (attn->k_proj) linear_free(attn->k_proj);
    if (attn->v_proj) linear_free(attn->v_proj);
    if (attn->o_proj) linear_free(attn->o_proj);
    if (attn->q_norm) rmsnorm_free(attn->q_norm);
    if (attn->k_norm) rmsnorm_free(attn->k_norm);
    free(attn);
}

int qwen3_attention_compute_kv_with_rope(
    Qwen3Attention* attn,
    const Tensor* x,
    const size_t* positions,
    size_t num_positions,
    Tensor** k_out,
    Tensor** v_out
) {
    if (!attn || !x || !k_out || !v_out) return MYLLM_ERROR_NULL_POINTER;

    /* K, V 投影 */
    Tensor* k = linear_forward(attn->k_proj, x);
    Tensor* v = linear_forward(attn->v_proj, x);

    if (!k || !v) {
        if (k) tensor_free(k);
        if (v) tensor_free(v);
        return MYLLM_ERROR_OUT_OF_MEMORY;
    }

    /* 可选的 K Norm */
    if (attn->k_norm) {
        Tensor* k_normed = rmsnorm_forward(attn->k_norm, k);
        tensor_free(k);
        k = k_normed;
    }

    /* 应用 RoPE 到 K */
    Tensor* k_rope = ops_rope(k, positions, num_positions, attn->rope_theta);
    tensor_free(k);

    *k_out = k_rope;
    *v_out = v;

    return MYLLM_OK;
}

/* ============================================================================
 * Qwen3 Transformer 块实现
 * ============================================================================ */

Qwen3TransformerBlock* qwen3_block_new(const Qwen3Config* config) {
    if (!config) return NULL;

    return qwen3_block_new_with_head_dim(
        config->base.hidden_size,
        config->base.num_attention_heads,
        config->base.num_key_value_heads,
        config->base.head_dim,
        config->base.intermediate_size,
        config->base.rms_norm_eps,
        config->base.rope_theta
    );
}

Qwen3TransformerBlock* qwen3_block_new_with_head_dim(
    size_t hidden_dim,
    size_t num_heads,
    size_t num_kv_heads,
    size_t head_dim,
    size_t intermediate_dim,
    float norm_eps,
    double rope_theta
) {
    Qwen3TransformerBlock* block = (Qwen3TransformerBlock*)malloc(sizeof(Qwen3TransformerBlock));
    if (!block) return NULL;

    /* 注意力层 */
    block->attention = qwen3_attention_new(
        hidden_dim, num_heads, num_kv_heads, head_dim, rope_theta, false
    );
    if (!block->attention) {
        free(block);
        return NULL;
    }

    /* MLP (SwiGLU) */
    block->mlp = mlp_new(hidden_dim, intermediate_dim);
    if (!block->mlp) {
        qwen3_attention_free(block->attention);
        free(block);
        return NULL;
    }

    /* 归一化层 */
    block->input_norm = rmsnorm_new(hidden_dim, norm_eps);
    block->post_attention_norm = rmsnorm_new(hidden_dim, norm_eps);
    if (!block->input_norm || !block->post_attention_norm) {
        if (block->input_norm) rmsnorm_free(block->input_norm);
        if (block->post_attention_norm) rmsnorm_free(block->post_attention_norm);
        qwen3_attention_free(block->attention);
        mlp_free(block->mlp);
        free(block);
        return NULL;
    }

    return block;
}

void qwen3_block_free(Qwen3TransformerBlock* block) {
    if (!block) return;

    if (block->attention) qwen3_attention_free(block->attention);
    if (block->mlp) mlp_free(block->mlp);
    if (block->input_norm) rmsnorm_free(block->input_norm);
    if (block->post_attention_norm) rmsnorm_free(block->post_attention_norm);
    free(block);
}

Qwen3Attention* qwen3_block_attention(Qwen3TransformerBlock* block) {
    return block ? block->attention : NULL;
}

RMSNorm* qwen3_block_input_norm(Qwen3TransformerBlock* block) {
    return block ? block->input_norm : NULL;
}

RMSNorm* qwen3_block_post_attention_norm(Qwen3TransformerBlock* block) {
    return block ? block->post_attention_norm : NULL;
}

MLP* qwen3_block_mlp(Qwen3TransformerBlock* block) {
    return block ? block->mlp : NULL;
}

/* ============================================================================
 * Qwen3 模型实现
 * ============================================================================ */

Qwen3Model* qwen3_model_new(const Qwen3Config* config) {
    if (!config) return NULL;

    Qwen3Model* model = (Qwen3Model*)malloc(sizeof(Qwen3Model));
    if (!model) return NULL;

    memcpy(&model->config, config, sizeof(Qwen3Config));
    model->has_cache = false;
    model->kv_caches = NULL;

    /* Token 嵌入层 */
    model->embed_tokens = embedding_new(config->base.vocab_size, config->base.hidden_size);
    if (!model->embed_tokens) {
        free(model);
        return NULL;
    }

    /* Transformer 层 */
    model->layers = (Qwen3TransformerBlock**)malloc(
        config->base.num_hidden_layers * sizeof(Qwen3TransformerBlock*)
    );
    if (!model->layers) {
        embedding_free(model->embed_tokens);
        free(model);
        return NULL;
    }

    for (size_t i = 0; i < config->base.num_hidden_layers; i++) {
        model->layers[i] = qwen3_block_new(config);
        if (!model->layers[i]) {
            for (size_t j = 0; j < i; j++) {
                qwen3_block_free(model->layers[j]);
            }
            free(model->layers);
            embedding_free(model->embed_tokens);
            free(model);
            return NULL;
        }
    }
    model->num_layers = config->base.num_hidden_layers;

    /* 最终归一化层 */
    model->norm = rmsnorm_new(config->base.hidden_size, config->base.rms_norm_eps);
    if (!model->norm) {
        for (size_t i = 0; i < model->num_layers; i++) {
            qwen3_block_free(model->layers[i]);
        }
        free(model->layers);
        embedding_free(model->embed_tokens);
        free(model);
        return NULL;
    }

    /* lm_head (Qwen3 默认不共享权重) */
    if (config->base.tie_word_embeddings) {
        model->lm_head = NULL;
    } else {
        model->lm_head = linear_new(config->base.hidden_size, config->base.vocab_size, false);
        if (!model->lm_head) {
            rmsnorm_free(model->norm);
            for (size_t i = 0; i < model->num_layers; i++) {
                qwen3_block_free(model->layers[i]);
            }
            free(model->layers);
            embedding_free(model->embed_tokens);
            free(model);
            return NULL;
        }
    }

    return model;
}

Qwen3Model* qwen3_model_new_with_cache(const Qwen3Config* config, size_t batch_size) {
    Qwen3Model* model = qwen3_model_new(config);
    if (!model) return NULL;

    /* 创建 KV 缓存 */
    model->kv_caches = (KVCache**)malloc(model->num_layers * sizeof(KVCache*));
    if (!model->kv_caches) {
        qwen3_model_free(model);
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
            qwen3_model_free(model);
            return NULL;
        }
    }
    model->has_cache = true;

    return model;
}

Qwen3Model* qwen3_model_from_loaded_config(const LoadedConfig* config) {
    if (!config) return NULL;

    Qwen3Config qwen3_config;
    loaded_config_to_qwen3(config, &qwen3_config);
    qwen3_config_set_defaults(&qwen3_config);

    return qwen3_model_new(&qwen3_config);
}

Qwen3Model* qwen3_model_from_loaded_config_with_cache(const LoadedConfig* config, size_t batch_size) {
    if (!config) return NULL;

    Qwen3Config qwen3_config;
    loaded_config_to_qwen3(config, &qwen3_config);
    qwen3_config_set_defaults(&qwen3_config);

    return qwen3_model_new_with_cache(&qwen3_config, batch_size);
}

void qwen3_model_free(Qwen3Model* model) {
    if (!model) return;

    if (model->embed_tokens) embedding_free(model->embed_tokens);

    if (model->layers) {
        for (size_t i = 0; i < model->num_layers; i++) {
            if (model->layers[i]) qwen3_block_free(model->layers[i]);
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

size_t qwen3_model_num_layers(const Qwen3Model* model) {
    return model ? model->num_layers : 0;
}

size_t qwen3_model_vocab_size(const Qwen3Model* model) {
    return model ? model->config.base.vocab_size : 0;
}

bool qwen3_model_has_cache(const Qwen3Model* model) {
    return model ? model->has_cache : false;
}

size_t qwen3_model_cache_len(const Qwen3Model* model) {
    if (!model || !model->has_cache || !model->kv_caches) return 0;
    return kv_cache_len(model->kv_caches[0]);
}

const char* qwen3_model_name(const Qwen3Model* model) {
    (void)model;
    return "Qwen3Model";
}

bool qwen3_model_tie_word_embeddings(const Qwen3Model* model) {
    return model ? model->config.base.tie_word_embeddings : false;
}

void qwen3_model_reset_cache(Qwen3Model* model) {
    if (!model || !model->kv_caches) return;

    for (size_t i = 0; i < model->num_layers; i++) {
        if (model->kv_caches[i]) {
            kv_cache_reset(model->kv_caches[i]);
        }
    }
}

Qwen3TransformerBlock* qwen3_model_layer(Qwen3Model* model, size_t idx) {
    if (!model || idx >= model->num_layers) return NULL;
    return model->layers[idx];
}

Embedding* qwen3_model_embed_tokens(Qwen3Model* model) {
    return model ? model->embed_tokens : NULL;
}

RMSNorm* qwen3_model_norm(Qwen3Model* model) {
    return model ? model->norm : NULL;
}

Linear* qwen3_model_lm_head(Qwen3Model* model) {
    return model ? model->lm_head : NULL;
}

/* ============================================================================
 * 模型推理
 * ============================================================================ */

Tensor* qwen3_model_prefill(Qwen3Model* model, const int32_t* tokens, size_t num_tokens) {
    if (!model || !tokens || num_tokens == 0) return NULL;
    if (!model->has_cache) return NULL;

    /* 创建位置数组 */
    size_t* positions = (size_t*)malloc(num_tokens * sizeof(size_t));
    if (!positions) return NULL;
    for (size_t i = 0; i < num_tokens; i++) {
        positions[i] = i;
    }

    /* 创建输入张量 */
    Tensor input;
    memset(&input, 0, sizeof(Tensor));
    size_t dims[2] = {1, num_tokens};
    input.shape = shape_new(dims, 2);
    input.dtype = DTYPE_I32;
    input.data = (void*)tokens;

    /* Token 嵌入 */
    Tensor* hidden = embedding_forward(model->embed_tokens, &input);
    if (!hidden) {
        free(positions);
        return NULL;
    }

    /* 逐层处理 */
    for (size_t layer_idx = 0; layer_idx < model->num_layers; layer_idx++) {
        Qwen3TransformerBlock* layer = model->layers[layer_idx];
        KVCache* cache = model->kv_caches[layer_idx];

        /* 输入归一化 */
        Tensor* normed = rmsnorm_forward(layer->input_norm, hidden);
        if (!normed) {
            tensor_free(hidden);
            free(positions);
            return NULL;
        }

        /* 计算 K, V 并应用 RoPE */
        Tensor *k_all, *v_all;
        int ret = qwen3_attention_compute_kv_with_rope(
            layer->attention, normed, positions, num_tokens, &k_all, &v_all
        );
        if (ret != MYLLM_OK) {
            tensor_free(normed);
            tensor_free(hidden);
            free(positions);
            return NULL;
        }

        /* 批量追加到缓存 */
        kv_cache_append_batch(cache, k_all, v_all);

        /* 获取完整的 K, V */
        Tensor *k_cached, *v_cached;
        kv_cache_get(cache, &k_cached, &v_cached);

        /* 注意力计算 */
        Tensor* attn_out = qwen3_attention_forward_with_kv(
            layer->attention, normed, k_cached, v_cached, positions, num_tokens
        );

        /* 释放临时张量 */
        tensor_free(k_all);
        tensor_free(v_all);
        tensor_free(k_cached);
        tensor_free(v_cached);
        tensor_free(normed);

        /* Residual + Attention output */
        /* TODO: 实现 tensor_add */

        /* 注意力后归一化 */
        Tensor* residual = hidden;
        normed = rmsnorm_forward(layer->post_attention_norm, hidden);

        /* MLP */
        Tensor* mlp_out = mlp_forward(layer->mlp, normed);

        tensor_free(normed);
        tensor_free(hidden);
        hidden = mlp_out;
    }

    /* 最终归一化 */
    Tensor* normed_final = rmsnorm_forward(model->norm, hidden);

    /* 提取最后一个 token */
    /* TODO: 实现 tensor_slice */

    /* lm_head */
    Tensor* logits = NULL;
    if (model->lm_head) {
        logits = linear_forward(model->lm_head, normed_final);
    } else {
        /* 共享权重: 使用 embed_tokens.weight 的转置 */
        /* TODO: 实现共享权重的矩阵乘法 */
    }

    tensor_free(hidden);
    tensor_free(normed_final);
    free(positions);

    return logits;
}

Tensor* qwen3_model_decode_step(Qwen3Model* model, int32_t token, size_t position) {
    if (!model) return NULL;
    if (!model->has_cache) return NULL;

    /* 创建输入张量 */
    Tensor input;
    memset(&input, 0, sizeof(Tensor));
    size_t dims[2] = {1, 1};
    input.shape = shape_new(dims, 2);
    input.dtype = DTYPE_I32;
    input.data = &token;

    size_t positions[1] = {position};

    /* Token 嵌入 */
    Tensor* hidden = embedding_forward(model->embed_tokens, &input);
    if (!hidden) return NULL;

    /* 逐层处理 */
    for (size_t layer_idx = 0; layer_idx < model->num_layers; layer_idx++) {
        Qwen3TransformerBlock* layer = model->layers[layer_idx];
        KVCache* cache = model->kv_caches[layer_idx];

        /* 输入归一化 */
        Tensor* normed = rmsnorm_forward(layer->input_norm, hidden);

        /* 计算 K, V 并应用 RoPE */
        Tensor *k_new, *v_new;
        qwen3_attention_compute_kv_with_rope(layer->attention, normed, positions, 1, &k_new, &v_new);

        /* 追加到缓存 */
        kv_cache_append(cache, k_new, v_new);

        /* 获取完整的 K, V */
        Tensor *k_cached, *v_cached;
        kv_cache_get(cache, &k_cached, &v_cached);

        /* 注意力计算 */
        Tensor* attn_out = qwen3_attention_forward_with_kv(
            layer->attention, normed, k_cached, v_cached, positions, 1
        );

        tensor_free(k_new);
        tensor_free(v_new);
        tensor_free(k_cached);
        tensor_free(v_cached);
        tensor_free(normed);

        /* 注意力后归一化 */
        Tensor* residual = hidden;
        normed = rmsnorm_forward(layer->post_attention_norm, hidden);
        Tensor* mlp_out = mlp_forward(layer->mlp, normed);

        tensor_free(normed);
        tensor_free(hidden);
        hidden = mlp_out;
    }

    /* 最终归一化 */
    Tensor* normed_final = rmsnorm_forward(model->norm, hidden);

    /* lm_head */
    Tensor* logits = NULL;
    if (model->lm_head) {
        logits = linear_forward(model->lm_head, normed_final);
    }

    tensor_free(hidden);
    tensor_free(normed_final);

    return logits;
}

Tensor* qwen3_model_forward(Qwen3Model* model, const Tensor* input_ids) {
    if (!model || !input_ids) return NULL;

    /* Token 嵌入 */
    Tensor* hidden = embedding_forward(model->embed_tokens, input_ids);
    if (!hidden) return NULL;

    /* 逐层处理 */
    for (size_t i = 0; i < model->num_layers; i++) {
        /* 获取序列长度 */
        size_t seq_len = 1;
        if (hidden->shape.ndim >= 2) {
            seq_len = hidden->shape.dims[hidden->shape.ndim - 2];
        }

        /* 创建位置数组 */
        size_t* positions = (size_t*)malloc(seq_len * sizeof(size_t));
        if (!positions) {
            tensor_free(hidden);
            return NULL;
        }
        for (size_t j = 0; j < seq_len; j++) {
            positions[j] = j;
        }

        Tensor* output = qwen3_block_forward_with_positions(model->layers[i], hidden, positions, seq_len);
        free(positions);

        if (!output) {
            tensor_free(hidden);
            return NULL;
        }

        tensor_free(hidden);
        hidden = output;
    }

    /* 最终归一化 */
    Tensor* normed = rmsnorm_forward(model->norm, hidden);

    /* lm_head */
    Tensor* logits = NULL;
    if (model->lm_head) {
        logits = linear_forward(model->lm_head, normed);
    }

    tensor_free(hidden);
    tensor_free(normed);

    return logits;
}
