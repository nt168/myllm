/**
 * @file model.c
 * @brief LLaMA 模型实现
 */

#include "models/llama/model.h"
#include "tensor/shape.h"
#include "kv/kv.h"  /* 使用独立的 KV 缓存模块 */
#include "ops/ops.h"  /* 使用 ops_add 实现残差连接 */
#include <stdlib.h>
#include <string.h>
#include <math.h>

/* ============================================================================
 * 嵌入层实现
 * ============================================================================ */

Embedding* embedding_new(size_t num_embeddings, size_t embedding_dim) {
    Embedding* layer = (Embedding*)malloc(sizeof(Embedding));
    if (!layer) return NULL;

    layer->num_embeddings = num_embeddings;
    layer->embedding_dim = embedding_dim;

    /* 分配权重: [num_embeddings, embedding_dim] */
    layer->weight = (Tensor*)malloc(sizeof(Tensor));
    if (!layer->weight) {
        free(layer);
        return NULL;
    }
    memset(layer->weight, 0, sizeof(Tensor));

    /* 使用 shape_new 创建形状 */
    size_t weight_dims[2] = {num_embeddings, embedding_dim};
    layer->weight->shape = shape_new(weight_dims, 2);
    layer->weight->strides[0] = embedding_dim;
    layer->weight->strides[1] = 1;
    layer->weight->dtype = DTYPE_F32;
    layer->weight->offset = 0;
    layer->weight->device.type = DEVICE_CPU;
    layer->weight->device.id = 0;
    layer->weight->owns_data = true;
    layer->weight->data = calloc(num_embeddings * embedding_dim, sizeof(float));

    if (!layer->weight->data) {
        free(layer->weight);
        free(layer);
        return NULL;
    }

    return layer;
}

void embedding_free(Embedding* layer) {
    if (!layer) return;

    if (layer->weight) {
        if (layer->weight->data) free(layer->weight->data);
        free(layer->weight);
    }
    free(layer);
}

Tensor* embedding_forward(Embedding* layer, const Tensor* input_ids) {
    if (!layer || !input_ids) return NULL;
    if (input_ids->dtype != DTYPE_I32) return NULL;

    /* input_ids: [batch, seq_len] */
    size_t ndim = input_ids->shape.ndim;
    size_t batch = 1, seq_len = 1;
    if (ndim == 1) {
        seq_len = input_ids->shape.dims[0];
    } else if (ndim == 2) {
        batch = input_ids->shape.dims[0];
        seq_len = input_ids->shape.dims[1];
    }

    /* 创建输出张量: [batch, seq_len, embedding_dim] */
    Tensor* output = (Tensor*)malloc(sizeof(Tensor));
    if (!output) return NULL;
    memset(output, 0, sizeof(Tensor));

    size_t out_ndim = ndim + 1;
    if (ndim == 1) {
        size_t out_dims[2] = {seq_len, layer->embedding_dim};
        output->shape = shape_new(out_dims, 2);
        output->strides[0] = layer->embedding_dim;
        output->strides[1] = 1;
    } else {
        size_t out_dims[3] = {batch, seq_len, layer->embedding_dim};
        output->shape = shape_new(out_dims, 3);
        output->strides[0] = seq_len * layer->embedding_dim;
        output->strides[1] = layer->embedding_dim;
        output->strides[2] = 1;
    }

    size_t output_numel = batch * seq_len * layer->embedding_dim;
    output->dtype = DTYPE_F32;
    output->offset = 0;
    output->device.type = DEVICE_CPU;
    output->device.id = 0;
    output->owns_data = true;
    output->data = malloc(output_numel * sizeof(float));

    if (!output->data) {
        free(output);
        return NULL;
    }

    /* 查找嵌入 */
    const int32_t* ids = (const int32_t*)input_ids->data;
    float* out = (float*)output->data;
    const float* weight = (const float*)layer->weight->data;

    for (size_t b = 0; b < batch; b++) {
        for (size_t s = 0; s < seq_len; s++) {
            int32_t id;
            if (ndim == 1) {
                id = ids[s];
            } else {
                id = ids[b * seq_len + s];
            }

            /* 检查 id 范围 */
            if (id < 0 || (size_t)id >= layer->num_embeddings) {
                id = 0;  /* 使用 0 作为 fallback */
            }

            size_t out_idx = (ndim == 1) ? s * layer->embedding_dim :
                                          (b * seq_len + s) * layer->embedding_dim;
            size_t weight_idx = id * layer->embedding_dim;
            memcpy(&out[out_idx], &weight[weight_idx], layer->embedding_dim * sizeof(float));
        }
    }

    return output;
}

Tensor* embedding_weight(Embedding* layer) {
    return layer ? layer->weight : NULL;
}

/* ============================================================================
 * LLaMA 模型实现
 * ============================================================================ */

LlamaModel* llama_model_new(const LlamaConfig* config) {
    if (!config) return NULL;

    LlamaModel* model = (LlamaModel*)malloc(sizeof(LlamaModel));
    if (!model) return NULL;

    memcpy(&model->config, config, sizeof(LlamaConfig));
    model->has_cache = false;
    model->kv_caches = NULL;

    /* 创建嵌入层 */
    model->embed_tokens = embedding_new(config->vocab_size, config->hidden_size);
    if (!model->embed_tokens) {
        free(model);
        return NULL;
    }

    /* 创建 Transformer 层 */
    model->layers = (LlamaTransformerBlock**)malloc(config->num_hidden_layers * sizeof(LlamaTransformerBlock*));
    if (!model->layers) {
        embedding_free(model->embed_tokens);
        free(model);
        return NULL;
    }

    for (size_t i = 0; i < config->num_hidden_layers; i++) {
        model->layers[i] = llama_block_new(
            config->hidden_size,
            config->num_attention_heads,
            config->num_key_value_heads,
            config->head_dim,
            config->intermediate_size,
            config->rms_norm_eps,
            config->rope_theta
        );
        if (!model->layers[i]) {
            for (size_t j = 0; j < i; j++) {
                llama_block_free(model->layers[j]);
            }
            free(model->layers);
            embedding_free(model->embed_tokens);
            free(model);
            return NULL;
        }
    }
    model->num_layers = config->num_hidden_layers;

    /* 创建最终归一化层 */
    model->norm = rmsnorm_new(config->hidden_size, config->rms_norm_eps);
    if (!model->norm) {
        for (size_t i = 0; i < model->num_layers; i++) {
            llama_block_free(model->layers[i]);
        }
        free(model->layers);
        embedding_free(model->embed_tokens);
        free(model);
        return NULL;
    }

    /* 创建 lm_head (如果不共享权重) */
    if (config->tie_word_embeddings) {
        model->lm_head = NULL;
    } else {
        model->lm_head = linear_new(config->hidden_size, config->vocab_size, false);
        if (!model->lm_head) {
            rmsnorm_free(model->norm);
            for (size_t i = 0; i < model->num_layers; i++) {
                llama_block_free(model->layers[i]);
            }
            free(model->layers);
            embedding_free(model->embed_tokens);
            free(model);
            return NULL;
        }
    }

    return model;
}

LlamaModel* llama_model_new_with_cache(const LlamaConfig* config, size_t batch_size) {
    LlamaModel* model = llama_model_new(config);
    if (!model) return NULL;

    /* 创建 KV 缓存 */
    model->kv_caches = (KVCache**)malloc(model->num_layers * sizeof(KVCache*));
    if (!model->kv_caches) {
        llama_model_free(model);
        return NULL;
    }

    for (size_t i = 0; i < model->num_layers; i++) {
        model->kv_caches[i] = kv_cache_new(
            config->max_position_embeddings,
            config->num_key_value_heads,
            config->head_dim,
            batch_size,
            config->torch_dtype
        );
        if (!model->kv_caches[i]) {
            for (size_t j = 0; j < i; j++) {
                kv_cache_free(model->kv_caches[j]);
            }
            free(model->kv_caches);
            llama_model_free(model);
            return NULL;
        }
    }
    model->has_cache = true;

    return model;
}

LlamaModel* llama_model_from_loaded_config(const LoadedConfig* config) {
    if (!config) return NULL;

    LlamaConfig llama_config;
    loaded_config_to_llama(config, &llama_config);
    llama_config_set_defaults(&llama_config);

    return llama_model_new(&llama_config);
}

LlamaModel* llama_model_from_loaded_config_with_cache(const LoadedConfig* config, size_t batch_size) {
    if (!config) return NULL;

    LlamaConfig llama_config;
    loaded_config_to_llama(config, &llama_config);
    llama_config_set_defaults(&llama_config);

    return llama_model_new_with_cache(&llama_config, batch_size);
}

void llama_model_free(LlamaModel* model) {
    if (!model) return;

    if (model->embed_tokens) embedding_free(model->embed_tokens);

    if (model->layers) {
        for (size_t i = 0; i < model->num_layers; i++) {
            if (model->layers[i]) llama_block_free(model->layers[i]);
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

size_t llama_model_num_layers(const LlamaModel* model) {
    return model ? model->num_layers : 0;
}

size_t llama_model_vocab_size(const LlamaModel* model) {
    return model ? model->config.vocab_size : 0;
}

bool llama_model_has_cache(const LlamaModel* model) {
    return model ? model->has_cache : false;
}

size_t llama_model_cache_len(const LlamaModel* model) {
    if (!model || !model->has_cache || !model->kv_caches) return 0;
    return kv_cache_len(model->kv_caches[0]);
}

const char* llama_model_name(const LlamaModel* model) {
    (void)model;
    return "LlamaModel";
}

/* 辅助函数：释放张量 */
static void tensor_free_internal(Tensor* t) {
    if (!t) return;
    if (t->data) free(t->data);
    free(t);
}

Tensor* llama_model_prefill(LlamaModel* model, const int32_t* tokens, size_t num_tokens) {
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
    size_t input_dims[2] = {1, num_tokens};
    input.shape = shape_new(input_dims, 2);
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
        LlamaTransformerBlock* layer = model->layers[layer_idx];
        KVCache* cache = model->kv_caches[layer_idx];

        /* 输入归一化 */
        Tensor* normed = rmsnorm_forward(layer->input_norm, hidden);
        if (!normed) {
            tensor_free_internal(hidden);
            free(positions);
            return NULL;
        }

        /* 计算 K, V 并应用 RoPE */
        Tensor *k_all, *v_all;
        int ret = llama_attention_compute_kv_with_rope(
            layer->attention, normed, positions, num_tokens, &k_all, &v_all
        );
        if (ret != MYLLM_OK) {
            tensor_free_internal(normed);
            tensor_free_internal(hidden);
            free(positions);
            return NULL;
        }

        /* 批量追加到缓存 */
        kv_cache_append_batch(cache, k_all, v_all);

        /* 获取完整的 K, V */
        Tensor *k_cached, *v_cached;
        kv_cache_get(cache, &k_cached, &v_cached);

        /* 注意力计算 */
        Tensor* attn_out = llama_attention_forward_with_kv(
            layer->attention, normed, k_cached, v_cached, positions, num_tokens
        );

        /* 释放临时张量 */
        tensor_free_internal(k_all);
        tensor_free_internal(v_all);
        tensor_free_internal(k_cached);
        tensor_free_internal(v_cached);
        tensor_free_internal(normed);

        /* residual + attn_out */
        Tensor* residual_after_attn = ops_add(hidden, attn_out);
        tensor_free_internal(attn_out);
        tensor_free_internal(hidden);

        if (!residual_after_attn) {
            free(positions);
            return NULL;
        }

        /* 注意力后归一化 */
        normed = rmsnorm_forward(layer->post_attention_norm, residual_after_attn);

        /* MLP */
        Tensor* mlp_out = mlp_forward(layer->mlp, normed);
        tensor_free_internal(normed);

        if (!mlp_out) {
            tensor_free_internal(residual_after_attn);
            free(positions);
            return NULL;
        }

        /* residual + mlp_out */
        hidden = ops_add(residual_after_attn, mlp_out);
        tensor_free_internal(residual_after_attn);
        tensor_free_internal(mlp_out);

        if (!hidden) {
            free(positions);
            return NULL;
        }
    }

    /* 最终归一化 */
    Tensor* normed_final = rmsnorm_forward(model->norm, hidden);

    /* 提取最后一个 token 的 hidden state */
    /* TODO: 实现 tensor_slice */

    /* lm_head */
    Tensor* logits = NULL;
    if (model->lm_head) {
        logits = linear_forward(model->lm_head, normed_final);
    } else {
        /* 共享权重 */
        /* TODO: 实现 embed_tokens.weight 的转置和矩阵乘法 */
    }

    tensor_free_internal(hidden);
    tensor_free_internal(normed_final);
    free(positions);

    return logits;
}

Tensor* llama_model_decode_step(LlamaModel* model, int32_t token, size_t position) {
    if (!model) return NULL;
    if (!model->has_cache) return NULL;

    /* 创建输入张量 */
    Tensor input;
    memset(&input, 0, sizeof(Tensor));
    size_t input_dims[2] = {1, 1};
    input.shape = shape_new(input_dims, 2);
    input.dtype = DTYPE_I32;
    input.data = &token;

    size_t positions[1] = {position};

    /* Token 嵌入 */
    Tensor* hidden = embedding_forward(model->embed_tokens, &input);
    if (!hidden) return NULL;

    /* 逐层处理 */
    for (size_t layer_idx = 0; layer_idx < model->num_layers; layer_idx++) {
        LlamaTransformerBlock* layer = model->layers[layer_idx];
        KVCache* cache = model->kv_caches[layer_idx];

        /* 输入归一化 */
        Tensor* normed = rmsnorm_forward(layer->input_norm, hidden);

        /* 计算 K, V 并应用 RoPE */
        Tensor *k_new, *v_new;
        llama_attention_compute_kv_with_rope(layer->attention, normed, positions, 1, &k_new, &v_new);

        /* 追加到缓存 */
        kv_cache_append(cache, k_new, v_new);

        /* 获取完整的 K, V */
        Tensor *k_cached, *v_cached;
        kv_cache_get(cache, &k_cached, &v_cached);

        /* 注意力计算 */
        Tensor* attn_out = llama_attention_forward_with_kv(
            layer->attention, normed, k_cached, v_cached, positions, 1
        );

        /* 释放临时张量 */
        tensor_free_internal(k_new);
        tensor_free_internal(v_new);
        tensor_free_internal(k_cached);
        tensor_free_internal(v_cached);
        tensor_free_internal(normed);

        if (!attn_out) {
            tensor_free_internal(hidden);
            return NULL;
        }

        /* residual + attn_out */
        Tensor* residual_after_attn = ops_add(hidden, attn_out);
        tensor_free_internal(attn_out);
        tensor_free_internal(hidden);

        if (!residual_after_attn) {
            return NULL;
        }

        /* 注意力后归一化 */
        normed = rmsnorm_forward(layer->post_attention_norm, residual_after_attn);

        /* MLP */
        Tensor* mlp_out = mlp_forward(layer->mlp, normed);
        tensor_free_internal(normed);

        if (!mlp_out) {
            tensor_free_internal(residual_after_attn);
            return NULL;
        }

        /* residual + mlp_out */
        hidden = ops_add(residual_after_attn, mlp_out);
        tensor_free_internal(residual_after_attn);
        tensor_free_internal(mlp_out);

        if (!hidden) {
            return NULL;
        }
    }

    /* 最终归一化 */
    Tensor* normed_final = rmsnorm_forward(model->norm, hidden);

    /* lm_head */
    Tensor* logits = NULL;
    if (model->lm_head) {
        logits = linear_forward(model->lm_head, normed_final);
    }

    tensor_free_internal(hidden);
    tensor_free_internal(normed_final);

    return logits;
}

Tensor* llama_model_forward(LlamaModel* model, const Tensor* input_ids) {
    if (!model || !input_ids) return NULL;

    /* Token 嵌入 */
    Tensor* hidden = embedding_forward(model->embed_tokens, input_ids);
    if (!hidden) return NULL;

    /* 逐层处理 */
    for (size_t i = 0; i < model->num_layers; i++) {
        Tensor* output = llama_block_forward(model->layers[i], hidden);
        if (!output) {
            tensor_free_internal(hidden);
            return NULL;
        }

        tensor_free_internal(hidden);
        hidden = output;
    }

    /* 最终归一化 */
    Tensor* normed = rmsnorm_forward(model->norm, hidden);

    /* lm_head */
    Tensor* logits = NULL;
    if (model->lm_head) {
        logits = linear_forward(model->lm_head, normed);
    }

    tensor_free_internal(hidden);
    tensor_free_internal(normed);

    return logits;
}

void llama_model_reset_cache(LlamaModel* model) {
    if (!model || !model->kv_caches) return;

    for (size_t i = 0; i < model->num_layers; i++) {
        if (model->kv_caches[i]) {
            kv_cache_reset(model->kv_caches[i]);
        }
    }
}

LlamaTransformerBlock* llama_model_layer(LlamaModel* model, size_t idx) {
    if (!model || idx >= model->num_layers) return NULL;
    return model->layers[idx];
}

Embedding* llama_model_embed_tokens(LlamaModel* model) {
    return model ? model->embed_tokens : NULL;
}

RMSNorm* llama_model_norm(LlamaModel* model) {
    return model ? model->norm : NULL;
}

Linear* llama_model_lm_head(LlamaModel* model) {
    return model ? model->lm_head : NULL;
}
