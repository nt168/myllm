/**
 * @file attention.c
 * @brief LLaMA 注意力层实现
 */

#include "models/llama/attention.h"
#include "ops/rope.h"  /* 使用 RoPE 位置编码 */
#include "tensor/shape.h"
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <stdio.h>

/* 定义内部错误码 */
#define MYLLM_ERROR_NULL_POINTER -1
#define MYLLM_ERROR_INTERNAL -2

/* ============================================================================
 * 线性层实现
 * ============================================================================ */

Linear* linear_new(size_t in_features, size_t out_features, bool use_bias) {
    Linear* layer = (Linear*)malloc(sizeof(Linear));
    if (!layer) return NULL;

    layer->in_features = in_features;
    layer->out_features = out_features;
    layer->use_bias = use_bias;

    /* 分配权重张量 [out_features, in_features] */
    layer->weight = (Tensor*)malloc(sizeof(Tensor));
    if (!layer->weight) {
        free(layer);
        return NULL;
    }
    memset(layer->weight, 0, sizeof(Tensor));

    /* 使用 shape_new 创建形状 */
    size_t weight_dims[2] = {out_features, in_features};
    layer->weight->shape = shape_new(weight_dims, 2);

    /* 设置步幅 */
    layer->weight->strides[0] = in_features;
    layer->weight->strides[1] = 1;

    layer->weight->dtype = DTYPE_F32;
    layer->weight->offset = 0;
    layer->weight->device.type = DEVICE_CPU;
    layer->weight->device.id = 0;
    layer->weight->owns_data = true;

    /* 分配并初始化权重数据 (全零) */
    layer->weight->data = calloc(out_features * in_features, sizeof(float));
    if (!layer->weight->data) {
        free(layer->weight);
        free(layer);
        return NULL;
    }

    if (use_bias) {
        layer->bias = (Tensor*)malloc(sizeof(Tensor));
        if (!layer->bias) {
            free(layer->weight->data);
            free(layer->weight);
            free(layer);
            return NULL;
        }
        memset(layer->bias, 0, sizeof(Tensor));

        /* 使用 shape_new 创建形状 */
        size_t bias_dims[1] = {out_features};
        layer->bias->shape = shape_new(bias_dims, 1);

        /* 设置步幅 */
        layer->bias->strides[0] = 1;

        layer->bias->dtype = DTYPE_F32;
        layer->bias->offset = 0;
        layer->bias->device.type = DEVICE_CPU;
        layer->bias->device.id = 0;
        layer->bias->owns_data = true;

        layer->bias->data = calloc(out_features, sizeof(float));
        if (!layer->bias->data) {
            free(layer->bias);
            free(layer->weight->data);
            free(layer->weight);
            free(layer);
            return NULL;
        }
    } else {
        layer->bias = NULL;
    }

    return layer;
}

void linear_free(Linear* layer) {
    if (!layer) return;

    if (layer->weight) {
        if (layer->weight->data) free(layer->weight->data);
        free(layer->weight);
    }

    if (layer->bias) {
        if (layer->bias->data) free(layer->bias->data);
        free(layer->bias);
    }

    free(layer);
}

Tensor* linear_forward(Linear* layer, const Tensor* x) {
    if (!layer || !x) return NULL;

    /*
     * x: [batch, seq, in_features]
     * weight: [out_features, in_features]
     * output: [batch, seq, out_features]
     *
     * y = x @ W^T + b
     */

    size_t x_ndim = x->shape.ndim;
    if (x_ndim < 2) return NULL;

    size_t batch_size = 1;
    size_t seq_len = 1;
    size_t in_features;

    if (x_ndim == 2) {
        seq_len = x->shape.dims[0];
        in_features = x->shape.dims[1];
    } else if (x_ndim == 3) {
        batch_size = x->shape.dims[0];
        seq_len = x->shape.dims[1];
        in_features = x->shape.dims[2];
    } else {
        return NULL;
    }

    if (in_features != layer->in_features) return NULL;

    /* 创建输出张量 */
    Tensor* output = (Tensor*)malloc(sizeof(Tensor));
    if (!output) return NULL;
    memset(output, 0, sizeof(Tensor));

    if (x_ndim == 2) {
        size_t out_dims[2] = {seq_len, layer->out_features};
        output->shape = shape_new(out_dims, 2);
        output->strides[0] = layer->out_features;
        output->strides[1] = 1;
    } else {
        size_t out_dims[3] = {batch_size, seq_len, layer->out_features};
        output->shape = shape_new(out_dims, 3);
        output->strides[0] = seq_len * layer->out_features;
        output->strides[1] = layer->out_features;
        output->strides[2] = 1;
    }

    size_t output_numel = batch_size * seq_len * layer->out_features;
    output->dtype = x->dtype;
    output->offset = 0;
    output->device = x->device;
    output->owns_data = true;

    output->data = malloc(output_numel * dtype_size(output->dtype));
    if (!output->data) {
        free(output);
        return NULL;
    }

    /* 执行矩阵乘法 (简化版本, 实际应使用 BLAS) */
    float* x_data = (float*)x->data;
    float* w_data = (float*)layer->weight->data;
    float* o_data = (float*)output->data;
    float* b_data = layer->bias ? (float*)layer->bias->data : NULL;

    size_t M = batch_size * seq_len;
    size_t K = layer->in_features;
    size_t N = layer->out_features;

    for (size_t i = 0; i < M; i++) {
        for (size_t j = 0; j < N; j++) {
            float sum = b_data ? b_data[j] : 0.0f;
            for (size_t k = 0; k < K; k++) {
                sum += x_data[i * K + k] * w_data[j * K + k];
            }
            o_data[i * N + j] = sum;
        }
    }

    return output;
}

/* ============================================================================
 * GQA 注意力层实现
 * ============================================================================ */

GQAAttention* gqa_attention_new(
    size_t hidden_dim,
    size_t num_heads,
    size_t num_kv_heads,
    size_t head_dim,
    double rope_theta,
    bool use_q_norm,
    bool use_k_norm,
    bool use_bias
) {
    GQAAttention* attn = (GQAAttention*)malloc(sizeof(GQAAttention));
    if (!attn) return NULL;

    memset(attn, 0, sizeof(GQAAttention));

    attn->hidden_dim = hidden_dim;
    attn->num_heads = num_heads;
    attn->num_kv_heads = num_kv_heads;
    attn->head_dim = head_dim;
    attn->scale = 1.0f / sqrtf((float)head_dim);
    attn->rope_theta = rope_theta;
    attn->use_q_norm = use_q_norm;
    attn->use_k_norm = use_k_norm;

    /* 创建投影层 */
    size_t q_dim = num_heads * head_dim;
    size_t kv_dim = num_kv_heads * head_dim;

    attn->q_proj = linear_new(hidden_dim, q_dim, use_bias);
    attn->k_proj = linear_new(hidden_dim, kv_dim, use_bias);
    attn->v_proj = linear_new(hidden_dim, kv_dim, use_bias);
    attn->o_proj = linear_new(q_dim, hidden_dim, use_bias);

    if (!attn->q_proj || !attn->k_proj || !attn->v_proj || !attn->o_proj) {
        gqa_attention_free(attn);
        return NULL;
    }

    /* Q/K Norm (可选) */
    if (use_q_norm) {
        /* 分配 Q norm 权重 */
        attn->q_norm_weight = (Tensor*)malloc(sizeof(Tensor));
        /* 实际权重由外部加载 */
    }
    if (use_k_norm) {
        attn->k_norm_weight = (Tensor*)malloc(sizeof(Tensor));
    }

    return attn;
}

void gqa_attention_free(GQAAttention* attn) {
    if (!attn) return;

    if (attn->q_proj) linear_free(attn->q_proj);
    if (attn->k_proj) linear_free(attn->k_proj);
    if (attn->v_proj) linear_free(attn->v_proj);
    if (attn->o_proj) linear_free(attn->o_proj);

    if (attn->q_norm_weight) {
        if (attn->q_norm_weight->data) free(attn->q_norm_weight->data);
        free(attn->q_norm_weight);
    }
    if (attn->k_norm_weight) {
        if (attn->k_norm_weight->data) free(attn->k_norm_weight->data);
        free(attn->k_norm_weight);
    }

    free(attn);
}

int gqa_attention_compute_qkv(
    GQAAttention* attn,
    const Tensor* x,
    Tensor** q_out,
    Tensor** k_out,
    Tensor** v_out
) {
    if (!attn || !x || !q_out || !k_out || !v_out) {
        return MYLLM_ERROR_NULL_POINTER;
    }

    *q_out = linear_forward(attn->q_proj, x);
    *k_out = linear_forward(attn->k_proj, x);
    *v_out = linear_forward(attn->v_proj, x);

    if (!*q_out || !*k_out || !*v_out) {
        return MYLLM_ERROR_INTERNAL;
    }

    return MYLLM_OK;
}

int gqa_attention_compute_kv_with_rope(
    GQAAttention* attn,
    const Tensor* x,
    const size_t* positions,
    size_t num_positions,
    Tensor** k_out,
    Tensor** v_out
) {
    if (!attn || !x || !positions || !k_out || !v_out) {
        return MYLLM_ERROR_NULL_POINTER;
    }

    /* 计算 K, V 投影 */
    Tensor* k = linear_forward(attn->k_proj, x);
    Tensor* v = linear_forward(attn->v_proj, x);

    if (!k || !v) {
        if (k) {
            if (k->data) free(k->data);
            free(k);
        }
        if (v) {
            if (v->data) free(v->data);
            free(v);
        }
        return MYLLM_ERROR_INTERNAL;
    }

    /* 对 K 应用 RoPE */
    Tensor* k_with_rope = ops_rope(k, positions, num_positions, attn->rope_theta);
    if (!k_with_rope) {
        /* 如果 RoPE 失败，直接使用原始 K */
        *k_out = k;
        *v_out = v;
        return MYLLM_OK;
    }

    /* 释放原始 K，使用应用了 RoPE 的 K */
    if (k->data) free(k->data);
    free(k);

    *k_out = k_with_rope;
    *v_out = v;

    return MYLLM_OK;
}

Tensor* gqa_attention_forward_with_kv(
    GQAAttention* attn,
    const Tensor* x,
    const Tensor* k_cached,
    const Tensor* v_cached,
    const size_t* positions,
    size_t num_positions
) {
    if (!attn || !x || !k_cached || !v_cached || !positions) {
        return NULL;
    }

    /* 1. 计算 Q 投影 */
    Tensor* q = linear_forward(attn->q_proj, x);
    if (!q) return NULL;

    /* 2. 对 Q 应用 RoPE */
    Tensor* q_with_rope = ops_rope(q, positions, num_positions, attn->rope_theta);
    if (!q_with_rope) {
        /* 如果 RoPE 失败，使用原始 Q */
        q_with_rope = q;
    } else {
        /* 释放原始 Q */
        if (q->data) free(q->data);
        free(q);
    }

    /* 3. 计算注意力分数 */
    /* Q: [batch, seq, num_heads, head_dim] */
    /* K: [batch, num_kv_heads, cache_len, head_dim] */
    /* V: [batch, num_kv_heads, cache_len, head_dim] */

    /* 获取形状信息 */
    size_t q_ndim = q_with_rope->shape.ndim;
    size_t batch_size = 1, q_seq_len = 1, num_q_heads = 1, head_dim;

    if (q_ndim == 3) {
        /* [seq, num_heads, head_dim] 或 [batch, num_heads * head_dim] */
        q_seq_len = q_with_rope->shape.dims[0];
        num_q_heads = attn->num_heads;
        head_dim = attn->head_dim;
    } else if (q_ndim == 4) {
        batch_size = q_with_rope->shape.dims[0];
        q_seq_len = q_with_rope->shape.dims[1];
        num_q_heads = q_with_rope->shape.dims[2];
        head_dim = q_with_rope->shape.dims[3];
    } else {
        if (q_with_rope->owns_data && q_with_rope->data) free(q_with_rope->data);
        free(q_with_rope);
        return NULL;
    }

    /* 获取 K, V 缓存的形状 */
    size_t k_cache_len = k_cached->shape.ndim >= 3 ? k_cached->shape.dims[k_cached->shape.ndim - 2] : 1;
    size_t num_kv_heads = attn->num_kv_heads;

    /* 4. 计算注意力: scaled dot-product attention */
    /* 简化实现: 逐头计算 */

    /* 分配输出张量 */
    Tensor* attn_output = (Tensor*)malloc(sizeof(Tensor));
    if (!attn_output) {
        if (q_with_rope->owns_data && q_with_rope->data) free(q_with_rope->data);
        free(q_with_rope);
        return NULL;
    }
    memset(attn_output, 0, sizeof(Tensor));

    /* 创建输出形状 [batch, seq, num_heads, head_dim] 或 [seq, num_heads * head_dim] */
    if (q_ndim == 3) {
        size_t out_dims[2] = {q_seq_len, num_q_heads * head_dim};
        attn_output->shape = shape_new(out_dims, 2);
        attn_output->strides[0] = num_q_heads * head_dim;
        attn_output->strides[1] = 1;
    } else {
        size_t out_dims[3] = {batch_size, q_seq_len, num_q_heads * head_dim};
        attn_output->shape = shape_new(out_dims, 3);
        attn_output->strides[0] = q_seq_len * num_q_heads * head_dim;
        attn_output->strides[1] = num_q_heads * head_dim;
        attn_output->strides[2] = 1;
    }

    size_t output_numel = shape_numel(&attn_output->shape);
    attn_output->dtype = DTYPE_F32;
    attn_output->offset = 0;
    attn_output->device.type = DEVICE_CPU;
    attn_output->device.id = 0;
    attn_output->owns_data = true;
    attn_output->data = calloc(output_numel, sizeof(float));

    if (!attn_output->data) {
        free(attn_output);
        if (q_with_rope->owns_data && q_with_rope->data) free(q_with_rope->data);
        free(q_with_rope);
        return NULL;
    }

    float* q_data = (float*)q_with_rope->data;
    float* k_data = (float*)k_cached->data;
    float* v_data = (float*)v_cached->data;
    float* out_data = (float*)attn_output->data;
    float scale = attn->scale;

    /* 计算 groups (GQA) */
    size_t heads_per_group = num_q_heads / num_kv_heads;

    /* 逐位置、逐头计算注意力 */
    for (size_t pos = 0; pos < num_positions && pos < q_seq_len; pos++) {
        size_t cur_pos = positions[pos];

        for (size_t h = 0; h < num_q_heads; h++) {
            size_t kv_h = h / heads_per_group;  /* 对应的 KV 头 */

            /* 获取当前 Q 向量 */
            float* q_vec;
            if (q_ndim == 3) {
                q_vec = &q_data[(pos * num_q_heads + h) * head_dim];
            } else {
                q_vec = &q_data[((0 * q_seq_len + pos) * num_q_heads + h) * head_dim];
            }

            /* 计算注意力分数 (只看当前位置之前的 token) */
            float* scores = (float*)malloc((cur_pos + 1) * sizeof(float));
            if (!scores) continue;

            float max_score = -1e30f;
            for (size_t k_pos = 0; k_pos <= cur_pos && k_pos < k_cache_len; k_pos++) {
                float* k_vec;
                if (k_cached->shape.ndim == 3) {
                    k_vec = &k_data[(kv_h * k_cache_len + k_pos) * head_dim];
                } else {
                    k_vec = &k_data[((0 * num_kv_heads + kv_h) * k_cache_len + k_pos) * head_dim];
                }

                float score = 0.0f;
                for (size_t d = 0; d < head_dim; d++) {
                    score += q_vec[d] * k_vec[d];
                }
                score *= scale;
                scores[k_pos] = score;
                if (score > max_score) max_score = score;
            }

            /* Softmax (numerical stable) */
            float sum_exp = 0.0f;
            for (size_t k_pos = 0; k_pos <= cur_pos && k_pos < k_cache_len; k_pos++) {
                scores[k_pos] = expf(scores[k_pos] - max_score);
                sum_exp += scores[k_pos];
            }
            for (size_t k_pos = 0; k_pos <= cur_pos && k_pos < k_cache_len; k_pos++) {
                scores[k_pos] /= sum_exp;
            }

            /* 加权求和 */
            float* out_vec;
            if (q_ndim == 3) {
                out_vec = &out_data[(pos * num_q_heads + h) * head_dim];
            } else {
                out_vec = &out_data[((0 * q_seq_len + pos) * num_q_heads + h) * head_dim];
            }

            for (size_t d = 0; d < head_dim; d++) {
                out_vec[d] = 0.0f;
            }

            for (size_t k_pos = 0; k_pos <= cur_pos && k_pos < k_cache_len; k_pos++) {
                float* v_vec;
                if (v_cached->shape.ndim == 3) {
                    v_vec = &v_data[(kv_h * k_cache_len + k_pos) * head_dim];
                } else {
                    v_vec = &v_data[((0 * num_kv_heads + kv_h) * k_cache_len + k_pos) * head_dim];
                }

                for (size_t d = 0; d < head_dim; d++) {
                    out_vec[d] += scores[k_pos] * v_vec[d];
                }
            }

            free(scores);
        }
    }

    /* 释放 Q */
    if (q_with_rope->owns_data && q_with_rope->data) free(q_with_rope->data);
    free(q_with_rope);

    /* 6. Output 投影 */
    Tensor* output = linear_forward(attn->o_proj, attn_output);

    /* 释放中间结果 */
    if (attn_output->data) free(attn_output->data);
    free(attn_output);

    return output;
}

Tensor* gqa_attention_forward_with_positions(
    GQAAttention* attn,
    const Tensor* x,
    const size_t* positions,
    size_t num_positions
) {
    if (!attn || !x || !positions) return NULL;

    /* 计算 Q, K, V */
    Tensor *q, *k, *v;
    int ret = gqa_attention_compute_qkv(attn, x, &q, &k, &v);
    if (ret != MYLLM_OK) return NULL;

    /* 对 Q, K 应用 RoPE */
    Tensor* q_with_rope = ops_rope(q, positions, num_positions, attn->rope_theta);
    Tensor* k_with_rope = ops_rope(k, positions, num_positions, attn->rope_theta);

    /* 如果 RoPE 失败，使用原始张量 */
    if (!q_with_rope) q_with_rope = q;
    if (!k_with_rope) k_with_rope = k;

    /* 计算注意力 */
    Tensor* output = gqa_attention_forward_with_kv(attn, x, k_with_rope, v, positions, num_positions);

    /* 释放临时张量 */
    if (q_with_rope != q) {
        if (q->owns_data && q->data) free(q->data);
        free(q);
    }
    if (q_with_rope->owns_data && q_with_rope->data) free(q_with_rope->data);
    free(q_with_rope);

    if (k_with_rope != k) {
        if (k->owns_data && k->data) free(k->data);
        free(k);
    }
    if (k_with_rope->owns_data && k_with_rope->data) free(k_with_rope->data);
    free(k_with_rope);

    if (v) {
        if (v->owns_data && v->data) free(v->data);
        free(v);
    }

    return output;
}

Tensor* gqa_attention_forward(GQAAttention* attn, const Tensor* x) {
    if (!attn || !x) return NULL;

    /* 假设序列位置为 0, 1, 2, ... */
    size_t seq_len = x->shape.ndim >= 2 ? x->shape.dims[x->shape.ndim - 2] : 1;
    size_t* positions = (size_t*)malloc(seq_len * sizeof(size_t));
    if (!positions) return NULL;

    for (size_t i = 0; i < seq_len; i++) {
        positions[i] = i;
    }

    Tensor* output = gqa_attention_forward_with_positions(attn, x, positions, seq_len);
    free(positions);

    return output;
}

/* ============================================================================
 * LLaMA 注意力层实现
 * ============================================================================ */

LlamaAttention* llama_attention_new(
    size_t hidden_dim,
    size_t num_heads,
    size_t num_kv_heads,
    size_t head_dim,
    double rope_theta
) {
    LlamaAttention* attn = (LlamaAttention*)malloc(sizeof(LlamaAttention));
    if (!attn) return NULL;

    /* LLaMA 默认: 无 Q/K Norm, 无 bias */
    attn->inner = gqa_attention_new(
        hidden_dim,
        num_heads,
        num_kv_heads,
        head_dim,
        rope_theta,
        false,  /* use_q_norm */
        false,  /* use_k_norm */
        false   /* use_bias */
    );

    if (!attn->inner) {
        free(attn);
        return NULL;
    }

    return attn;
}

void llama_attention_free(LlamaAttention* attn) {
    if (!attn) return;
    if (attn->inner) gqa_attention_free(attn->inner);
    free(attn);
}

int llama_attention_compute_kv_with_rope(
    LlamaAttention* attn,
    const Tensor* x,
    const size_t* positions,
    size_t num_positions,
    Tensor** k_out,
    Tensor** v_out
) {
    if (!attn || !attn->inner) return MYLLM_ERROR_NULL_POINTER;
    return gqa_attention_compute_kv_with_rope(attn->inner, x, positions, num_positions, k_out, v_out);
}

Tensor* llama_attention_forward_with_kv(
    LlamaAttention* attn,
    const Tensor* x,
    const Tensor* k_cached,
    const Tensor* v_cached,
    const size_t* positions,
    size_t num_positions
) {
    if (!attn || !attn->inner) return NULL;
    return gqa_attention_forward_with_kv(attn->inner, x, k_cached, v_cached, positions, num_positions);
}

Tensor* llama_attention_forward_with_positions(
    LlamaAttention* attn,
    const Tensor* x,
    const size_t* positions,
    size_t num_positions
) {
    if (!attn || !attn->inner) return NULL;
    return gqa_attention_forward_with_positions(attn->inner, x, positions, num_positions);
}

Linear* llama_attention_q_proj(LlamaAttention* attn) {
    return attn && attn->inner ? attn->inner->q_proj : NULL;
}

Linear* llama_attention_k_proj(LlamaAttention* attn) {
    return attn && attn->inner ? attn->inner->k_proj : NULL;
}

Linear* llama_attention_v_proj(LlamaAttention* attn) {
    return attn && attn->inner ? attn->inner->v_proj : NULL;
}

Linear* llama_attention_o_proj(LlamaAttention* attn) {
    return attn && attn->inner ? attn->inner->o_proj : NULL;
}
