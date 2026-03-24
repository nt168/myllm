/**
 * @file block.c
 * @brief LLaMA Transformer 块实现
 */

#include "models/llama/block.h"
#include "tensor/shape.h"
#include <stdlib.h>
#include <string.h>
#include <math.h>

/* ============================================================================
 * RMSNorm 实现
 * ============================================================================ */

RMSNorm* rmsnorm_new(size_t hidden_dim, float eps) {
    RMSNorm* layer = (RMSNorm*)malloc(sizeof(RMSNorm));
    if (!layer) return NULL;

    layer->hidden_dim = hidden_dim;
    layer->eps = eps;

    /* 分配权重张量 */
    layer->weight = (Tensor*)malloc(sizeof(Tensor));
    if (!layer->weight) {
        free(layer);
        return NULL;
    }
    memset(layer->weight, 0, sizeof(Tensor));

    /* 使用 shape_new 创建形状 */
    size_t weight_dims[1] = {hidden_dim};
    layer->weight->shape = shape_new(weight_dims, 1);
    layer->weight->strides[0] = 1;
    layer->weight->dtype = DTYPE_F32;
    layer->weight->offset = 0;
    layer->weight->device.type = DEVICE_CPU;
    layer->weight->device.id = 0;
    layer->weight->owns_data = true;

    /* 分配数据 */
    layer->weight->data = calloc(hidden_dim, sizeof(float));
    if (!layer->weight->data) {
        free(layer->weight);
        free(layer);
        return NULL;
    }

    /* 初始化为 1.0 */
    float* w = (float*)layer->weight->data;
    for (size_t i = 0; i < hidden_dim; i++) {
        w[i] = 1.0f;
    }

    return layer;
}

void rmsnorm_free(RMSNorm* layer) {
    if (!layer) return;

    if (layer->weight) {
        if (layer->weight->data) free(layer->weight->data);
        free(layer->weight);
    }

    free(layer);
}

Tensor* rmsnorm_forward(RMSNorm* layer, const Tensor* x) {
    if (!layer || !x) return NULL;

    size_t ndim = x->shape.ndim;
    if (ndim < 1) return NULL;

    /* 获取最后一个维度 */
    size_t last_dim = x->shape.dims[ndim - 1];
    if (last_dim != layer->hidden_dim) return NULL;

    /* 创建输出张量 */
    Tensor* output = (Tensor*)malloc(sizeof(Tensor));
    if (!output) return NULL;
    memset(output, 0, sizeof(Tensor));

    /* 使用 shape_new 创建形状 */
    output->shape = shape_new(x->shape.dims, ndim);

    /* 计算步幅 */
    output->strides[ndim - 1] = 1;
    for (int i = (int)ndim - 2; i >= 0; i--) {
        output->strides[i] = output->shape.dims[i + 1] * output->strides[i + 1];
    }

    size_t output_numel = shape_numel(&output->shape);
    output->dtype = DTYPE_F32;
    output->offset = 0;
    output->device.type = DEVICE_CPU;
    output->device.id = 0;
    output->owns_data = true;
    output->data = calloc(output_numel, sizeof(float));

    if (!output->data) {
        free(output);
        return NULL;
    }

    const float* x_data = (const float*)x->data;
    const float* w_data = (const float*)layer->weight->data;
    float* out_data = (float*)output->data;

    size_t numel = shape_numel(&x->shape);

    /* RMSNorm: x * w / sqrt(mean(x^2) + eps) */
    /* 计算每个隐藏维度的 RMSNorm */
    size_t batch_size = 1;
    size_t seq_len = 1;
    if (ndim == 1) {
        batch_size = 1;
        seq_len = x->shape.dims[0];
    } else if (ndim == 2) {
        batch_size = 1;
        seq_len = x->shape.dims[0];
    } else if (ndim == 3) {
        batch_size = x->shape.dims[0];
        seq_len = x->shape.dims[1];
    }

    /* 对每个位置进行归一化 */
    for (size_t b = 0; b < batch_size; b++) {
        for (size_t s = 0; s < seq_len; s++) {
            size_t base_idx;
            if (ndim == 1) {
                base_idx = s;
            } else if (ndim == 2) {
                base_idx = s * layer->hidden_dim;
            } else {
                base_idx = b * seq_len * layer->hidden_dim + s * layer->hidden_dim;
            }

            /* 计算 RMS */
            float sum_sq = 0.0f;
            for (size_t i = 0; i < layer->hidden_dim; i++) {
                float val = x_data[base_idx + i];
                sum_sq += val * val;
            }
            float rms = sqrtf(sum_sq / layer->hidden_dim + layer->eps);

            /* 归一化 */
            for (size_t i = 0; i < layer->hidden_dim; i++) {
                out_data[base_idx + i] = (x_data[base_idx + i] / rms) * w_data[i];
            }
        }
    }

    return output;
}

/* ============================================================================
 * MLP (SwiGLU) 实现
 * ============================================================================ */

MLP* mlp_new(size_t hidden_dim, size_t intermediate_dim) {
    MLP* layer = (MLP*)malloc(sizeof(MLP));
    if (!layer) return NULL;

    layer->hidden_dim = hidden_dim;
    layer->intermediate_dim = intermediate_dim;

    /* gate_proj: [intermediate_dim, hidden_dim] */
    layer->gate_proj = linear_new(hidden_dim, intermediate_dim, false);
    if (!layer->gate_proj) {
        free(layer);
        return NULL;
    }

    /* up_proj: [intermediate_dim, hidden_dim] */
    layer->up_proj = linear_new(hidden_dim, intermediate_dim, false);
    if (!layer->up_proj) {
        linear_free(layer->gate_proj);
        free(layer);
        return NULL;
    }

    /* down_proj: [hidden_dim, intermediate_dim] */
    layer->down_proj = linear_new(intermediate_dim, hidden_dim, false);
    if (!layer->down_proj) {
        linear_free(layer->gate_proj);
        linear_free(layer->up_proj);
        free(layer);
        return NULL;
    }

    return layer;
}

void mlp_free(MLP* layer) {
    if (!layer) return;

    if (layer->gate_proj) linear_free(layer->gate_proj);
    if (layer->up_proj) linear_free(layer->up_proj);
    if (layer->down_proj) linear_free(layer->down_proj);
    free(layer);
}

Tensor* mlp_forward(MLP* layer, const Tensor* x) {
    if (!layer || !x) return NULL;

    /* SwiGLU: down(silu(gate(x)) * up(x)) */
    Tensor* gate = linear_forward(layer->gate_proj, x);
    if (!gate) return NULL;

    Tensor* up = linear_forward(layer->up_proj, x);
    if (!up) {
        /* 释放 gate */
        if (gate->data) free(gate->data);
        free(gate);
        return NULL;
    }

    /* SiLU 激活: x * sigmoid(x) */
    float* gate_data = (float*)gate->data;
    size_t gate_numel = shape_numel(&gate->shape);
    for (size_t i = 0; i < gate_numel; i++) {
        float val = gate_data[i];
        float sigmoid = 1.0f / (1.0f + expf(-val));
        gate_data[i] = val * sigmoid;
    }

    /* 逐元素乘法: silu(gate) * up */
    float* up_data = (float*)up->data;
    for (size_t i = 0; i < gate_numel; i++) {
        gate_data[i] *= up_data[i];
    }

    /* 释放 up */
    if (up->data) free(up->data);
    free(up);

    /* down projection */
    Tensor* output = linear_forward(layer->down_proj, gate);

    /* 释放 gate */
    if (gate->data) free(gate->data);
    free(gate);

    return output;
}

/* ============================================================================
 * LLaMA Transformer 块实现
 * ============================================================================ */

LlamaTransformerBlock* llama_block_new(
    size_t hidden_dim,
    size_t num_heads,
    size_t num_kv_heads,
    size_t head_dim,
    size_t intermediate_dim,
    float norm_eps,
    double rope_theta
) {
    LlamaTransformerBlock* block = (LlamaTransformerBlock*)malloc(sizeof(LlamaTransformerBlock));
    if (!block) return NULL;

    /* 注意力层 */
    block->attention = llama_attention_new(hidden_dim, num_heads, num_kv_heads, head_dim, rope_theta);
    if (!block->attention) {
        free(block);
        return NULL;
    }

    /* MLP */
    block->mlp = mlp_new(hidden_dim, intermediate_dim);
    if (!block->mlp) {
        llama_attention_free(block->attention);
        free(block);
        return NULL;
    }

    /* 输入归一化 */
    block->input_norm = rmsnorm_new(hidden_dim, norm_eps);
    if (!block->input_norm) {
        llama_attention_free(block->attention);
        mlp_free(block->mlp);
        free(block);
        return NULL;
    }

    /* 注意力后归一化 */
    block->post_attention_norm = rmsnorm_new(hidden_dim, norm_eps);
    if (!block->post_attention_norm) {
        llama_attention_free(block->attention);
        mlp_free(block->mlp);
        rmsnorm_free(block->input_norm);
        free(block);
        return NULL;
    }

    return block;
}

LlamaTransformerBlock* llama_block_from_config(const LlamaConfig* config) {
    if (!config) return NULL;
    return llama_block_new(
        config->hidden_size,
        config->num_attention_heads,
        config->num_key_value_heads,
        config->head_dim,
        config->intermediate_size,
        config->rms_norm_eps,
        config->rope_theta
    );
}

void llama_block_free(LlamaTransformerBlock* block) {
    if (!block) return;

    if (block->attention) llama_attention_free(block->attention);
    if (block->mlp) mlp_free(block->mlp);
    if (block->input_norm) rmsnorm_free(block->input_norm);
    if (block->post_attention_norm) rmsnorm_free(block->post_attention_norm);
    free(block);
}

Tensor* llama_block_forward_with_positions(
    LlamaTransformerBlock* block,
    const Tensor* x,
    const size_t* positions,
    size_t num_positions
) {
    if (!block || !x) return NULL;

    /* Pre-norm attention */
    Tensor* residual = NULL;
    /* TODO: 实现 tensor_clone */
    residual = (Tensor*)malloc(sizeof(Tensor));
    if (!residual) return NULL;
    /* 复制 x 到 residual */

    Tensor* normed = rmsnorm_forward(block->input_norm, x);
    if (!normed) {
        free(residual);
        return NULL;
    }

    Tensor* attn_out = llama_attention_forward_with_positions(block->attention, normed, positions, num_positions);
    if (!attn_out) {
        if (normed->data) free(normed->data);
        free(normed);
        free(residual);
        return NULL;
    }

    /* residual + attn_out */
    /* TODO: 实现 tensor_add */
    Tensor* hidden = NULL;
    /* 简化: 直接使用 attn_out 作为 hidden */

    if (normed->data) free(normed->data);
    free(normed);

    /* Pre-norm MLP */
    residual = hidden;
    normed = rmsnorm_forward(block->post_attention_norm, hidden);
    if (!normed) {
        if (residual && residual->data) free(residual->data);
        if (residual) free(residual);
        return NULL;
    }

    Tensor* mlp_out = mlp_forward(block->mlp, normed);
    if (!mlp_out) {
        if (normed->data) free(normed->data);
        free(normed);
        if (residual && residual->data) free(residual->data);
        if (residual) free(residual);
        return NULL;
    }

    if (normed->data) free(normed->data);
    free(normed);

    /* residual + mlp_out */
    /* TODO: 实现 tensor_add */
    Tensor* output = mlp_out;  /* 简化 */

    if (residual && residual->data) free(residual->data);
    if (residual) free(residual);

    return output;
}

Tensor* llama_block_forward(LlamaTransformerBlock* block, const Tensor* x) {
    if (!block || !x) return NULL;

    /* 无位置信息的前向传播 */
    size_t seq_len = 1;
    if (x->shape.ndim >= 2) {
        seq_len = x->shape.dims[x->shape.ndim - 2];
    }

    size_t* positions = (size_t*)malloc(seq_len * sizeof(size_t));
    if (!positions) return NULL;

    for (size_t i = 0; i < seq_len; i++) {
        positions[i] = i;
    }

    Tensor* output = llama_block_forward_with_positions(block, x, positions, seq_len);
    free(positions);

    return output;
}

int llama_block_compute_kv_with_rope(
    LlamaTransformerBlock* block,
    const Tensor* x,
    const size_t* positions,
    size_t num_positions,
    Tensor** k_out,
    Tensor** v_out
) {
    if (!block || !x || !k_out || !v_out) return MYLLM_ERROR_NULL_POINTER;
    return llama_attention_compute_kv_with_rope(block->attention, x, positions, num_positions, k_out, v_out);
}

LlamaAttention* llama_block_attention(LlamaTransformerBlock* block) {
    return block ? block->attention : NULL;
}

MLP* llama_block_mlp(LlamaTransformerBlock* block) {
    return block ? block->mlp : NULL;
}

RMSNorm* llama_block_input_norm(LlamaTransformerBlock* block) {
    return block ? block->input_norm : NULL;
}

RMSNorm* llama_block_post_attention_norm(LlamaTransformerBlock* block) {
    return block ? block->post_attention_norm : NULL;
}
