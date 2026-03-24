/**
 * @file rope.c
 * @brief RoPE (旋转位置编码) 实现
 */

#include "ops/rope.h"
#include "tensor/shape.h"
#include <stdlib.h>
#include <string.h>
#include <math.h>

/**
 * @brief 应用 RoPE 位置编码
 *
 * 简化实现，仅作为占位符
 */
Tensor* ops_rope(const Tensor* x, const size_t* positions, size_t num_positions, double theta) {
    if (!x || !positions) return NULL;

    /* 获取形状信息 */
    size_t ndim = x->shape.ndim;
    if (ndim < 3) return NULL;

    size_t seq_len = x->shape.dims[ndim - 2];
    size_t head_dim = x->shape.dims[ndim - 1];

    /* 创建输出张量 */
    Tensor* output = (Tensor*)malloc(sizeof(Tensor));
    if (!output) return NULL;
    memset(output, 0, sizeof(Tensor));

    output->shape = shape_new(x->shape.dims, ndim);

    /* 计算步幅 */
    output->strides[ndim - 1] = 1;
    for (int i = (int)ndim - 2; i >= 0; i--) {
        output->strides[i] = output->shape.dims[i + 1] * output->strides[i + 1];
    }

    size_t output_numel = shape_numel(&output->shape);
    output->dtype = x->dtype;
    output->offset = 0;
    output->device = x->device;
    output->owns_data = true;
    output->data = malloc(output_numel * sizeof(float));

    if (!output->data) {
        free(output);
        return NULL;
    }

    /* 复制数据并应用 RoPE */
    const float* x_data = (const float*)x->data;
    float* out_data = (float*)output->data;

    /* RoPE 编码 */
    for (size_t pos_idx = 0; pos_idx < num_positions && pos_idx < seq_len; pos_idx++) {
        size_t pos = positions[pos_idx];

        /* 计算每个维度对的旋转 */
        for (size_t i = 0; i < head_dim; i += 2) {
            double freq = 1.0 / pow(theta, (double)(i % (head_dim / 2)) * 2.0 / head_dim);
            double angle = pos * freq;

            float cos_val = cosf(angle);
            float sin_val = sinf(angle);

            /* 对所有批次和头应用旋转 */
            size_t batch_size = 1;
            size_t num_heads = 1;
            if (ndim == 4) {
                batch_size = x->shape.dims[0];
                num_heads = x->shape.dims[1];
            } else if (ndim == 3) {
                num_heads = x->shape.dims[0];
            }

            for (size_t b = 0; b < batch_size; b++) {
                for (size_t h = 0; h < num_heads; h++) {
                    size_t base_idx;
                    if (ndim == 4) {
                        base_idx = ((b * num_heads + h) * seq_len + pos_idx) * head_dim;
                    } else if (ndim == 3) {
                        base_idx = (h * seq_len + pos_idx) * head_dim;
                    } else {
                        base_idx = pos_idx * head_dim;
                    }

                    float x0 = x_data[base_idx + i];
                    float x1 = (i + 1 < head_dim) ? x_data[base_idx + i + 1] : 0.0f;

                    out_data[base_idx + i] = x0 * cos_val - x1 * sin_val;
                    if (i + 1 < head_dim) {
                        out_data[base_idx + i + 1] = x0 * sin_val + x1 * cos_val;
                    }
                }
            }
        }
    }

    return output;
}
