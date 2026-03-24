/**
 * @file normalization.c
 * @brief 归一化运算实现
 */

#include "ops/normalization.h"
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <stdio.h>

/* ============================================================================
 * LayerNorm 实现
 * ============================================================================ */

/**
 * @brief 计算 layernorm 的线性索引
 */
static size_t compute_layernorm_linear_index(
    const size_t* outer_dims,
    size_t outer_ndim,
    const size_t* inner_dims,
    size_t inner_ndim,
    size_t outer_idx,
    size_t inner_idx
) {
    /* 构建 full shape: outer_dims + inner_dims */
    size_t full_shape[MYLLM_MAX_NDIM];
    size_t full_ndim = outer_ndim + inner_ndim;

    for (size_t i = 0; i < outer_ndim; i++) {
        full_shape[i] = outer_dims[i];
    }
    for (size_t i = 0; i < inner_ndim; i++) {
        full_shape[outer_ndim + i] = inner_dims[i];
    }

    /* 计算 full strides */
    size_t full_strides[MYLLM_MAX_NDIM];
    size_t current = 1;
    for (size_t i = full_ndim; i > 0; i--) {
        full_strides[i - 1] = current;
        current *= full_shape[i - 1];
    }

    /* 转换 outer_idx 为多维索引 */
    size_t outer_indices[MYLLM_MAX_NDIM];
    size_t remaining = outer_idx;
    for (size_t i = outer_ndim; i > 0; i--) {
        outer_indices[i - 1] = remaining % outer_dims[i - 1];
        remaining /= outer_dims[i - 1];
    }

    /* 转换 inner_idx 为多维索引 */
    size_t inner_indices[MYLLM_MAX_NDIM];
    remaining = inner_idx;
    for (size_t i = inner_ndim; i > 0; i--) {
        inner_indices[i - 1] = remaining % inner_dims[i - 1];
        remaining /= inner_dims[i - 1];
    }

    /* 合并索引并计算线性索引 */
    size_t linear_idx = 0;
    for (size_t i = 0; i < outer_ndim; i++) {
        linear_idx += outer_indices[i] * full_strides[i];
    }
    for (size_t i = 0; i < inner_ndim; i++) {
        linear_idx += inner_indices[i] * full_strides[outer_ndim + i];
    }

    return linear_idx;
}

Tensor* ops_layernorm(const Tensor* input, size_t normalized_shape,
                       const float* weight, const float* bias, float eps) {
    if (!input) {
        return NULL;
    }

    const Shape* shape = tensor_shape(input);
    size_t ndim = shape->ndim;

    /* 检查 normalized_shape 是否匹配最后一个维度 */
    if (normalized_shape != shape->dims[ndim - 1]) {
        fprintf(stderr, "LayerNorm: normalized_shape (%zu) != last dim (%zu)\n",
                normalized_shape, shape->dims[ndim - 1]);
        return NULL;
    }

    /* 计算外层和内层维度 */
    size_t outer_dims[MYLLM_MAX_NDIM];
    size_t outer_ndim = ndim - 1;
    for (size_t i = 0; i < outer_ndim; i++) {
        outer_dims[i] = shape->dims[i];
    }

    size_t inner_dims[1] = { normalized_shape };
    size_t inner_ndim = 1;

    /* 计算元素数量 */
    size_t outer_numel = 1;
    for (size_t i = 0; i < outer_ndim; i++) {
        outer_numel *= outer_dims[i];
    }
    size_t inner_numel = normalized_shape;

    /* 创建输出张量 */
    Tensor* output = tensor_new(shape, input->dtype);
    if (!output) {
        return NULL;
    }

    /* 对每个外层位置计算 LayerNorm */
    for (size_t outer_idx = 0; outer_idx < outer_numel; outer_idx++) {
        float sum = 0.0f;
        float sum_sq = 0.0f;

        /* 计算均值和方差 */
        for (size_t inner_idx = 0; inner_idx < inner_numel; inner_idx++) {
            size_t linear_idx = compute_layernorm_linear_index(
                outer_dims, outer_ndim, inner_dims, inner_ndim,
                outer_idx, inner_idx
            );
            float val = tensor_get_f32(input, linear_idx);
            sum += val;
            sum_sq += val * val;
        }

        float mean = sum / (float)inner_numel;
        float variance = (sum_sq / (float)inner_numel) - (mean * mean);
        if (variance < 0.0f) {
            variance = 0.0f;
        }

        /* 归一化 */
        for (size_t inner_idx = 0; inner_idx < inner_numel; inner_idx++) {
            size_t linear_idx = compute_layernorm_linear_index(
                outer_dims, outer_ndim, inner_dims, inner_ndim,
                outer_idx, inner_idx
            );
            float val = tensor_get_f32(input, linear_idx);
            float normalized = (val - mean) / (sqrtf(variance) + eps);

            /* 应用 weight 和 bias */
            float w = weight ? weight[inner_idx] : 1.0f;
            float b = bias ? bias[inner_idx] : 0.0f;
            normalized = normalized * w + b;

            tensor_set_f32(output, linear_idx, normalized);
        }
    }

    return output;
}

/* ============================================================================
 * RMSNorm 实现
 * ============================================================================ */

Tensor* ops_rmsnorm(const Tensor* input, size_t normalized_shape,
                     const float* weight, float eps) {
    if (!input) {
        return NULL;
    }

    const Shape* shape = tensor_shape(input);
    size_t ndim = shape->ndim;

    if (normalized_shape != shape->dims[ndim - 1]) {
        fprintf(stderr, "RMSNorm: normalized_shape (%zu) != last dim (%zu)\n",
                normalized_shape, shape->dims[ndim - 1]);
        return NULL;
    }

    /* 计算外层和内层维度 */
    size_t outer_dims[MYLLM_MAX_NDIM];
    size_t outer_ndim = ndim - 1;
    for (size_t i = 0; i < outer_ndim; i++) {
        outer_dims[i] = shape->dims[i];
    }

    size_t inner_dims[1] = { normalized_shape };
    size_t inner_ndim = 1;

    size_t outer_numel = 1;
    for (size_t i = 0; i < outer_ndim; i++) {
        outer_numel *= outer_dims[i];
    }
    size_t inner_numel = normalized_shape;

    Tensor* output = tensor_new(shape, input->dtype);
    if (!output) {
        return NULL;
    }

    for (size_t outer_idx = 0; outer_idx < outer_numel; outer_idx++) {
        float sum_sq = 0.0f;

        /* 计算 RMS */
        for (size_t inner_idx = 0; inner_idx < inner_numel; inner_idx++) {
            size_t linear_idx = compute_layernorm_linear_index(
                outer_dims, outer_ndim, inner_dims, inner_ndim,
                outer_idx, inner_idx
            );
            float val = tensor_get_f32(input, linear_idx);
            sum_sq += val * val;
        }

        float rms_sq = sum_sq / (float)inner_numel + eps;
        float rms = sqrtf(rms_sq);

        /* 归一化 */
        for (size_t inner_idx = 0; inner_idx < inner_numel; inner_idx++) {
            size_t linear_idx = compute_layernorm_linear_index(
                outer_dims, outer_ndim, inner_dims, inner_ndim,
                outer_idx, inner_idx
            );
            float val = tensor_get_f32(input, linear_idx);
            float normalized = val / rms;

            /* 应用 weight */
            float w = weight ? weight[inner_idx] : 1.0f;
            normalized = normalized * w;

            tensor_set_f32(output, linear_idx, normalized);
        }
    }

    return output;
}

/* ============================================================================
 * Softmax 实现
 * ============================================================================ */

Tensor* ops_softmax(const Tensor* input, size_t dim) {
    if (!input) {
        return NULL;
    }

    const Shape* shape = tensor_shape(input);
    size_t ndim = shape->ndim;

    if (dim >= ndim) {
        fprintf(stderr, "Softmax: dim %zu >= ndim %zu\n", dim, ndim);
        return NULL;
    }

    size_t dim_size = shape->dims[dim];

    /* 计算其他维度的乘积 */
    size_t outer_numel = 1;
    for (size_t i = 0; i < dim; i++) {
        outer_numel *= shape->dims[i];
    }

    size_t inner_numel = 1;
    for (size_t i = dim + 1; i < ndim; i++) {
        inner_numel *= shape->dims[i];
    }

    /* 创建输出张量 */
    Tensor* output = tensor_new(shape, input->dtype);
    if (!output) {
        return NULL;
    }

    /* 对每个 outer 和 inner 位置计算 softmax */
    for (size_t outer = 0; outer < outer_numel; outer++) {
        for (size_t inner = 0; inner < inner_numel; inner++) {
            /* 找到该位置在 dim 维度上的最大值 */
            float max_val = -INFINITY;
            for (size_t d = 0; d < dim_size; d++) {
                /* 计算线性索引 */
                size_t idx = (outer * dim_size + d) * inner_numel + inner;
                float val = tensor_get_f32(input, idx);
                if (val > max_val) {
                    max_val = val;
                }
            }

            /* 计算 exp(x - max) 的和 */
            float sum_exp = 0.0f;
            for (size_t d = 0; d < dim_size; d++) {
                size_t idx = (outer * dim_size + d) * inner_numel + inner;
                float val = tensor_get_f32(input, idx);
                sum_exp += expf(val - max_val);
            }

            /* 计算 softmax */
            for (size_t d = 0; d < dim_size; d++) {
                size_t idx = (outer * dim_size + d) * inner_numel + inner;
                float val = tensor_get_f32(input, idx);
                float softmax_val = expf(val - max_val) / sum_exp;
                tensor_set_f32(output, idx, softmax_val);
            }
        }
    }

    return output;
}
