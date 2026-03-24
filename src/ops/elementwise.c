/**
 * @file elementwise.c
 * @brief 逐元素运算实现
 */

#include "ops/elementwise.h"
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <stdio.h>

/* ============================================================================
 * 内部辅助函数
 * ============================================================================ */

/**
 * @brief 计算 broadcast 形状
 */
int ops_broadcast_shape(const Shape* a, const Shape* b, Shape* result) {
    size_t max_ndim = a->ndim > b->ndim ? a->ndim : b->ndim;

    if (max_ndim > MYLLM_MAX_NDIM) {
        return -1;
    }

    result->ndim = max_ndim;

    /* 从右向左对齐维度 */
    for (size_t i = 0; i < max_ndim; i++) {
        size_t idx = max_ndim - 1 - i;

        size_t dim_a = (i < a->ndim) ? a->dims[a->ndim - 1 - i] : 1;
        size_t dim_b = (i < b->ndim) ? b->dims[b->ndim - 1 - i] : 1;

        if (dim_a == dim_b) {
            result->dims[idx] = dim_a;
        } else if (dim_a == 1) {
            result->dims[idx] = dim_b;
        } else if (dim_b == 1) {
            result->dims[idx] = dim_a;
        } else {
            /* 无法 broadcast */
            fprintf(stderr, "Broadcast error: shapes incompatible at dim %zu: %zu vs %zu\n",
                    idx, dim_a, dim_b);
            return -1;
        }
    }

    return 0;
}

/**
 * @brief 将线性索引转换为多维索引
 */
static void linear_to_multi_idx(size_t linear_idx, const size_t* dims, size_t ndim,
                                 size_t* indices) {
    size_t remaining = linear_idx;
    for (ssize_t i = (ssize_t)ndim - 1; i >= 0; i--) {
        indices[i] = remaining % dims[i];
        remaining /= dims[i];
    }
}

/**
 * @brief 计算在 broadcast 后的多维索引对应的输入张量的线性索引
 *
 * @param output_indices 输出张量的多维索引
 * @param output_ndim 输出张量的维度数
 * @param tensor_shape 输入张量的形状
 * @return 输入张量的线性索引
 */
static size_t compute_broadcast_linear_index(
    const size_t* output_indices,
    size_t output_ndim,
    const Shape* tensor_shape
) {
    if (tensor_shape->ndim == 0) {
        return 0;
    }

    /* 计算张量在输出维度中的起始位置 (右对齐) */
    size_t tensor_start_dim = output_ndim - tensor_shape->ndim;

    /* 计算张量的多维索引 */
    size_t tensor_indices[MYLLM_MAX_NDIM];
    for (size_t i = 0; i < tensor_shape->ndim; i++) {
        size_t output_dim_idx = tensor_start_dim + i;
        size_t dim_size = tensor_shape->dims[i];
        /* 如果维度大小为1，索引始终为0 (broadcast) */
        tensor_indices[i] = (dim_size == 1) ? 0 : output_indices[output_dim_idx];
    }

    /* 计算线性索引 */
    size_t linear_idx = 0;
    size_t stride = 1;
    for (ssize_t i = (ssize_t)tensor_shape->ndim - 1; i >= 0; i--) {
        linear_idx += tensor_indices[i] * stride;
        stride *= tensor_shape->dims[i];
    }

    return linear_idx;
}

/**
 * @brief 通用的逐元素运算实现
 */
typedef float (*elementwise_op_func)(float a, float b);

static Tensor* tensor_elementwise_op(
    const Tensor* a,
    const Tensor* b,
    elementwise_op_func op
) {
    /* 检查数据类型是否匹配 */
    if (a->dtype != b->dtype) {
        fprintf(stderr, "Elementwise op requires matching dtypes\n");
        return NULL;
    }

    /* 计算 broadcast 形状 */
    Shape output_shape;
    if (ops_broadcast_shape(&a->shape, &b->shape, &output_shape) != 0) {
        return NULL;
    }

    size_t numel = shape_numel(&output_shape);

    /* 分配输出数据 */
    float* output_data = (float*)malloc(numel * sizeof(float));
    if (!output_data) {
        return NULL;
    }

    /* 用于存储多维索引 */
    size_t output_indices[MYLLM_MAX_NDIM];

    /* 逐元素计算 */
    for (size_t linear_idx = 0; linear_idx < numel; linear_idx++) {
        /* 将线性索引转换为多维索引 */
        linear_to_multi_idx(linear_idx, output_shape.dims, output_shape.ndim, output_indices);

        /* 计算 a 的线性索引 (考虑 broadcast) */
        size_t a_linear_idx = compute_broadcast_linear_index(
            output_indices, output_shape.ndim, &a->shape
        );

        /* 计算 b 的线性索引 (考虑 broadcast) */
        size_t b_linear_idx = compute_broadcast_linear_index(
            output_indices, output_shape.ndim, &b->shape
        );

        /* 读取值 */
        float a_val = tensor_get_f32(a, a_linear_idx);
        float b_val = tensor_get_f32(b, b_linear_idx);

        /* 应用运算 */
        output_data[linear_idx] = op(a_val, b_val);
    }

    /* 创建输出张量 */
    Tensor* result = tensor_from_f32(output_data, output_shape.dims, output_shape.ndim, a->dtype);
    free(output_data);

    return result;
}

/* ============================================================================
 * 运算函数
 * ============================================================================ */

static float op_add(float a, float b) { return a + b; }
static float op_sub(float a, float b) { return a - b; }
static float op_mul(float a, float b) { return a * b; }
static float op_div(float a, float b) {
    return (b == 0.0f) ? INFINITY : (a / b);
}

Tensor* ops_add(const Tensor* a, const Tensor* b) {
    return tensor_elementwise_op(a, b, op_add);
}

Tensor* ops_sub(const Tensor* a, const Tensor* b) {
    return tensor_elementwise_op(a, b, op_sub);
}

Tensor* ops_mul(const Tensor* a, const Tensor* b) {
    return tensor_elementwise_op(a, b, op_mul);
}

Tensor* ops_div(const Tensor* a, const Tensor* b) {
    return tensor_elementwise_op(a, b, op_div);
}
