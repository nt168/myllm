/**
 * @file matmul.c
 * @brief 矩阵乘法实现
 */

#include "ops/matmul.h"
#include <stdlib.h>
#include <string.h>
#include <stdio.h>

/* ============================================================================
 * 内部辅助函数
 * ============================================================================ */

/**
 * @brief 简单的矩阵乘法实现 (用于小矩阵或作为 fallback)
 */
static void matmul_simple(
    const float* a,
    const float* b,
    float* c,
    size_t m,
    size_t n,
    size_t k,
    size_t a_stride0,
    size_t a_stride1,
    size_t b_stride0,
    size_t b_stride1
) {
    for (size_t i = 0; i < m; i++) {
        for (size_t j = 0; j < n; j++) {
            float sum = 0.0f;
            for (size_t l = 0; l < k; l++) {
                float a_val = a[i * a_stride0 + l * a_stride1];
                float b_val = b[l * b_stride0 + j * b_stride1];
                sum += a_val * b_val;
            }
            c[i * n + j] = sum;
        }
    }
}

/**
 * @brief 分块矩阵乘法 (优化缓存利用)
 */
#define BLOCK_SIZE 32

static void matmul_blocked(
    const float* a,
    const float* b,
    float* c,
    size_t m,
    size_t n,
    size_t k,
    size_t a_stride0,
    size_t a_stride1,
    size_t b_stride0,
    size_t b_stride1
) {
    /* 初始化 C */
    memset(c, 0, m * n * sizeof(float));

    /* 分块计算 */
    for (size_t i0 = 0; i0 < m; i0 += BLOCK_SIZE) {
        size_t i_end = (i0 + BLOCK_SIZE < m) ? i0 + BLOCK_SIZE : m;

        for (size_t j0 = 0; j0 < n; j0 += BLOCK_SIZE) {
            size_t j_end = (j0 + BLOCK_SIZE < n) ? j0 + BLOCK_SIZE : n;

            for (size_t l0 = 0; l0 < k; l0 += BLOCK_SIZE) {
                size_t l_end = (l0 + BLOCK_SIZE < k) ? l0 + BLOCK_SIZE : k;

                /* 计算小块 */
                for (size_t i = i0; i < i_end; i++) {
                    for (size_t l = l0; l < l_end; l++) {
                        float a_val = a[i * a_stride0 + l * a_stride1];

                        for (size_t j = j0; j < j_end; j++) {
                            float b_val = b[l * b_stride0 + j * b_stride1];
                            c[i * n + j] += a_val * b_val;
                        }
                    }
                }
            }
        }
    }
}

/* ============================================================================
 * 2D 矩阵乘法
 * ============================================================================ */

static Tensor* matmul_2d(const Tensor* a, const Tensor* b) {
    const Shape* a_shape = tensor_shape(a);
    const Shape* b_shape = tensor_shape(b);

    size_t m = a_shape->dims[0];
    size_t k1 = a_shape->dims[1];
    size_t k2 = b_shape->dims[0];
    size_t n = b_shape->dims[1];

    if (k1 != k2) {
        fprintf(stderr, "MatMul: incompatible shapes [%zu, %zu] x [%zu, %zu]\n",
                m, k1, k2, n);
        return NULL;
    }
    size_t k = k1;

    /* 分配输出 */
    float* c_data = (float*)calloc(m * n, sizeof(float));
    if (!c_data) {
        return NULL;
    }

    /* 获取输入数据指针 */
    const float* a_data = (const float*)tensor_data_ptr_const(a);
    const float* b_data = (const float*)tensor_data_ptr_const(b);

    /* 获取 strides */
    size_t a_stride0 = a->strides[0];
    size_t a_stride1 = a->strides[1];
    size_t b_stride0 = b->strides[0];
    size_t b_stride1 = b->strides[1];

    /* 使用分块矩阵乘法 */
    matmul_blocked(a_data, b_data, c_data, m, n, k,
                   a_stride0, a_stride1, b_stride0, b_stride1);

    /* 创建输出张量 */
    size_t out_dims[2] = { m, n };
    Tensor* result = tensor_from_f32(c_data, out_dims, 2, a->dtype);
    free(c_data);

    return result;
}

/* ============================================================================
 * 3D 批量矩阵乘法
 * ============================================================================ */

static Tensor* batch_matmul_3d(const Tensor* a, const Tensor* b) {
    const Shape* a_shape = tensor_shape(a);
    const Shape* b_shape = tensor_shape(b);

    size_t batch_a = a_shape->dims[0];
    size_t m = a_shape->dims[1];
    size_t k1 = a_shape->dims[2];

    size_t batch_b = b_shape->dims[0];
    size_t k2 = b_shape->dims[1];
    size_t n = b_shape->dims[2];

    if (batch_a != batch_b || k1 != k2) {
        fprintf(stderr, "MatMul: incompatible batch shapes [%zu, %zu, %zu] x [%zu, %zu, %zu]\n",
                batch_a, m, k1, batch_b, k2, n);
        return NULL;
    }

    size_t batch = batch_a;
    size_t k = k1;

    /* 分配输出 */
    float* c_data = (float*)calloc(batch * m * n, sizeof(float));
    if (!c_data) {
        return NULL;
    }

    const float* a_data = (const float*)tensor_data_ptr_const(a);
    const float* b_data = (const float*)tensor_data_ptr_const(b);

    size_t a_batch_stride = a->strides[0];
    size_t b_batch_stride = b->strides[0];
    size_t a_stride0 = a->strides[1];
    size_t a_stride1 = a->strides[2];
    size_t b_stride0 = b->strides[1];
    size_t b_stride1 = b->strides[2];

    /* 逐批计算 */
    for (size_t b_idx = 0; b_idx < batch; b_idx++) {
        const float* a_batch = a_data + b_idx * a_batch_stride;
        const float* b_batch = b_data + b_idx * b_batch_stride;
        float* c_batch = c_data + b_idx * m * n;

        matmul_blocked(a_batch, b_batch, c_batch, m, n, k,
                       a_stride0, a_stride1, b_stride0, b_stride1);
    }

    /* 创建输出张量 */
    size_t out_dims[3] = { batch, m, n };
    Tensor* result = tensor_from_f32(c_data, out_dims, 3, a->dtype);
    free(c_data);

    return result;
}

/* ============================================================================
 * 主接口
 * ============================================================================ */

Tensor* ops_matmul(const Tensor* a, const Tensor* b) {
    if (!a || !b) {
        return NULL;
    }

    /* 目前只支持 F32 */
    if (tensor_dtype(a) != DTYPE_F32 || tensor_dtype(b) != DTYPE_F32) {
        fprintf(stderr, "MatMul: only F32 is supported\n");
        return NULL;
    }

    const Shape* a_shape = tensor_shape(a);
    const Shape* b_shape = tensor_shape(b);

    size_t a_ndim = a_shape->ndim;
    size_t b_ndim = b_shape->ndim;

    /* 2D x 2D */
    if (a_ndim == 2 && b_ndim == 2) {
        return matmul_2d(a, b);
    }

    /* 3D x 3D */
    if (a_ndim == 3 && b_ndim == 3) {
        return batch_matmul_3d(a, b);
    }

    fprintf(stderr, "MatMul: unsupported dimensions %zuD x %zuD\n", a_ndim, b_ndim);
    return NULL;
}
