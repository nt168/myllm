/**
 * @file matmul.h
 * @brief 矩阵乘法 - 对应 phyllm/src/ops/matmul.rs
 *
 * 支持:
 * - 2D 矩阵乘法: (M, K) x (K, N) -> (M, N)
 * - 3D 批量矩阵乘法: (B, M, K) x (B, K, N) -> (B, M, N)
 */

#ifndef MYLLM_OPS_MATMUL_H
#define MYLLM_OPS_MATMUL_H

#include "tensor/tensor.h"

#ifdef __cplusplus
extern "C" {
#endif

/**
 * @brief 矩阵乘法
 *
 * 支持 2D 和 3D 张量:
 * - 2D: (M, K) x (K, N) -> (M, N)
 * - 3D: (B, M, K) x (B, K, N) -> (B, M, N)
 *
 * @param a 左矩阵
 * @param b 右矩阵
 * @return 结果矩阵，失败返回 NULL
 */
Tensor* ops_matmul(const Tensor* a, const Tensor* b);

#ifdef __cplusplus
}
#endif

#endif /* MYLLM_OPS_MATMUL_H */
