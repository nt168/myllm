/**
 * @file elementwise.h
 * @brief 逐元素运算 - 对应 phyllm/src/ops/elementwise.rs
 *
 * 支持 broadcasting 的逐元素运算:
 * - Add, Sub, Mul, Div
 */

#ifndef MYLLM_OPS_ELEMENTWISE_H
#define MYLLM_OPS_ELEMENTWISE_H

#include "tensor/tensor.h"
#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

/* ============================================================================
 * 逐元素运算
 * ============================================================================ */

/**
 * @brief 张量加法 (支持 broadcasting)
 * @param a 输入张量 a
 * @param b 输入张量 b
 * @return 结果张量，失败返回 NULL
 */
Tensor* ops_add(const Tensor* a, const Tensor* b);

/**
 * @brief 张量减法 (支持 broadcasting)
 * @param a 输入张量 a
 * @param b 输入张量 b
 * @return 结果张量，失败返回 NULL
 */
Tensor* ops_sub(const Tensor* a, const Tensor* b);

/**
 * @brief 张量乘法 (支持 broadcasting)
 * @param a 输入张量 a
 * @param b 输入张量 b
 * @return 结果张量，失败返回 NULL
 */
Tensor* ops_mul(const Tensor* a, const Tensor* b);

/**
 * @brief 张量除法 (支持 broadcasting)
 * @param a 输入张量 a
 * @param b 输入张量 b
 * @return 结果张量，失败返回 NULL
 */
Tensor* ops_div(const Tensor* a, const Tensor* b);

/* ============================================================================
 * Broadcasting 辅助函数
 * ============================================================================ */

/**
 * @brief 计算 broadcast 后的形状
 * @param a 形状 a
 * @param b 形状 b
 * @param result 输出形状
 * @return 成功返回 0，失败返回 -1
 */
int ops_broadcast_shape(const Shape* a, const Shape* b, Shape* result);

#ifdef __cplusplus
}
#endif

#endif /* MYLLM_OPS_ELEMENTWISE_H */
