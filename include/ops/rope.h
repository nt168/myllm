/**
 * @file rope.h
 * @brief RoPE (旋转位置编码) 运算
 */

#ifndef MYLLM_OPS_ROPE_H
#define MYLLM_OPS_ROPE_H

#include "tensor/tensor.h"
#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

/**
 * @brief 应用 RoPE 位置编码
 *
 * @param x 输入张量
 * @param positions 位置数组
 * @param num_positions 位置数量
 * @param theta RoPE theta 参数
 * @return 应用了 RoPE 的新张量
 */
Tensor* ops_rope(const Tensor* x, const size_t* positions, size_t num_positions, double theta);

#ifdef __cplusplus
}
#endif

#endif /* MYLLM_OPS_ROPE_H */
