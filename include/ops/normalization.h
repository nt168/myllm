/**
 * @file normalization.h
 * @brief 归一化运算 - 对应 phyllm/src/ops/normalization.rs
 *
 * 支持的归一化:
 * - LayerNorm
 * - RMSNorm
 * - Softmax
 */

#ifndef MYLLM_OPS_NORMALIZATION_H
#define MYLLM_OPS_NORMALIZATION_H

#include "tensor/tensor.h"
#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

/**
 * @brief Layer Normalization
 *
 * 公式:
 * - mean = mean(x, dim)
 * - var = variance(x, dim)
 * - output = (x - mean) / sqrt(var + eps)
 *
 * @param input 输入张量
 * @param normalized_shape 归一化的维度大小 (最后一个维度)
 * @param weight 权重 (gamma)，NULL 则使用 1
 * @param bias 偏置 (beta)，NULL 则使用 0
 * @param eps 防止除零的小值
 * @return 输出张量，失败返回 NULL
 */
Tensor* ops_layernorm(const Tensor* input, size_t normalized_shape,
                       const float* weight, const float* bias, float eps);

/**
 * @brief RMS Normalization
 *
 * 公式:
 * - rms = sqrt(mean(x^2))
 * - output = x / (rms + eps)
 *
 * @param input 输入张量
 * @param normalized_shape 归一化的维度大小 (最后一个维度)
 * @param weight 权重 (gamma)，NULL 则使用 1
 * @param eps 防止除零的小值
 * @return 输出张量，失败返回 NULL
 */
Tensor* ops_rmsnorm(const Tensor* input, size_t normalized_shape,
                     const float* weight, float eps);

/**
 * @brief Softmax
 *
 * 公式: softmax(x)_i = exp(x_i - max(x)) / sum(exp(x - max(x)))
 *
 * @param input 输入张量
 * @param dim 计算 softmax 的维度
 * @return 输出张量，失败返回 NULL
 */
Tensor* ops_softmax(const Tensor* input, size_t dim);

#ifdef __cplusplus
}
#endif

#endif /* MYLLM_OPS_NORMALIZATION_H */
