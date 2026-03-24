/**
 * @file activation.h
 * @brief 激活函数 - 对应 phyllm/src/ops/activation.rs
 *
 * 支持的激活函数:
 * - GELU (Gaussian Error Linear Unit)
 * - SiLU (Sigmoid Linear Unit / Swish)
 * - Sigmoid
 * - ReLU (Rectified Linear Unit)
 */

#ifndef MYLLM_OPS_ACTIVATION_H
#define MYLLM_OPS_ACTIVATION_H

#include "tensor/tensor.h"

#ifdef __cplusplus
extern "C" {
#endif

/**
 * @brief GELU 激活函数
 *
 * 公式: 0.5 * x * (1 + tanh(sqrt(2/π) * (x + 0.044715 * x³)))
 *
 * @param input 输入张量
 * @return 输出张量，失败返回 NULL
 */
Tensor* ops_gelu(const Tensor* input);

/**
 * @brief SiLU (Swish) 激活函数
 *
 * 公式: x * sigmoid(x) = x / (1 + exp(-x))
 *
 * @param input 输入张量
 * @return 输出张量，失败返回 NULL
 */
Tensor* ops_silu(const Tensor* input);

/**
 * @brief Sigmoid 激活函数
 *
 * 公式: 1 / (1 + exp(-x))
 *
 * @param input 输入张量
 * @return 输出张量，失败返回 NULL
 */
Tensor* ops_sigmoid(const Tensor* input);

/**
 * @brief ReLU 激活函数
 *
 * 公式: max(0, x)
 *
 * @param input 输入张量
 * @return 输出张量，失败返回 NULL
 */
Tensor* ops_relu(const Tensor* input);

#ifdef __cplusplus
}
#endif

#endif /* MYLLM_OPS_ACTIVATION_H */
