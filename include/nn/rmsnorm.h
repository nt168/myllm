/**
 * @file rmsnorm.h
 * @brief RMS 归一化层 - 对应 phyllm/src/nn/layers/norm.rs
 *
 * 计算: output = RMSNorm(x) * weight
 *
 * 带有可学习 weight (gamma) 参数的 RMSNorm 层
 */

#ifndef MYLLM_NN_RMSNORM_H
#define MYLLM_NN_RMSNORM_H

#include "tensor/tensor.h"
#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

/**
 * @brief RMSNorm 层结构
 */
typedef struct NN_RMSNorm {
    Tensor* weight;             /**< 权重 [normalized_shape] */
    size_t normalized_shape;    /**< 归一化的维度大小 */
    float eps;                  /**< 防止除零的小值 */
} NN_RMSNorm;

/**
 * @brief 创建 RMSNorm 层
 * @param normalized_shape 归一化的维度大小
 * @param eps 防止除零的小值
 * @param dtype 数据类型
 * @return 新的 RMSNorm 层，失败返回 NULL
 */
NN_RMSNorm* nn_rmsnorm_new(size_t normalized_shape, float eps, DType dtype);

/**
 * @brief 从权重创建 RMSNorm 层
 * @param weight 权重张量 [normalized_shape]
 * @param eps 防止除零的小值
 * @return 新的 RMSNorm 层，失败返回 NULL
 */
NN_RMSNorm* nn_rmsnorm_from_weights(Tensor* weight, float eps);

/**
 * @brief 释放 RMSNorm 层
 */
void nn_rmsnorm_free(NN_RMSNorm* rmsnorm);

/**
 * @brief 前向传播
 * @param rmsnorm RMSNorm 层
 * @param input 输入张量 [..., normalized_shape]
 * @return 输出张量，失败返回 NULL
 */
Tensor* nn_rmsnorm_forward(NN_RMSNorm* rmsnorm, const Tensor* input);

/**
 * @brief 获取归一化维度大小
 */
static inline size_t nn_rmsnorm_normalized_shape(const NN_RMSNorm* rmsnorm) {
    return rmsnorm->normalized_shape;
}

/**
 * @brief 获取 eps
 */
static inline float nn_rmsnorm_eps(const NN_RMSNorm* rmsnorm) {
    return rmsnorm->eps;
}

/**
 * @brief 获取权重
 */
static inline Tensor* nn_rmsnorm_weight(const NN_RMSNorm* rmsnorm) {
    return rmsnorm->weight;
}

#ifdef __cplusplus
}
#endif

#endif /* MYLLM_NN_RMSNORM_H */
