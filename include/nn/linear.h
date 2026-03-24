/**
 * @file linear.h
 * @brief 线性层 (全连接层) - 对应 phyllm/src/nn/layers/linear.rs
 *
 * 计算: output = input @ weight.T + bias
 *
 * Weight 形状: [out_features, in_features]
 * Bias 形状: [out_features]
 */

#ifndef MYLLM_NN_LINEAR_H
#define MYLLM_NN_LINEAR_H

#include "tensor/tensor.h"
#include <stddef.h>
#include <stdbool.h>

#ifdef __cplusplus
extern "C" {
#endif

/**
 * @brief 线性层结构
 */
typedef struct NN_Linear {
    Tensor* weight;         /**< 权重 [out_features, in_features] */
    Tensor* bias;           /**< 偏置 [out_features]，可为 NULL */
    size_t in_features;     /**< 输入特征数 */
    size_t out_features;    /**< 输出特征数 */
} NN_Linear;

/**
 * @brief 创建线性层
 * @param in_features 输入特征数
 * @param out_features 输出特征数
 * @param use_bias 是否使用偏置
 * @param dtype 数据类型
 * @return 新的线性层，失败返回 NULL
 */
NN_Linear* nn_linear_new(size_t in_features, size_t out_features, bool use_bias, DType dtype);

/**
 * @brief 从权重创建线性层
 * @param weight 权重张量 [out_features, in_features]
 * @param bias 偏置张量 [out_features]，可为 NULL
 * @return 新的线性层，失败返回 NULL
 */
NN_Linear* nn_linear_from_weights(Tensor* weight, Tensor* bias);

/**
 * @brief 释放线性层
 */
void nn_linear_free(NN_Linear* linear);

/**
 * @brief 前向传播
 * @param linear 线性层
 * @param input 输入张量 [batch, in_features] 或 [in_features]
 * @return 输出张量，失败返回 NULL
 */
Tensor* nn_linear_forward(NN_Linear* linear, const Tensor* input);

/**
 * @brief 获取输入特征数
 */
static inline size_t nn_linear_in_features(const NN_Linear* linear) {
    return linear->in_features;
}

/**
 * @brief 获取输出特征数
 */
static inline size_t nn_linear_out_features(const NN_Linear* linear) {
    return linear->out_features;
}

/**
 * @brief 获取权重
 */
static inline Tensor* nn_linear_weight(const NN_Linear* linear) {
    return linear->weight;
}

/**
 * @brief 获取偏置
 */
static inline Tensor* nn_linear_bias(const NN_Linear* linear) {
    return linear->bias;
}

#ifdef __cplusplus
}
#endif

#endif /* MYLLM_NN_LINEAR_H */
