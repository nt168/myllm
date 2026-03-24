/**
 * @file mlp.h
 * @brief 前馈网络 (SwiGLU) - 对应 phyllm/src/nn/layers/mlp.rs
 *
 * 实现 SwiGLU 变体的前馈网络:
 * output = down(silu(gate(x)) * up(x))
 *
 * 其中:
 * - gate(x) 是线性投影
 * - up(x) 是线性投影
 * - silu 是 SiLU/Swish 激活函数
 * - down 是最终线性投影
 */

#ifndef MYLLM_NN_MLP_H
#define MYLLM_NN_MLP_H

#include "nn/linear.h"
#include "tensor/tensor.h"
#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

/**
 * @brief MLP 层结构 (SwiGLU)
 */
typedef struct NN_MLP {
    NN_Linear* gate_proj;      /**< 门控投影 [hidden_dim -> intermediate_dim] */
    NN_Linear* up_proj;        /**< 上投影 [hidden_dim -> intermediate_dim] */
    NN_Linear* down_proj;      /**< 下投影 [intermediate_dim -> hidden_dim] */
    size_t hidden_dim;         /**< 隐藏维度 */
    size_t intermediate_dim;   /**< 中间维度 */
} NN_MLP;

/**
 * @brief 创建 MLP 层
 * @param hidden_dim 隐藏维度
 * @param intermediate_dim 中间维度 (通常是 hidden_dim * 4)
 * @param dtype 数据类型
 * @return 新的 MLP 层，失败返回 NULL
 */
NN_MLP* nn_mlp_new(size_t hidden_dim, size_t intermediate_dim, DType dtype);

/**
 * @brief 释放 MLP 层
 */
void nn_mlp_free(NN_MLP* mlp);

/**
 * @brief 前向传播
 * @param mlp MLP 层
 * @param input 输入张量 [batch, seq_len, hidden_dim] 或 [seq_len, hidden_dim]
 * @return 输出张量，失败返回 NULL
 */
Tensor* nn_mlp_forward(NN_MLP* mlp, const Tensor* input);

/**
 * @brief 获取隐藏维度
 */
static inline size_t nn_mlp_hidden_dim(const NN_MLP* mlp) {
    return mlp->hidden_dim;
}

/**
 * @brief 获取中间维度
 */
static inline size_t nn_mlp_intermediate_dim(const NN_MLP* mlp) {
    return mlp->intermediate_dim;
}

/**
 * @brief 获取门控投影
 */
static inline NN_Linear* nn_mlp_gate_proj(const NN_MLP* mlp) {
    return mlp->gate_proj;
}

/**
 * @brief 获取上投影
 */
static inline NN_Linear* nn_mlp_up_proj(const NN_MLP* mlp) {
    return mlp->up_proj;
}

/**
 * @brief 获取下投影
 */
static inline NN_Linear* nn_mlp_down_proj(const NN_MLP* mlp) {
    return mlp->down_proj;
}

#ifdef __cplusplus
}
#endif

#endif /* MYLLM_NN_MLP_H */
