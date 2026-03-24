/**
 * @file embedding.h
 * @brief 词嵌入层 - 对应 phyllm/src/nn/layers/embedding.rs
 *
 * 从权重矩阵中查找 token 嵌入向量
 *
 * Weight 形状: [vocab_size, hidden_dim]
 * Input: [seq_len] (token IDs as i32)
 * Output: [seq_len, hidden_dim]
 */

#ifndef MYLLM_NN_EMBEDDING_H
#define MYLLM_NN_EMBEDDING_H

#include "tensor/tensor.h"
#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

/**
 * @brief 嵌入层结构
 */
typedef struct NN_Embedding {
    Tensor* weight;         /**< 权重 [vocab_size, hidden_dim] */
    size_t vocab_size;      /**< 词表大小 */
    size_t hidden_dim;      /**< 隐藏维度 */
} NN_Embedding;

/**
 * @brief 创建嵌入层
 * @param vocab_size 词表大小
 * @param hidden_dim 隐藏维度
 * @param dtype 数据类型
 * @return 新的嵌入层，失败返回 NULL
 */
NN_Embedding* nn_embedding_new(size_t vocab_size, size_t hidden_dim, DType dtype);

/**
 * @brief 从权重创建嵌入层
 * @param weight 权重张量 [vocab_size, hidden_dim]
 * @return 新的嵌入层，失败返回 NULL
 */
NN_Embedding* nn_embedding_from_weights(Tensor* weight);

/**
 * @brief 释放嵌入层
 */
void nn_embedding_free(NN_Embedding* embedding);

/**
 * @brief 前向传播 (1D 输入)
 * @param embedding 嵌入层
 * @param token_ids token ID 数组
 * @param seq_len 序列长度
 * @return 输出张量 [seq_len, hidden_dim]，失败返回 NULL
 */
Tensor* nn_embedding_forward(NN_Embedding* embedding, const int32_t* token_ids, size_t seq_len);

/**
 * @brief 前向传播 (从张量输入)
 * @param embedding 嵌入层
 * @param input 输入张量 [seq_len] 或 [batch, seq_len] (I32 类型)
 * @return 输出张量，失败返回 NULL
 */
Tensor* nn_embedding_forward_tensor(NN_Embedding* embedding, const Tensor* input);

/**
 * @brief 获取词表大小
 */
static inline size_t nn_embedding_vocab_size(const NN_Embedding* embedding) {
    return embedding->vocab_size;
}

/**
 * @brief 获取隐藏维度
 */
static inline size_t nn_embedding_hidden_dim(const NN_Embedding* embedding) {
    return embedding->hidden_dim;
}

/**
 * @brief 获取权重
 */
static inline Tensor* nn_embedding_weight(const NN_Embedding* embedding) {
    return embedding->weight;
}

#ifdef __cplusplus
}
#endif

#endif /* MYLLM_NN_EMBEDDING_H */
