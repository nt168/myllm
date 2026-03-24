/**
 * @file attention.h
 * @brief LLaMA 注意力层 - 对应 phyllm/src/models/llama/attention.rs
 *
 * LLaMA 使用标准的 Grouped-Query Attention:
 * - RoPE (旋转位置编码)
 * - 无 Q/K Norm
 * - 所有投影无 bias (默认 LLaMA 配置)
 */

#ifndef MYLLM_LLAMA_ATTENTION_H
#define MYLLM_LLAMA_ATTENTION_H

#include "tensor/tensor.h"
#include "config.h"

#ifdef __cplusplus
extern "C" {
#endif

/* ============================================================================
 * 线性层结构
 * ============================================================================ */

/**
 * @brief 线性层结构
 */
typedef struct {
    Tensor* weight;     /**< 权重 [out_features, in_features] */
    Tensor* bias;       /**< 偏置 [out_features] (可为NULL) */
    size_t in_features;
    size_t out_features;
    bool use_bias;
} Linear;

/**
 * @brief 创建线性层
 */
Linear* linear_new(size_t in_features, size_t out_features, bool use_bias);

/**
 * @brief 释放线性层
 */
void linear_free(Linear* layer);

/**
 * @brief 线性层前向传播: y = x @ W^T + b
 */
Tensor* linear_forward(Linear* layer, const Tensor* x);

/* ============================================================================
 * GQA 注意力层
 * ============================================================================ */

/**
 * @brief GQA 注意力配置
 */
typedef struct {
    size_t hidden_dim;
    size_t num_heads;
    size_t num_kv_heads;
    size_t head_dim;
    double rope_theta;
    bool use_q_norm;
    bool use_k_norm;
    bool use_bias;
} GQAAttentionConfig;

/**
 * @brief GQA 注意力层结构
 */
typedef struct {
    Linear* q_proj;         /**< Query 投影 */
    Linear* k_proj;         /**< Key 投影 */
    Linear* v_proj;         /**< Value 投影 */
    Linear* o_proj;         /**< Output 投影 */

    /* 可选的 Q/K Norm */
    Tensor* q_norm_weight;  /**< Q 归一化权重 (可为NULL) */
    Tensor* k_norm_weight;  /**< K 归一化权重 (可为NULL) */
    float norm_eps;         /**< 归一化 epsilon */

    /* 配置 */
    size_t hidden_dim;
    size_t num_heads;
    size_t num_kv_heads;
    size_t head_dim;
    float scale;            /**< 1/sqrt(head_dim) */
    double rope_theta;
    bool use_q_norm;
    bool use_k_norm;
} GQAAttention;

/**
 * @brief 创建 GQA 注意力层
 */
GQAAttention* gqa_attention_new(
    size_t hidden_dim,
    size_t num_heads,
    size_t num_kv_heads,
    size_t head_dim,
    double rope_theta,
    bool use_q_norm,
    bool use_k_norm,
    bool use_bias
);

/**
 * @brief 释放 GQA 注意力层
 */
void gqa_attention_free(GQAAttention* attn);

/**
 * @brief 计算 Q, K, V 投影
 */
int gqa_attention_compute_qkv(
    GQAAttention* attn,
    const Tensor* x,
    Tensor** q_out,
    Tensor** k_out,
    Tensor** v_out
);

/**
 * @brief 计算 K, V 投影并对 K 应用 RoPE
 */
int gqa_attention_compute_kv_with_rope(
    GQAAttention* attn,
    const Tensor* x,
    const size_t* positions,
    size_t num_positions,
    Tensor** k_out,
    Tensor** v_out
);

/**
 * @brief 使用预计算的 K, V 进行前向传播
 */
Tensor* gqa_attention_forward_with_kv(
    GQAAttention* attn,
    const Tensor* x,
    const Tensor* k_cached,
    const Tensor* v_cached,
    const size_t* positions,
    size_t num_positions
);

/**
 * @brief 带位置信息的前向传播 (无缓存)
 */
Tensor* gqa_attention_forward_with_positions(
    GQAAttention* attn,
    const Tensor* x,
    const size_t* positions,
    size_t num_positions
);

/**
 * @brief 基本前向传播
 */
Tensor* gqa_attention_forward(GQAAttention* attn, const Tensor* x);

/* ============================================================================
 * LLaMA 注意力层
 * ============================================================================ */

/**
 * @brief LLaMA 注意力层 (封装 GQAAttention)
 */
typedef struct {
    GQAAttention* inner;
} LlamaAttention;

/**
 * @brief 创建 LLaMA 注意力层
 */
LlamaAttention* llama_attention_new(
    size_t hidden_dim,
    size_t num_heads,
    size_t num_kv_heads,
    size_t head_dim,
    double rope_theta
);

/**
 * @brief 释放 LLaMA 注意力层
 */
void llama_attention_free(LlamaAttention* attn);

/**
 * @brief 计算 K, V 并应用 RoPE
 */
int llama_attention_compute_kv_with_rope(
    LlamaAttention* attn,
    const Tensor* x,
    const size_t* positions,
    size_t num_positions,
    Tensor** k_out,
    Tensor** v_out
);

/**
 * @brief 使用预计算的 K, V 进行前向传播
 */
Tensor* llama_attention_forward_with_kv(
    LlamaAttention* attn,
    const Tensor* x,
    const Tensor* k_cached,
    const Tensor* v_cached,
    const size_t* positions,
    size_t num_positions
);

/**
 * @brief 带位置信息的前向传播
 */
Tensor* llama_attention_forward_with_positions(
    LlamaAttention* attn,
    const Tensor* x,
    const size_t* positions,
    size_t num_positions
);

/**
 * @brief 获取各投影层的访问器
 */
Linear* llama_attention_q_proj(LlamaAttention* attn);
Linear* llama_attention_k_proj(LlamaAttention* attn);
Linear* llama_attention_v_proj(LlamaAttention* attn);
Linear* llama_attention_o_proj(LlamaAttention* attn);

#ifdef __cplusplus
}
#endif

#endif /* MYLLM_LLAMA_ATTENTION_H */
