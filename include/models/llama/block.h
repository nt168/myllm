/**
 * @file block.h
 * @brief LLaMA Transformer 块 - 对应 phyllm/src/models/llama/block.rs
 *
 * LLaMA Transformer 块包含:
 * - 多头自注意力 (带 RoPE)
 * - 前馈 MLP (SwiGLU)
 * - Pre-norm 残差连接
 */

#ifndef MYLLM_LLAMA_BLOCK_H
#define MYLLM_LLAMA_BLOCK_H

#include "../model_types.h"
#include "config.h"
#include "attention.h"

#ifdef __cplusplus
extern "C" {
#endif

/* ============================================================================
 * RMSNorm 层
 * ============================================================================ */

/**
 * @brief RMSNorm 层
 */
typedef struct {
    Tensor* weight;     /**< 归一化权重 [hidden_dim] */
    size_t hidden_dim;
    float eps;
} RMSNorm;

/**
 * @brief 创建 RMSNorm 层
 */
RMSNorm* rmsnorm_new(size_t hidden_dim, float eps);

/**
 * @brief 释放 RMSNorm 层
 */
void rmsnorm_free(RMSNorm* layer);

/**
 * @brief RMSNorm 前向传播
 */
Tensor* rmsnorm_forward(RMSNorm* layer, const Tensor* x);

/* ============================================================================
 * MLP 层 (SwiGLU)
 * ============================================================================ */

/**
 * @brief MLP 层 (SwiGLU 激活)
 */
typedef struct {
    Linear* gate_proj;      /**< Gate 投影 [intermediate_dim, hidden_dim] */
    Linear* up_proj;        /**< Up 投影 [intermediate_dim, hidden_dim] */
    Linear* down_proj;      /**< Down 投影 [hidden_dim, intermediate_dim] */
    size_t hidden_dim;
    size_t intermediate_dim;
} MLP;

/**
 * @brief 创建 MLP 层
 */
MLP* mlp_new(size_t hidden_dim, size_t intermediate_dim);

/**
 * @brief 释放 MLP 层
 */
void mlp_free(MLP* layer);

/**
 * @brief MLP 前向传播 (SwiGLU: down(silu(gate(x)) * up(x)))
 */
Tensor* mlp_forward(MLP* layer, const Tensor* x);

/* ============================================================================
 * LLaMA Transformer 块
 * ============================================================================ */

/**
 * @brief LLaMA Transformer 块结构
 */
typedef struct {
    LlamaAttention* attention;          /**< 注意力层 */
    MLP* mlp;                           /**< MLP层 */
    RMSNorm* input_norm;                /**< 输入归一化 */
    RMSNorm* post_attention_norm;       /**< 注意力后归一化 */
} LlamaTransformerBlock;

/**
 * @brief 创建 LLaMA Transformer 块
 */
LlamaTransformerBlock* llama_block_new(
    size_t hidden_dim,
    size_t num_heads,
    size_t num_kv_heads,
    size_t head_dim,
    size_t intermediate_dim,
    float norm_eps,
    double rope_theta
);

/**
 * @brief 从配置创建 LLaMA Transformer 块
 */
LlamaTransformerBlock* llama_block_from_config(const LlamaConfig* config);

/**
 * @brief 释放 LLaMA Transformer 块
 */
void llama_block_free(LlamaTransformerBlock* block);

/**
 * @brief Transformer 块前向传播 (带位置信息)
 */
Tensor* llama_block_forward_with_positions(
    LlamaTransformerBlock* block,
    const Tensor* x,
    const size_t* positions,
    size_t num_positions
);

/**
 * @brief 使用预计算的 K, V 进行前向传播
 */
Tensor* llama_block_forward_with_kv(
    LlamaTransformerBlock* block,
    const Tensor* x,
    const Tensor* k_cached,
    const Tensor* v_cached,
    const size_t* positions,
    size_t num_positions
);

/**
 * @brief 基本前向传播 (无位置信息, 无缓存)
 */
Tensor* llama_block_forward(LlamaTransformerBlock* block, const Tensor* x);

/**
 * @brief 计算 K, V 投影并应用 RoPE
 */
int llama_block_compute_kv_with_rope(
    LlamaTransformerBlock* block,
    const Tensor* x,
    const size_t* positions,
    size_t num_positions,
    Tensor** k_out,
    Tensor** v_out
);

/**
 * @brief 访问器: 获取注意力层
 */
LlamaAttention* llama_block_attention(LlamaTransformerBlock* block);

/**
 * @brief 访问器: 获取 MLP 层
 */
MLP* llama_block_mlp(LlamaTransformerBlock* block);

/**
 * @brief 访问器: 获取输入归一化层
 */
RMSNorm* llama_block_input_norm(LlamaTransformerBlock* block);

/**
 * @brief 访问器: 获取注意力后归一化层
 */
RMSNorm* llama_block_post_attention_norm(LlamaTransformerBlock* block);

#ifdef __cplusplus
}
#endif

#endif /* MYLLM_LLAMA_BLOCK_H */
