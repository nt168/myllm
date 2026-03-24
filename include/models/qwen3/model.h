/**
 * @file model.h
 * @brief Qwen3 模型 - 对应 phyllm/src/models/qwen3/model.rs
 *
 * Qwen3 模型架构:
 * - Token 嵌入层
 * - 堆叠的 Transformer 块 (带 GQA 注意力)
 * - 最终 RMSNorm
 * - 可选的 lm_head (可与 embed_tokens 共享权重)
 *
 * 主要特性:
 * - Grouped Query Attention (GQA)
 * - RoPE 位置编码
 * - SwiGLU 激活函数
 * - RMSNorm 归一化
 */

#ifndef MYLLM_QWEN3_MODEL_H
#define MYLLM_QWEN3_MODEL_H

#include "../model_types.h"
#include "config.h"
#include "../llama/block.h"
#include "../llama/model.h"  /* For Embedding, KVCache, Linear, RMSNorm */

#ifdef __cplusplus
extern "C" {
#endif

/* ============================================================================
 * Qwen3 注意力层
 * ============================================================================ */

/**
 * @brief Qwen3 注意力层结构
 *
 * 与 LLaMA 注意力类似，但使用 GQA 和 RoPE
 */
typedef struct {
    Linear* q_proj;         /**< Query 投影 */
    Linear* k_proj;         /**< Key 投影 */
    Linear* v_proj;         /**< Value 投影 */
    Linear* o_proj;         /**< Output 投影 */

    /* Q/K Norm (可选) */
    RMSNorm* q_norm;        /**< Query 归一化 (可选) */
    RMSNorm* k_norm;        /**< Key 归一化 (可选) */

    size_t hidden_dim;
    size_t num_heads;
    size_t num_kv_heads;
    size_t head_dim;
    float scale;            /**< 1/sqrt(head_dim) */
    double rope_theta;
    bool use_qk_norm;       /**< 是否使用 Q/K Norm */
} Qwen3Attention;

/**
 * @brief 创建 Qwen3 注意力层
 */
Qwen3Attention* qwen3_attention_new(
    size_t hidden_dim,
    size_t num_heads,
    size_t num_kv_heads,
    size_t head_dim,
    double rope_theta,
    bool use_qk_norm
);

/**
 * @brief 释放 Qwen3 注意力层
 */
void qwen3_attention_free(Qwen3Attention* attn);

/**
 * @brief 计算 K, V 并应用 RoPE
 */
int qwen3_attention_compute_kv_with_rope(
    Qwen3Attention* attn,
    const Tensor* x,
    const size_t* positions,
    size_t num_positions,
    Tensor** k_out,
    Tensor** v_out
);

/**
 * @brief 使用预计算的 K, V 进行前向传播
 */
Tensor* qwen3_attention_forward_with_kv(
    Qwen3Attention* attn,
    const Tensor* x,
    const Tensor* k_cached,
    const Tensor* v_cached,
    const size_t* positions,
    size_t num_positions
);

/**
 * @brief 带位置信息的前向传播
 */
Tensor* qwen3_attention_forward_with_positions(
    Qwen3Attention* attn,
    const Tensor* x,
    const size_t* positions,
    size_t num_positions
);

/* ============================================================================
 * Qwen3 Transformer 块
 * ============================================================================ */

/**
 * @brief Qwen3 Transformer 块结构
 */
typedef struct {
    Qwen3Attention* attention;          /**< Qwen3 注意力层 */
    MLP* mlp;                           /**< MLP层 (SwiGLU) */
    RMSNorm* input_norm;                /**< 输入归一化 */
    RMSNorm* post_attention_norm;       /**< 注意力后归一化 */
} Qwen3TransformerBlock;

/**
 * @brief 创建 Qwen3 Transformer 块
 */
Qwen3TransformerBlock* qwen3_block_new(const Qwen3Config* config);

/**
 * @brief 从参数创建 Qwen3 Transformer 块
 */
Qwen3TransformerBlock* qwen3_block_new_with_head_dim(
    size_t hidden_dim,
    size_t num_heads,
    size_t num_kv_heads,
    size_t head_dim,
    size_t intermediate_dim,
    float norm_eps,
    double rope_theta
);

/**
 * @brief 释放 Qwen3 Transformer 块
 */
void qwen3_block_free(Qwen3TransformerBlock* block);

/**
 * @brief 带位置信息的前向传播
 */
Tensor* qwen3_block_forward_with_positions(
    Qwen3TransformerBlock* block,
    const Tensor* x,
    const size_t* positions,
    size_t num_positions
);

/**
 * @brief 计算 K, V 并应用 RoPE
 */
int qwen3_block_compute_kv_with_rope(
    Qwen3TransformerBlock* block,
    const Tensor* x,
    const size_t* positions,
    size_t num_positions,
    Tensor** k_out,
    Tensor** v_out
);

/**
 * @brief 使用预计算的 K, V 进行前向传播
 */
Tensor* qwen3_block_forward_with_kv(
    Qwen3TransformerBlock* block,
    const Tensor* x,
    const Tensor* k_cached,
    const Tensor* v_cached,
    const size_t* positions,
    size_t num_positions
);

/**
 * @brief 访问器
 */
Qwen3Attention* qwen3_block_attention(Qwen3TransformerBlock* block);
RMSNorm* qwen3_block_input_norm(Qwen3TransformerBlock* block);
RMSNorm* qwen3_block_post_attention_norm(Qwen3TransformerBlock* block);
MLP* qwen3_block_mlp(Qwen3TransformerBlock* block);

/* ============================================================================
 * Qwen3 模型
 * ============================================================================ */

/**
 * @brief Qwen3 模型结构
 *
 * 完整的 Qwen3 Transformer 模型:
 * - Token 嵌入层
 * - 堆叠的预归一化 Transformer 块
 * - 最终 RMSNorm 层
 * - 可选的输出投影 (lm_head)
 * - 可选的模型层 KV 缓存管理
 */
typedef struct {
    Embedding* embed_tokens;                /**< Token 嵌入层 */
    Qwen3TransformerBlock** layers;         /**< Transformer 层数组 */
    RMSNorm* norm;                          /**< 最终归一化层 */
    Linear* lm_head;                        /**< 输出投影 (可为NULL) */

    Qwen3Config config;                     /**< 模型配置 */
    KVCache** kv_caches;                    /**< 每层的 KV 缓存 */
    size_t num_layers;                      /**< 层数 */
    bool has_cache;                         /**< 是否启用缓存 */
} Qwen3Model;

/* ============================================================================
 * 模型创建与释放
 * ============================================================================ */

/**
 * @brief 创建 Qwen3 模型 (无缓存)
 */
Qwen3Model* qwen3_model_new(const Qwen3Config* config);

/**
 * @brief 创建 Qwen3 模型 (带 KV 缓存)
 */
Qwen3Model* qwen3_model_new_with_cache(const Qwen3Config* config, size_t batch_size);

/**
 * @brief 从 LoadedConfig 创建 Qwen3 模型
 */
Qwen3Model* qwen3_model_from_loaded_config(const LoadedConfig* config);

/**
 * @brief 从 LoadedConfig 创建 Qwen3 模型 (带缓存)
 */
Qwen3Model* qwen3_model_from_loaded_config_with_cache(const LoadedConfig* config, size_t batch_size);

/**
 * @brief 释放 Qwen3 模型
 */
void qwen3_model_free(Qwen3Model* model);

/* ============================================================================
 * 模型属性
 * ============================================================================ */

/**
 * @brief 获取层数
 */
size_t qwen3_model_num_layers(const Qwen3Model* model);

/**
 * @brief 获取词表大小
 */
size_t qwen3_model_vocab_size(const Qwen3Model* model);

/**
 * @brief 检查是否有 KV 缓存
 */
bool qwen3_model_has_cache(const Qwen3Model* model);

/**
 * @brief 获取 KV 缓存当前长度
 */
size_t qwen3_model_cache_len(const Qwen3Model* model);

/**
 * @brief 获取模型名称
 */
const char* qwen3_model_name(const Qwen3Model* model);

/* ============================================================================
 * 模型推理
 * ============================================================================ */

/**
 * @brief Prefill 阶段: 处理整个输入序列
 *
 * @param model Qwen3 模型
 * @param tokens 输入 token ID 数组
 * @param num_tokens token 数量
 * @return logits 张量, shape [1, 1, vocab_size]
 */
Tensor* qwen3_model_prefill(Qwen3Model* model, const int32_t* tokens, size_t num_tokens);

/**
 * @brief Decode 阶段: 生成单个 token
 *
 * @param model Qwen3 模型
 * @param token 当前 token ID
 * @param position 序列中的位置
 * @return logits 张量. shape [1, 1, vocab_size]
 */
Tensor* qwen3_model_decode_step(Qwen3Model* model, int32_t token, size_t position);

/**
 * @brief 基本前向传播 (无缓存, 用于训练或测试)
 *
 * @param model Qwen3 模型
 * @param input_ids 输入 token IDs, shape [batch, seq_len]
 * @return logits 张量. shape [batch, seq_len, vocab_size]
 */
Tensor* qwen3_model_forward(Qwen3Model* model, const Tensor* input_ids);

/**
 * @brief 重置 KV 缓存
 */
void qwen3_model_reset_cache(Qwen3Model* model);

/**
 * @brief 获取指定层的 Transformer 块
 */
Qwen3TransformerBlock* qwen3_model_layer(Qwen3Model* model, size_t idx);

/**
 * @brief 获取嵌入层
 */
Embedding* qwen3_model_embed_tokens(Qwen3Model* model);

/**
 * @brief 获取最终归一化层
 */
RMSNorm* qwen3_model_norm(Qwen3Model* model);

/**
 * @brief 获取输出投影层 (可能为NULL)
 */
Linear* qwen3_model_lm_head(Qwen3Model* model);

/**
 * @brief 是否共享词嵌入权重
 */
bool qwen3_model_tie_word_embeddings(const Qwen3Model* model);

#ifdef __cplusplus
}
#endif

#endif /* MYLLM_QWEN3_MODEL_H */
