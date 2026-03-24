/**
 * @file model.h
 * @brief Qwen2 模型 - 对应 phyllm/src/models/qwen2/model.rs
 *
 * Qwen2 模型架构与 LLaMA 类似，但有以下区别:
 * - 注意力投影使用 bias=True
 * - 可选的滑动窗口注意力
 * - 默认共享词嵌入权重
 */

#ifndef MYLLM_QWEN2_MODEL_H
#define MYLLM_QWEN2_MODEL_H

#include "../model_types.h"
#include "config.h"
#include "../llama/block.h"
#include "../llama/model.h"  /* For Embedding, KVCache, Linear, RMSNorm */

#ifdef __cplusplus
extern "C" {
#endif

/* ============================================================================
 * Qwen2 注意力层
 * ============================================================================ */

/**
 * @brief Qwen2 注意力层 (带 bias 和可选滑动窗口)
 */
typedef struct {
    Linear* q_proj;         /**< Query 投影 (带 bias) */
    Linear* k_proj;         /**< Key 投影 (带 bias) */
    Linear* v_proj;         /**< Value 投影 (带 bias) */
    Linear* o_proj;         /**< Output 投影 (带 bias) */

    size_t hidden_dim;
    size_t num_heads;
    size_t num_kv_heads;
    size_t head_dim;
    float scale;            /**< 1/sqrt(head_dim) */
    double rope_theta;

    /* Qwen2 特有 */
    size_t sliding_window;      /**< 滑动窗口大小 */
    bool use_sliding_window;    /**< 是否使用滑动窗口 */
} Qwen2Attention;

/**
 * @brief 创建 Qwen2 注意力层
 */
Qwen2Attention* qwen2_attention_new(
    size_t hidden_dim,
    size_t num_heads,
    size_t num_kv_heads,
    size_t head_dim,
    double rope_theta,
    size_t sliding_window,
    bool use_bias
);

/**
 * @brief 释放 Qwen2 注意力层
 */
void qwen2_attention_free(Qwen2Attention* attn);

/**
 * @brief 计算 K, V 并应用 RoPE
 */
int qwen2_attention_compute_kv_with_rope(
    Qwen2Attention* attn,
    const Tensor* x,
    const size_t* positions,
    size_t num_positions,
    Tensor** k_out,
    Tensor** v_out
);

/**
 * @brief 使用预计算的 K, V 进行前向传播
 */
Tensor* qwen2_attention_forward_with_kv(
    Qwen2Attention* attn,
    const Tensor* x,
    const Tensor* k_cached,
    const Tensor* v_cached,
    const size_t* positions,
    size_t num_positions
);

/**
 * @brief 带滑动窗口的前向传播
 */
Tensor* qwen2_attention_forward_with_sliding_window(
    Qwen2Attention* attn,
    const Tensor* x,
    const Tensor* k_cached,
    const Tensor* v_cached,
    const size_t* positions,
    size_t num_positions,
    size_t window_size
);

/* ============================================================================
 * Qwen2 Transformer 块
 * ============================================================================ */

/**
 * @brief Qwen2 Transformer 块
 */
typedef struct {
    Qwen2Attention* attention;          /**< Qwen2 注意力层 */
    MLP* mlp;                           /**< MLP层 (SwiGLU) */
    RMSNorm* input_norm;                /**< 输入归一化 */
    RMSNorm* post_attention_norm;       /**< 注意力后归一化 */
} Qwen2TransformerBlock;

/**
 * @brief 创建 Qwen2 Transformer 块
 */
Qwen2TransformerBlock* qwen2_block_new(const Qwen2Config* config);

/**
 * @brief 释放 Qwen2 Transformer 块
 */
void qwen2_block_free(Qwen2TransformerBlock* block);

/**
 * @brief 带位置信息的前向传播
 */
Tensor* qwen2_block_forward_with_positions(
    Qwen2TransformerBlock* block,
    const Tensor* x,
    const size_t* positions,
    size_t num_positions
);

/**
 * @brief 计算 K, V 并应用 RoPE
 */
int qwen2_block_compute_kv_with_rope(
    Qwen2TransformerBlock* block,
    const Tensor* x,
    const size_t* positions,
    size_t num_positions,
    Tensor** k_out,
    Tensor** v_out
);

/* ============================================================================
 * Qwen2 模型
 * ============================================================================ */

/**
 * @brief Qwen2 模型结构
 */
typedef struct {
    Embedding* embed_tokens;                /**< Token 嵌入层 */
    Qwen2TransformerBlock** layers;         /**< Transformer 层数组 */
    RMSNorm* norm;                          /**< 最终归一化层 */
    Linear* lm_head;                        /**< 输出投影 (通常为NULL, 与embed共享) */

    Qwen2Config config;                     /**< 模型配置 */
    KVCache** kv_caches;                    /**< 每层的 KV 缓存 */
    size_t num_layers;                      /**< 层数 */
    bool has_cache;                         /**< 是否启用缓存 */
} Qwen2Model;

/**
 * @brief 创建 Qwen2 模型 (无缓存)
 */
Qwen2Model* qwen2_model_new(const Qwen2Config* config);

/**
 * @brief 创建 Qwen2 模型 (带 KV 缓存)
 */
Qwen2Model* qwen2_model_new_with_cache(const Qwen2Config* config, size_t batch_size);

/**
 * @brief 从 LoadedConfig 创建 Qwen2 模型
 */
Qwen2Model* qwen2_model_from_loaded_config(const LoadedConfig* config);

/**
 * @brief 从 LoadedConfig 创建 Qwen2 模型 (带缓存)
 */
Qwen2Model* qwen2_model_from_loaded_config_with_cache(const LoadedConfig* config, size_t batch_size);

/**
 * @brief 释放 Qwen2 模型
 */
void qwen2_model_free(Qwen2Model* model);

/**
 * @brief Prefill 阶段
 */
Tensor* qwen2_model_prefill(Qwen2Model* model, const int32_t* tokens, size_t num_tokens);

/**
 * @brief Decode 阶段
 */
Tensor* qwen2_model_decode_step(Qwen2Model* model, int32_t token, size_t position);

/**
 * @brief 基本前向传播
 */
Tensor* qwen2_model_forward(Qwen2Model* model, const Tensor* input_ids);

/**
 * @brief 重置 KV 缓存
 */
void qwen2_model_reset_cache(Qwen2Model* model);

/**
 * @brief 获取层数
 */
size_t qwen2_model_num_layers(const Qwen2Model* model);

/**
 * @brief 获取词表大小
 */
size_t qwen2_model_vocab_size(const Qwen2Model* model);

/**
 * @brief 检查是否有缓存
 */
bool qwen2_model_has_cache(const Qwen2Model* model);

/**
 * @brief 获取缓存长度
 */
size_t qwen2_model_cache_len(const Qwen2Model* model);

/**
 * @brief 获取模型名称
 */
const char* qwen2_model_name(const Qwen2Model* model);

#ifdef __cplusplus
}
#endif

#endif /* MYLLM_QWEN2_MODEL_H */
