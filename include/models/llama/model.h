/**
 * @file model.h
 * @brief LLaMA 模型 - 对应 phyllm/src/models/llama/model.rs
 *
 * LLaMA 语言模型:
 * - Token 嵌入层
 * - 堆叠的 Transformer 块 (带 GQA 注意力)
 * - 最终 RMSNorm
 * - 可选的 lm_head (输出投影, 可与 embed_tokens 共享权重)
 */

#ifndef MYLLM_LLAMA_MODEL_H
#define MYLLM_LLAMA_MODEL_H

#include "../model_types.h"
#include "config.h"
#include "block.h"
#include "kv/kv.h"  /* 使用独立的 KV 缓存模块 */

#ifdef __cplusplus
extern "C" {
#endif

/* ============================================================================
 * 嵌入层
 * ============================================================================ */

/**
 * @brief 嵌入层结构
 */
typedef struct {
    Tensor* weight;             /**< 嵌入权重 [vocab_size, hidden_dim] */
    size_t num_embeddings;      /**< 词表大小 */
    size_t embedding_dim;       /**< 嵌入维度 */
} Embedding;

/**
 * @brief 创建嵌入层
 */
Embedding* embedding_new(size_t num_embeddings, size_t embedding_dim);

/**
 * @brief 释放嵌入层
 */
void embedding_free(Embedding* layer);

/**
 * @brief 嵌入层前向传播
 * @param layer 嵌入层
 * @param input_ids 输入 token IDs, shape [batch, seq_len], dtype I32
 * @return 嵌入输出, shape [batch, seq_len, hidden_dim]
 */
Tensor* embedding_forward(Embedding* layer, const Tensor* input_ids);

/**
 * @brief 获取嵌入权重
 */
Tensor* embedding_weight(Embedding* layer);

/* ============================================================================
 * LLaMA 模型
 * ============================================================================ */

/**
 * @brief LLaMA 模型结构
 */
typedef struct {
    Embedding* embed_tokens;                /**< Token 嵌入层 */
    LlamaTransformerBlock** layers;         /**< Transformer 层数组 */
    RMSNorm* norm;                          /**< 最终归一化层 */
    Linear* lm_head;                        /**< 输出投影 (可为NULL, 与embed共享) */

    LlamaConfig config;                     /**< 模型配置 */
    KVCache** kv_caches;                    /**< 每层的 KV 缓存 */
    size_t num_layers;                      /**< 层数 */
    bool has_cache;                         /**< 是否启用缓存 */
} LlamaModel;

/* ============================================================================
 * 模型创建与释放
 * ============================================================================ */

/**
 * @brief 创建 LLaMA 模型 (无缓存)
 */
LlamaModel* llama_model_new(const LlamaConfig* config);

/**
 * @brief 创建 LLaMA 模型 (带 KV 缓存)
 */
LlamaModel* llama_model_new_with_cache(const LlamaConfig* config, size_t batch_size);

/**
 * @brief 从 LoadedConfig 创建 LLaMA 模型
 */
LlamaModel* llama_model_from_loaded_config(const LoadedConfig* config);

/**
 * @brief 从 LoadedConfig 创建 LLaMA 模型 (带缓存)
 */
LlamaModel* llama_model_from_loaded_config_with_cache(const LoadedConfig* config, size_t batch_size);

/**
 * @brief 释放 LLaMA 模型
 */
void llama_model_free(LlamaModel* model);

/* ============================================================================
 * 模型属性
 * ============================================================================ */

/**
 * @brief 获取层数
 */
size_t llama_model_num_layers(const LlamaModel* model);

/**
 * @brief 获取词表大小
 */
size_t llama_model_vocab_size(const LlamaModel* model);

/**
 * @brief 检查是否有 KV 缓存
 */
bool llama_model_has_cache(const LlamaModel* model);

/**
 * @brief 获取 KV 缓存当前长度
 */
size_t llama_model_cache_len(const LlamaModel* model);

/**
 * @brief 获取模型名称
 */
const char* llama_model_name(const LlamaModel* model);

/* ============================================================================
 * 模型推理
 * ============================================================================ */

/**
 * @brief Prefill 阶段: 处理整个输入序列
 *
 * @param model LLaMA 模型
 * @param tokens 输入 token ID 数组
 * @param num_tokens token 数量
 * @return logits 张量, shape [1, 1, vocab_size]
 */
Tensor* llama_model_prefill(LlamaModel* model, const int32_t* tokens, size_t num_tokens);

/**
 * @brief Decode 阶段: 生成单个 token
 *
 * @param model LLaMA 模型
 * @param token 当前 token ID
 * @param position 序列中的位置
 * @return logits 张量, shape [1, 1, vocab_size]
 */
Tensor* llama_model_decode_step(LlamaModel* model, int32_t token, size_t position);

/**
 * @brief 基本前向传播 (无缓存, 用于训练或测试)
 *
 * @param model LLaMA 模型
 * @param input_ids 输入 token IDs, shape [batch, seq_len]
 * @return logits 张量, shape [batch, seq_len, vocab_size]
 */
Tensor* llama_model_forward(LlamaModel* model, const Tensor* input_ids);

/**
 * @brief 重置 KV 缓存
 */
void llama_model_reset_cache(LlamaModel* model);

/* ============================================================================
 * 层访问器
 * ============================================================================ */

/**
 * @brief 获取指定层的 Transformer 块
 */
LlamaTransformerBlock* llama_model_layer(LlamaModel* model, size_t idx);

/**
 * @brief 获取嵌入层
 */
Embedding* llama_model_embed_tokens(LlamaModel* model);

/**
 * @brief 获取最终归一化层
 */
RMSNorm* llama_model_norm(LlamaModel* model);

/**
 * @brief 获取输出投影层 (可能为NULL)
 */
Linear* llama_model_lm_head(LlamaModel* model);

#ifdef __cplusplus
}
#endif

#endif /* MYLLM_LLAMA_MODEL_H */
