/**
 * @file engine.h
 * @brief 推理引擎模块
 *
 * 整合所有模块，提供完整的文本生成功能:
 * - 模型加载
 * - 分词
 * - 推理 (Prefill + Decode)
 * - 采样
 * - 流式输出
 */

#ifndef MYLLM_ENGINE_H
#define MYLLM_ENGINE_H

#include "loader/loader.h"
#include "tokenizer/tokenizer.h"
#include "sampler/sampler.h"
#include "tensor/tensor.h"
#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

/* ============================================================================
 * 常量定义
 * ============================================================================ */

/** 默认最大生成长度 */
#define ENGINE_DEFAULT_MAX_TOKENS 256

/** 默认批大小 */
#define ENGINE_DEFAULT_BATCH_SIZE 1

/** 最大 KV 缓存长度 (避免大模型 OOM) */
#define ENGINE_MAX_CACHE_LEN 2048

/* ============================================================================
 * 回调类型
 * ============================================================================ */

/**
 * @brief 生成回调函数类型
 *
 * @param token 生成的 token ID
 * @param token_str token 字符串 (可能为 NULL)
 * @param user_data 用户数据
 * @return true 继续生成, false 停止生成
 */
typedef bool (*GenerateCallback)(
    int32_t token,
    const char* token_str,
    void* user_data
);

/* ============================================================================
 * 生成配置
 * ============================================================================ */

/**
 * @brief 生成配置
 */
typedef struct GenerateConfig {
    int32_t max_tokens;             /**< 最大生成 token 数 */
    float temperature;              /**< 采样温度 */
    int32_t top_k;                  /**< Top-K 采样 */
    float top_p;                    /**< Top-P 采样 */
    float repetition_penalty;       /**< 重复惩罚 */
    int32_t* stop_tokens;           /**< 停止 token 列表 */
    size_t num_stop_tokens;         /**< 停止 token 数量 */
    bool add_bos;                   /**< 是否添加 BOS token */
    bool add_eos;                   /**< 是否添加 EOS token */
    bool stream;                    /**< 是否流式输出 */
    uint64_t seed;                  /**< 随机种子 */
} GenerateConfig;

/**
 * @brief 获取默认生成配置
 */
GenerateConfig generate_config_default(void);

/* ============================================================================
 * 生成结果
 * ============================================================================ */

/**
 * @brief 生成结果
 */
typedef struct GenerateResult {
    int32_t* tokens;                /**< 生成的 token 数组 */
    size_t num_tokens;              /**< 生成的 token 数量 */
    char* text;                     /**< 生成的文本 */
    size_t text_len;                /**< 文本长度 */
    bool finished;                  /**< 是否正常结束 */
    int32_t stop_token;             /**< 停止 token (-1 表示达到 max_tokens) */
    float prefill_time_ms;          /**< Prefill 耗时 (毫秒) */
    float decode_time_ms;           /**< Decode 耗时 (毫秒) */
    float tokens_per_second;        /**< 生成速度 (tokens/s) */
} GenerateResult;

/**
 * @brief 释放生成结果
 */
void generate_result_free(GenerateResult* result);

/* ============================================================================
 * 推理引擎
 * ============================================================================ */

/**
 * @brief 推理引擎结构 (不透明指针)
 */
typedef struct InferenceEngine InferenceEngine;

/* ============================================================================
 * 生命周期管理
 * ============================================================================ */

/**
 * @brief 从模型目录创建推理引擎
 *
 * @param model_dir 模型目录路径
 * @return 成功返回引擎指针，失败返回 NULL
 */
InferenceEngine* engine_new(const char* model_dir);

/**
 * @brief 从已加载的组件创建推理引擎
 *
 * @param loader 权重加载器
 * @param tokenizer 分词器
 * @param config 模型配置
 * @return 成功返回引擎指针，失败返回 NULL
 */
InferenceEngine* engine_new_with_components(
    WeightLoader* loader,
    BPETokenizer* tokenizer,
    const ModelConfig* config
);

/**
 * @brief 释放推理引擎
 */
void engine_free(InferenceEngine* engine);

/* ============================================================================
 * 模型信息
 * ============================================================================ */

/**
 * @brief 获取模型配置
 */
const ModelConfig* engine_get_config(const InferenceEngine* engine);

/**
 * @brief 获取词表大小
 */
size_t engine_vocab_size(const InferenceEngine* engine);

/**
 * @brief 获取模型层数
 */
size_t engine_num_layers(const InferenceEngine* engine);

/**
 * @brief 获取隐藏维度
 */
size_t engine_hidden_size(const InferenceEngine* engine);

/* ============================================================================
 * 分词器接口
 * ============================================================================ */

/**
 * @brief 编码文本为 token IDs
 *
 * @param engine 引擎
 * @param text 输入文本
 * @param add_special 是否添加特殊 token
 * @param ids 输出 token ID 数组
 * @param max_ids 数组最大容量
 * @return 实际编码的 token 数量
 */
int engine_encode(
    const InferenceEngine* engine,
    const char* text,
    bool add_special,
    int32_t* ids,
    size_t max_ids
);

/**
 * @brief 解码 token IDs 为文本
 *
 * @param engine 引擎
 * @param ids token ID 数组
 * @param num_ids token 数量
 * @param skip_special 是否跳过特殊 token
 * @param out 输出缓冲区
 * @param max_out 缓冲区最大容量
 * @return 实际解码的字符数
 */
int engine_decode(
    const InferenceEngine* engine,
    const int32_t* ids,
    size_t num_ids,
    bool skip_special,
    char* out,
    size_t max_out
);

/**
 * @brief 获取特殊 token ID
 */
int32_t engine_bos_id(const InferenceEngine* engine);
int32_t engine_eos_id(const InferenceEngine* engine);
int32_t engine_pad_id(const InferenceEngine* engine);

/* ============================================================================
 * 推理接口
 * ============================================================================ */

/**
 * @brief Prefill: 处理输入序列
 *
 * @param engine 引擎
 * @param tokens 输入 token IDs
 * @param num_tokens token 数量
 * @return 成功返回 logits 张量，失败返回 NULL
 * @note 调用者负责释放返回的张量
 */
Tensor* engine_prefill(
    InferenceEngine* engine,
    const int32_t* tokens,
    size_t num_tokens
);

/**
 * @brief Decode Step: 生成单个 token 的推理
 *
 * @param engine 引擎
 * @param token 当前 token ID
 * @return 成功返回 logits 张量，失败返回 NULL
 * @note 调用者负责释放返回的张量
 */
Tensor* engine_decode_step(
    InferenceEngine* engine,
    int32_t token
);

/**
 * @brief 重置 KV 缓存
 */
void engine_reset_cache(InferenceEngine* engine);

/**
 * @brief 获取当前 KV 缓存长度
 */
size_t engine_cache_len(const InferenceEngine* engine);

/* ============================================================================
 * 生成接口
 * ============================================================================ */

/**
 * @brief 从文本生成回复
 *
 * @param engine 引擎
 * @param prompt 输入提示文本
 * @param config 生成配置
 * @return 生成结果
 */
GenerateResult engine_generate(
    InferenceEngine* engine,
    const char* prompt,
    const GenerateConfig* config
);

/**
 * @brief 从 token IDs 生成回复
 *
 * @param engine 引擎
 * @param prompt_tokens 输入 token IDs
 * @param num_prompt_tokens token 数量
 * @param config 生成配置
 * @return 生成结果
 */
GenerateResult engine_generate_from_tokens(
    InferenceEngine* engine,
    const int32_t* prompt_tokens,
    size_t num_prompt_tokens,
    const GenerateConfig* config
);

/**
 * @brief 流式生成
 *
 * @param engine 引擎
 * @param prompt 输入提示文本
 * @param config 生成配置
 * @param callback 回调函数
 * @param user_data 用户数据
 * @return 生成的 token 数量
 */
size_t engine_generate_stream(
    InferenceEngine* engine,
    const char* prompt,
    const GenerateConfig* config,
    GenerateCallback callback,
    void* user_data
);

/* ============================================================================
 * 批处理接口
 * ============================================================================ */

/**
 * @brief 批量 Prefill
 *
 * @param engine 引擎
 * @param batch_tokens 批量 token IDs [batch][seq]
 * @param batch_sizes 每个 batch 的 token 数量
 * @param batch_size 批大小
 * @return 成功返回 logits 张量，失败返回 NULL
 */
Tensor* engine_prefill_batch(
    InferenceEngine* engine,
    const int32_t** batch_tokens,
    const size_t* batch_sizes,
    size_t batch_size
);

#ifdef __cplusplus
}
#endif

#endif /* MYLLM_ENGINE_H */
