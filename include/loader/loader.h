/**
 * @file loader.h
 * @brief 模型加载器模块
 *
 * 提供统一的模型加载接口，支持多种格式:
 * - SafeTensors (.safetensors)
 * - GGUF (.gguf) - 未来支持
 */

#ifndef MYLLM_LOADER_H
#define MYLLM_LOADER_H

#include "safetensors.h"
#include "models/model_types.h"
#include <stdbool.h>
#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

/* ============================================================================
 * 模型配置
 * ============================================================================ */

/**
 * @brief 模型配置 (从 config.json 解析)
 */
typedef struct ModelConfig {
    char architecture[128];         /**< 模型架构 (e.g., "LlamaForCausalLM") */
    char model_type[64];            /**< 模型类型 (e.g., "llama", "qwen2") */

    size_t hidden_size;             /**< 隐藏层维度 */
    size_t intermediate_size;       /**< FFN 中间维度 */
    size_t num_attention_heads;     /**< 注意力头数 */
    size_t num_hidden_layers;       /**< 隐藏层数 */
    size_t vocab_size;              /**< 词表大小 */
    size_t num_key_value_heads;     /**< KV 头数 (GQA) */
    size_t head_dim;                /**< 头维度 */
    size_t max_position_embeddings; /**< 最大位置嵌入 */

    double rope_theta;              /**< RoPE theta */
    float rms_norm_eps;             /**< RMSNorm epsilon */

    DType torch_dtype;              /**< 数据类型 */
    bool tie_word_embeddings;       /**< 是否共享嵌入权重 */
} ModelConfig;

/**
 * @brief 初始化默认配置
 *
 * @param config 配置结构指针
 */
void model_config_init(ModelConfig* config);

/**
 * @brief 从 JSON 文件加载模型配置
 *
 * @param path config.json 文件路径
 * @param config 输出配置结构
 * @return 成功返回 true
 */
bool model_config_load(const char* path, ModelConfig* config);

/* ============================================================================
 * 权重加载器
 * ============================================================================ */

/**
 * @brief 权重加载器结构
 */
typedef struct WeightLoader {
    SafeTensorsLoader* safetensors; /**< SafeTensors 加载器 */
    ModelConfig config;             /**< 模型配置 */
    char model_dir[512];            /**< 模型目录路径 */
} WeightLoader;

/**
 * @brief 从目录创建权重加载器
 *
 * 自动检测并加载:
 * - config.json
 * - model.safetensors 或分片文件
 *
 * @param model_dir 模型目录路径
 * @return 成功返回加载器指针，失败返回 NULL
 */
WeightLoader* weight_loader_new(const char* model_dir);

/**
 * @brief 释放权重加载器
 *
 * @param loader 加载器指针
 */
void weight_loader_free(WeightLoader* loader);

/**
 * @brief 获取模型配置
 *
 * @param loader 加载器指针
 * @return 配置指针
 */
const ModelConfig* weight_loader_config(const WeightLoader* loader);

/**
 * @brief 加载指定张量
 *
 * @param loader 加载器指针
 * @param name 张量名称
 * @return 成功返回 Tensor 指针，失败返回 NULL
 */
Tensor* weight_loader_load(const WeightLoader* loader, const char* name);

/**
 * @brief 加载指定张量并转换为 F32
 *
 * @param loader 加载器指针
 * @param name 张量名称
 * @return 成功返回 F32 Tensor 指针，失败返回 NULL
 */
Tensor* weight_loader_load_f32(const WeightLoader* loader, const char* name);

/**
 * @brief 检查张量是否存在
 *
 * @param loader 加载器指针
 * @param name 张量名称
 * @return 存在返回 true
 */
bool weight_loader_has_tensor(const WeightLoader* loader, const char* name);

/**
 * @brief 获取张量原始数据指针 (零拷贝)
 *
 * @param loader 加载器指针
 * @param name 张量名称
 * @return 数据指针，失败返回 NULL
 */
const void* weight_loader_get_raw(const WeightLoader* loader, const char* name);

/* ============================================================================
 * 便捷宏
 * ============================================================================ */

/**
 * @brief 加载嵌入层权重
 */
#define WEIGHT_LOADER_EMBED(loader, name) \
    weight_loader_load_f32(loader, name ".weight")

/**
 * @brief 加载线性层权重
 */
#define WEIGHT_LOADER_LINEAR_W(loader, name) \
    weight_loader_load_f32(loader, name ".weight")

/**
 * @brief 加载线性层偏置
 */
#define WEIGHT_LOADER_LINEAR_B(loader, name) \
    weight_loader_load_f32(loader, name ".bias")

/**
 * @brief 加载 RMSNorm 权重
 */
#define WEIGHT_LOADER_RMSNORM(loader, name) \
    weight_loader_load_f32(loader, name ".weight")

#ifdef __cplusplus
}
#endif

#endif /* MYLLM_LOADER_H */
