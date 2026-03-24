/**
 * @file myllm.h
 * @brief MyLLM - 纯 C 语言 LLM 推理框架
 *
 * 对应 phyllm (Rust) 的 C 语言实现
 */

#ifndef MYLLM_H
#define MYLLM_H

/* 版本信息 */
#define MYLLM_VERSION_MAJOR 0
#define MYLLM_VERSION_MINOR 1
#define MYLLM_VERSION_PATCH 0

/* 模型类型 */
#include "models/model_types.h"

/* LLaMA 模型 */
#include "models/llama/llama.h"

/* Qwen2 模型 */
#include "models/qwen2/qwen2.h"

/* Qwen3 模型 */
#include "models/qwen3/qwen3.h"

/**
 * @brief 获取版本字符串
 */
static inline const char* myllm_version(void) {
    return "0.1.0";
}

#endif /* MYLLM_H */
