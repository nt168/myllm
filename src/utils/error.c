/**
 * @file error.c
 * @brief 错误处理工具实现
 */

#include "utils/error.h"
#include <string.h>
#include <stdio.h>
#include <stdarg.h>

/* 错误名称 */
static const char* error_names[] = {
    [MYLLM_OK] = "OK",

    /* 通用错误 */
    [MYLLM_ERROR_UNKNOWN] = "ERROR_UNKNOWN",
    [MYLLM_ERROR_INVALID_ARG] = "ERROR_INVALID_ARG",
    [MYLLM_ERROR_NULL_POINTER] = "ERROR_NULL_POINTER",
    [MYLLM_ERROR_OUT_OF_MEMORY] = "ERROR_OUT_OF_MEMORY",
    [MYLLM_ERROR_NOT_SUPPORTED] = "ERROR_NOT_SUPPORTED",
    [MYLLM_ERROR_NOT_IMPLEMENTED] = "ERROR_NOT_IMPLEMENTED",
    [MYLLM_ERROR_OVERFLOW] = "ERROR_OVERFLOW",
    [MYLLM_ERROR_UNDERFLOW] = "ERROR_UNDERFLOW",
    [MYLLM_ERROR_TIMEOUT] = "ERROR_TIMEOUT",

    /* 张量错误 */
    [MYLLM_ERROR_TENSOR_SHAPE] = "ERROR_TENSOR_SHAPE",
    [MYLLM_ERROR_TENSOR_DTYPE] = "ERROR_TENSOR_DTYPE",
    [MYLLM_ERROR_TENSOR_DEVICE] = "ERROR_TENSOR_DEVICE",
    [MYLLM_ERROR_TENSOR_STRIDE] = "ERROR_TENSOR_STRIDE",
    [MYLLM_ERROR_TENSOR_INDEX] = "ERROR_TENSOR_INDEX",
    [MYLLM_ERROR_TENSOR_BROADCAST] = "ERROR_TENSOR_BROADCAST",

    /* 缓存错误 */
    [MYLLM_ERROR_CACHE_OVERFLOW] = "ERROR_CACHE_OVERFLOW",
    [MYLLM_ERROR_CACHE_EMPTY] = "ERROR_CACHE_EMPTY",
    [MYLLM_ERROR_CACHE_INVALID] = "ERROR_CACHE_INVALID",

    /* 加载错误 */
    [MYLLM_ERROR_LOAD_FILE] = "ERROR_LOAD_FILE",
    [MYLLM_ERROR_LOAD_FORMAT] = "ERROR_LOAD_FORMAT",
    [MYLLM_ERROR_LOAD_CHECKSUM] = "ERROR_LOAD_CHECKSUM",
    [MYLLM_ERROR_LOAD_VERSION] = "ERROR_LOAD_VERSION",

    /* 模型错误 */
    [MYLLM_ERROR_MODEL_INIT] = "ERROR_MODEL_INIT",
    [MYLLM_ERROR_MODEL_WEIGHT] = "ERROR_MODEL_WEIGHT",
    [MYLLM_ERROR_MODEL_CONFIG] = "ERROR_MODEL_CONFIG",
    [MYLLM_ERROR_MODEL_INFER] = "ERROR_MODEL_INFER",

    /* 分词器错误 */
    [MYLLM_ERROR_TOKENIZER_INIT] = "ERROR_TOKENIZER_INIT",
    [MYLLM_ERROR_TOKENIZER_ENCODE] = "ERROR_TOKENIZER_ENCODE",
    [MYLLM_ERROR_TOKENIZER_DECODE] = "ERROR_TOKENIZER_DECODE",

    /* 量化错误 */
    [MYLLM_ERROR_QUANT_RANGE] = "ERROR_QUANT_RANGE",
    [MYLLM_ERROR_QUANT_PRECISION] = "ERROR_QUANT_PRECISION",
    [MYLLM_ERROR_QUANT_FORMAT] = "ERROR_QUANT_FORMAT",
};

/* 错误消息 */
static const char* error_messages[] = {
    [MYLLM_OK] = "Success",

    /* 通用错误 */
    [MYLLM_ERROR_UNKNOWN] = "Unknown error",
    [MYLLM_ERROR_INVALID_ARG] = "Invalid argument",
    [MYLLM_ERROR_NULL_POINTER] = "Null pointer",
    [MYLLM_ERROR_OUT_OF_MEMORY] = "Out of memory",
    [MYLLM_ERROR_NOT_SUPPORTED] = "Operation not supported",
    [MYLLM_ERROR_NOT_IMPLEMENTED] = "Not implemented",
    [MYLLM_ERROR_OVERFLOW] = "Overflow",
    [MYLLM_ERROR_UNDERFLOW] = "Underflow",
    [MYLLM_ERROR_TIMEOUT] = "Timeout",

    /* 张量错误 */
    [MYLLM_ERROR_TENSOR_SHAPE] = "Invalid tensor shape",
    [MYLLM_ERROR_TENSOR_DTYPE] = "Invalid tensor dtype",
    [MYLLM_ERROR_TENSOR_DEVICE] = "Invalid tensor device",
    [MYLLM_ERROR_TENSOR_STRIDE] = "Invalid tensor stride",
    [MYLLM_ERROR_TENSOR_INDEX] = "Tensor index out of bounds",
    [MYLLM_ERROR_TENSOR_BROADCAST] = "Broadcast error",

    /* 缓存错误 */
    [MYLLM_ERROR_CACHE_OVERFLOW] = "Cache overflow",
    [MYLLM_ERROR_CACHE_EMPTY] = "Cache is empty",
    [MYLLM_ERROR_CACHE_INVALID] = "Invalid cache",

    /* 加载错误 */
    [MYLLM_ERROR_LOAD_FILE] = "Failed to load file",
    [MYLLM_ERROR_LOAD_FORMAT] = "Invalid file format",
    [MYLLM_ERROR_LOAD_CHECKSUM] = "Checksum mismatch",
    [MYLLM_ERROR_LOAD_VERSION] = "Unsupported version",

    /* 模型错误 */
    [MYLLM_ERROR_MODEL_INIT] = "Model initialization failed",
    [MYLLM_ERROR_MODEL_WEIGHT] = "Invalid model weight",
    [MYLLM_ERROR_MODEL_CONFIG] = "Invalid model configuration",
    [MYLLM_ERROR_MODEL_INFER] = "Inference error",

    /* 分词器错误 */
    [MYLLM_ERROR_TOKENIZER_INIT] = "Tokenizer initialization failed",
    [MYLLM_ERROR_TOKENIZER_ENCODE] = "Encoding failed",
    [MYLLM_ERROR_TOKENIZER_DECODE] = "Decoding failed",

    /* 量化错误 */
    [MYLLM_ERROR_QUANT_RANGE] = "Quantization range error",
    [MYLLM_ERROR_QUANT_PRECISION] = "Quantization precision loss",
    [MYLLM_ERROR_QUANT_FORMAT] = "Unsupported quantization format",
};

/* 线程局部错误上下文 (简化实现，使用全局变量) */
static ErrorContext g_error_context = {0};

const char* error_message(MyLLMError error) {
    if (error < 0 || error >= (int)(sizeof(error_messages) / sizeof(error_messages[0]))) {
        return "Unknown error";
    }
    return error_messages[error] ? error_messages[error] : "Unknown error";
}

const char* error_name(MyLLMError error) {
    if (error < 0 || error >= (int)(sizeof(error_names) / sizeof(error_names[0]))) {
        return "UNKNOWN";
    }
    return error_names[error] ? error_names[error] : "UNKNOWN";
}

void error_set_context(MyLLMError code, const char* file, int line,
                       const char* func, const char* fmt, ...) {
    g_error_context.code = code;
    g_error_context.file = file;
    g_error_context.line = line;
    g_error_context.func = func;

    if (fmt) {
        va_list args;
        va_start(args, fmt);
        vsnprintf(g_error_context.message, sizeof(g_error_context.message), fmt, args);
        va_end(args);
    } else {
        strncpy(g_error_context.message, error_message(code),
                sizeof(g_error_context.message) - 1);
    }
}

const ErrorContext* error_get_context(void) {
    return &g_error_context;
}

void error_clear_context(void) {
    memset(&g_error_context, 0, sizeof(g_error_context));
}
