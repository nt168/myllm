/**
 * @file error.h
 * @brief 错误处理工具
 *
 * 提供统一的错误码定义和错误消息处理
 */

#ifndef MYLLM_UTILS_ERROR_H
#define MYLLM_UTILS_ERROR_H

#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

/**
 * @brief 通用错误码
 */
typedef enum {
    /* 成功 */
    MYLLM_OK = 0,

    /* 通用错误 (1-99) */
    MYLLM_ERROR_UNKNOWN = 1,            ///< 未知错误
    MYLLM_ERROR_INVALID_ARG = 2,        ///< 无效参数
    MYLLM_ERROR_NULL_POINTER = 3,       ///< 空指针
    MYLLM_ERROR_OUT_OF_MEMORY = 4,      ///< 内存不足
    MYLLM_ERROR_NOT_SUPPORTED = 5,      ///< 不支持的操作
    MYLLM_ERROR_NOT_IMPLEMENTED = 6,    ///< 未实现
    MYLLM_ERROR_OVERFLOW = 7,           ///< 溢出
    MYLLM_ERROR_UNDERFLOW = 8,          ///< 下溢
    MYLLM_ERROR_TIMEOUT = 9,            ///< 超时

    /* 张量错误 (100-199) */
    MYLLM_ERROR_TENSOR_SHAPE = 100,     ///< 形状错误
    MYLLM_ERROR_TENSOR_DTYPE = 101,     ///< 数据类型错误
    MYLLM_ERROR_TENSOR_DEVICE = 102,    ///< 设备错误
    MYLLM_ERROR_TENSOR_STRIDE = 103,    ///< 步幅错误
    MYLLM_ERROR_TENSOR_INDEX = 104,     ///< 索引越界
    MYLLM_ERROR_TENSOR_BROADCAST = 105, ///< 广播错误

    /* 缓存错误 (200-299) */
    MYLLM_ERROR_CACHE_OVERFLOW = 200,   ///< 缓存溢出
    MYLLM_ERROR_CACHE_EMPTY = 201,      ///< 缓存为空
    MYLLM_ERROR_CACHE_INVALID = 202,    ///< 缓存无效

    /* 加载错误 (300-399) */
    MYLLM_ERROR_LOAD_FILE = 300,        ///< 文件加载失败
    MYLLM_ERROR_LOAD_FORMAT = 301,      ///< 格式错误
    MYLLM_ERROR_LOAD_CHECKSUM = 302,    ///< 校验和错误
    MYLLM_ERROR_LOAD_VERSION = 303,     ///< 版本不兼容

    /* 模型错误 (400-499) */
    MYLLM_ERROR_MODEL_INIT = 400,       ///< 模型初始化失败
    MYLLM_ERROR_MODEL_WEIGHT = 401,     ///< 权重错误
    MYLLM_ERROR_MODEL_CONFIG = 402,     ///< 配置错误
    MYLLM_ERROR_MODEL_INFER = 403,      ///< 推理错误

    /* 分词器错误 (500-599) */
    MYLLM_ERROR_TOKENIZER_INIT = 500,   ///< 分词器初始化失败
    MYLLM_ERROR_TOKENIZER_ENCODE = 501, ///< 编码失败
    MYLLM_ERROR_TOKENIZER_DECODE = 502, ///< 解码失败

    /* 量化错误 (600-699) */
    MYLLM_ERROR_QUANT_RANGE = 600,      ///< 量化范围错误
    MYLLM_ERROR_QUANT_PRECISION = 601,  ///< 精度损失
    MYLLM_ERROR_QUANT_FORMAT = 602,     ///< 格式不支持

} MyLLMError;

/* ============================================================================
 * 错误消息
 * ============================================================================ */

/**
 * @brief 获取错误消息
 * @param error 错误码
 * @return 错误消息字符串
 */
const char* error_message(MyLLMError error);

/**
 * @brief 获取错误名称
 * @param error 错误码
 * @return 错误名称字符串
 */
const char* error_name(MyLLMError error);

/* ============================================================================
 * 错误上下文
 * ============================================================================ */

/**
 * @brief 错误上下文结构 (用于详细错误信息)
 */
typedef struct {
    MyLLMError code;            ///< 错误码
    const char* file;           ///< 源文件
    int line;                   ///< 行号
    const char* func;           ///< 函数名
    char message[256];          ///< 详细消息
} ErrorContext;

/**
 * @brief 设置错误上下文
 * @param code 错误码
 * @param file 源文件
 * @param line 行号
 * @param func 函数名
 * @param fmt 格式化消息
 * @param ... 参数
 */
void error_set_context(MyLLMError code, const char* file, int line,
                       const char* func, const char* fmt, ...);

/**
 * @brief 获取最后错误上下文
 * @return 错误上下文指针 (线程局部)
 */
const ErrorContext* error_get_context(void);

/**
 * @brief 清除错误上下文
 */
void error_clear_context(void);

/* ============================================================================
 * 便捷宏
 * ============================================================================ */

#define ERROR_SET(code, fmt, ...) \
    error_set_context(code, __FILE__, __LINE__, __func__, fmt, ##__VA_ARGS__)

#define ERROR_RETURN(code, fmt, ...) \
    do { \
        ERROR_SET(code, fmt, ##__VA_ARGS__); \
        return code; \
    } while(0)

#define ERROR_RETURN_NULL(code, fmt, ...) \
    do { \
        ERROR_SET(code, fmt, ##__VA_ARGS__); \
        return NULL; \
    } while(0)

/* 检查宏 */
#define CHECK_NULL(ptr, msg) \
    do { \
        if ((ptr) == NULL) { \
            ERROR_RETURN(MYLLM_ERROR_NULL_POINTER, msg); \
        } \
    } while(0)

#define CHECK_NULL_RET_NULL(ptr, msg) \
    do { \
        if ((ptr) == NULL) { \
            ERROR_SET(MYLLM_ERROR_NULL_POINTER, msg); \
            return NULL; \
        } \
    } while(0)

#define CHECK_ARG(cond, msg) \
    do { \
        if (!(cond)) { \
            ERROR_RETURN(MYLLM_ERROR_INVALID_ARG, msg); \
        } \
    } while(0)

#define CHECK_ALLOC(ptr, msg) \
    do { \
        if ((ptr) == NULL) { \
            ERROR_RETURN(MYLLM_ERROR_OUT_OF_MEMORY, msg); \
        } \
    } while(0)

#ifdef __cplusplus
}
#endif

#endif /* MYLLM_UTILS_ERROR_H */
