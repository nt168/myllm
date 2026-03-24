/**
 * @file dtype.h
 * @brief 数据类型定义 - 对应 phyllm/src/tensor/dtype.rs
 */

#ifndef MYLLM_DTYPE_H
#define MYLLM_DTYPE_H

#include <stdint.h>
#include <stddef.h>
#include <stdbool.h>

#ifdef __cplusplus
extern "C" {
#endif

/**
 * @brief 支持的数据类型
 *
 * 注意：这些值与 model_types.h 中的 DType 定义一致
 * 如果 model_types.h 已经定义了 DType，则使用那个定义
 */
#ifndef DTYPE_DEFINED
#define DTYPE_DEFINED

typedef enum {
    DTYPE_F32 = 0,      /**< 32-bit 浮点数 (默认) */
    DTYPE_F16 = 1,      /**< 16-bit 浮点数 */
    DTYPE_BF16 = 2,     /**< Brain Float 16-bit */
    DTYPE_I32 = 3,      /**< 32-bit 整数 */
    DTYPE_I64 = 4,      /**< 64-bit 整数 */
    DTYPE_COUNT         /**< 类型数量 */
} DType;

#endif /* DTYPE_DEFINED */

/**
 * @brief 获取数据类型的字节大小
 */
static inline size_t dtype_size(DType dtype) {
    switch (dtype) {
        case DTYPE_F16:  return 2;
        case DTYPE_F32:  return 4;
        case DTYPE_BF16: return 2;
        case DTYPE_I32:  return 4;
        case DTYPE_I64:  return 8;
        default:         return 4;
    }
}

/**
 * @brief 获取数据类型的位数
 */
static inline size_t dtype_size_in_bits(DType dtype) {
    return dtype_size(dtype) * 8;
}

/**
 * @brief 检查是否为浮点类型
 */
static inline bool dtype_is_float(DType dtype) {
    return dtype == DTYPE_F16 || dtype == DTYPE_F32 || dtype == DTYPE_BF16;
}

/**
 * @brief 检查是否为整数类型
 */
static inline bool dtype_is_integer(DType dtype) {
    return dtype == DTYPE_I32 || dtype == DTYPE_I64;
}

/**
 * @brief 获取数据类型名称字符串
 */
static inline const char* dtype_name(DType dtype) {
    switch (dtype) {
        case DTYPE_F16:  return "f16";
        case DTYPE_F32:  return "f32";
        case DTYPE_BF16: return "bf16";
        case DTYPE_I32:  return "i32";
        case DTYPE_I64:  return "i64";
        default:         return "unknown";
    }
}

/**
 * @brief 默认数据类型
 */
#define DTYPE_DEFAULT DTYPE_F32

#ifdef __cplusplus
}
#endif

#endif /* MYLLM_DTYPE_H */
