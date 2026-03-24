/**
 * @file safetensors.h
 * @brief SafeTensors 格式加载器
 *
 * SafeTensors 是一种安全、高效的张量序列化格式。
 * 格式布局:
 * - 头部: 8字节 (小端 uint64) 表示 JSON 元数据长度
 * - 元数据: JSON 格式的张量描述
 * - 数据: 连续的张量数据
 */

#ifndef MYLLM_SAFETENSORS_H
#define MYLLM_SAFETENSORS_H

#include "tensor/tensor.h"
#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

/* ============================================================================
 * SafeTensors 数据类型
 * ============================================================================ */

/**
 * @brief SafeTensors 数据类型枚举
 */
typedef enum {
    ST_F32,     /**< 32位浮点 */
    ST_F16,     /**< 16位浮点 (FP16) */
    ST_BF16,    /**< 16位脑浮点 (BF16) */
    ST_I32,     /**< 32位整数 */
    ST_I64,     /**< 64位整数 */
    ST_I16,     /**< 16位整数 */
    ST_I8,      /**< 8位整数 */
    ST_U8,      /**< 无符号8位整数 */
    ST_U32,     /**< 无符号32位整数 */
    ST_U64,     /**< 无符号64位整数 */
    ST_F8_E4M3, /**< 8位浮点 E4M3 格式 */
    ST_F8_E5M2, /**< 8位浮点 E5M2 格式 */
    ST_UNKNOWN  /**< 未知类型 */
} SafeTensorsDType;

/* ============================================================================
 * 张量信息结构
 * ============================================================================ */

/**
 * @brief 张量元数据
 */
typedef struct {
    char name[256];             /**< 张量名称 */
    size_t dims[8];             /**< 维度数组 */
    size_t ndim;                /**< 维度数量 */
    SafeTensorsDType dtype;     /**< 数据类型 */
    size_t offset;              /**< 数据偏移 (字节) */
    size_t num_bytes;           /**< 数据字节数 */
} TensorInfo;

/* ============================================================================
 * SafeTensors 加载器
 * ============================================================================ */

/**
 * @brief SafeTensors 加载器结构
 */
typedef struct SafeTensorsLoader {
    uint8_t* data;              /**< 文件数据 (mmap 或读取) */
    size_t data_size;           /**< 数据大小 */
    TensorInfo* tensors;        /**< 张量信息数组 */
    size_t num_tensors;         /**< 张量数量 */
    char* json_metadata;        /**< JSON 元数据 */
    size_t json_size;           /**< JSON 大小 */
    bool owns_data;             /**< 是否拥有数据 */
} SafeTensorsLoader;

/* ============================================================================
 * 生命周期管理
 * ============================================================================ */

/**
 * @brief 从文件创建 SafeTensors 加载器
 *
 * @param path 文件路径
 * @return 成功返回加载器指针，失败返回 NULL
 */
SafeTensorsLoader* safetensors_new(const char* path);

/**
 * @brief 释放 SafeTensors 加载器
 *
 * @param loader 加载器指针
 */
void safetensors_free(SafeTensorsLoader* loader);

/**
 * @brief 从内存创建 SafeTensors 加载器
 *
 * @param data 数据指针
 * @param size 数据大小
 * @return 成功返回加载器指针，失败返回 NULL
 */
SafeTensorsLoader* safetensors_from_memory(const void* data, size_t size);

/* ============================================================================
 * 状态查询
 * ============================================================================ */

/**
 * @brief 获取张量数量
 *
 * @param loader 加载器指针
 * @return 张量数量
 */
size_t safetensors_num_tensors(const SafeTensorsLoader* loader);

/**
 * @brief 获取所有张量名称
 *
 * @param loader 加载器指针
 * @param names 输出名称数组 (需预先分配)
 * @param max_names 最大名称数
 * @return 实际名称数
 */
size_t safetensors_get_names(const SafeTensorsLoader* loader, char** names, size_t max_names);

/**
 * @brief 检查张量是否存在
 *
 * @param loader 加载器指针
 * @param name 张量名称
 * @return 存在返回 true
 */
bool safetensors_has_tensor(const SafeTensorsLoader* loader, const char* name);

/**
 * @brief 获取张量信息
 *
 * @param loader 加载器指针
 * @param name 张量名称
 * @param info 输出张量信息
 * @return 成功返回 true
 */
bool safetensors_get_info(const SafeTensorsLoader* loader, const char* name, TensorInfo* info);

/* ============================================================================
 * 张量加载
 * ============================================================================ */

/**
 * @brief 加载张量到新的 Tensor 结构
 *
 * @param loader 加载器指针
 * @param name 张量名称
 * @return 成功返回 Tensor 指针，失败返回 NULL
 * @note 调用者负责释放返回的 Tensor
 */
Tensor* safetensors_load_tensor(const SafeTensorsLoader* loader, const char* name);

/**
 * @brief 加载张量并转换为 F32
 *
 * @param loader 加载器指针
 * @param name 张量名称
 * @return 成功返回 F32 Tensor 指针，失败返回 NULL
 */
Tensor* safetensors_load_tensor_f32(const SafeTensorsLoader* loader, const char* name);

/**
 * @brief 获取张量的原始数据指针 (零拷贝)
 *
 * @param loader 加载器指针
 * @param name 张量名称
 * @param info 输出张量信息 (可选)
 * @return 数据指针，失败返回 NULL
 */
const void* safetensors_get_raw_data(
    const SafeTensorsLoader* loader,
    const char* name,
    TensorInfo* info
);

/* ============================================================================
 * 辅助函数
 * ============================================================================ */

/**
 * @brief SafeTensors DType 转换为模型 DType
 *
 * @param st_dtype SafeTensors 数据类型
 * @return 模型 DType
 */
DType safetensors_dtype_to_model(SafeTensorsDType st_dtype);

/**
 * @brief 获取 SafeTensors DType 的大小 (字节)
 *
 * @param dtype SafeTensors 数据类型
 * @return 字节大小
 */
size_t safetensors_dtype_size(SafeTensorsDType dtype);

/**
 * @brief 从字符串解析 SafeTensors DType
 *
 * @param str 类型字符串
 * @return SafeTensors DType
 */
SafeTensorsDType safetensors_dtype_from_string(const char* str);

#ifdef __cplusplus
}
#endif

#endif /* MYLLM_SAFETENSORS_H */
