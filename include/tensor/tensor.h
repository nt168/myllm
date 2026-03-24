/**
 * @file tensor.h
 * @brief 张量核心定义 - 对应 phyllm/src/tensor/core.rs
 */

#ifndef MYLLM_TENSOR_H
#define MYLLM_TENSOR_H

#include "dtype.h"
#include "shape.h"
#include <stddef.h>
#include <stdint.h>
#include <stdbool.h>
#include <sys/types.h>  /* for ssize_t */

#ifdef __cplusplus
extern "C" {
#endif

/**
 * @brief 设备类型
 */
typedef enum {
    DEVICE_CPU,     /**< CPU 设备 */
    DEVICE_CUDA,    /**< CUDA GPU */
} DeviceType;

/**
 * @brief 设备结构
 */
typedef struct {
    DeviceType type;
    int id;         /**< 设备 ID (如 GPU 编号) */
} Device;

/**
 * @brief 张量结构
 *
 * 支持非连续视图 (strides 可能与 shape 不同)
 */
typedef struct Tensor {
    void* data;             /**< 数据指针 */
    Shape shape;            /**< 形状 */
    size_t strides[MYLLM_MAX_NDIM];  /**< 步幅 (元素数) */
    DType dtype;            /**< 数据类型 */
    Device device;          /**< 设备 */
    size_t offset;          /**< 数据偏移 (字节) */
    bool owns_data;         /**< 是否拥有数据 (用于释放) */
} Tensor;

/* ============================================================================
 * 创建与释放
 * ============================================================================ */

/**
 * @brief 创建空张量
 */
Tensor* tensor_new(const Shape* shape, DType dtype);

/**
 * @brief 创建零初始化张量
 */
Tensor* tensor_zeros(const size_t* dims, size_t ndim, DType dtype);

/**
 * @brief 创建全1张量
 */
Tensor* tensor_ones(const size_t* dims, size_t ndim, DType dtype);

/**
 * @brief 从数据数组创建张量
 * @param data 数据指针
 * @param dims 维度数组
 * @param ndim 维度数
 * @param dtype 数据类型
 * @param copy 是否复制数据
 */
Tensor* tensor_from_data(const void* data, const size_t* dims, size_t ndim,
                          DType dtype, bool copy);

/**
 * @brief 从 f32 数组创建张量 (自动类型转换)
 */
Tensor* tensor_from_f32(const float* data, const size_t* dims, size_t ndim, DType dtype);

/**
 * @brief 从 i32 数组创建张量
 */
Tensor* tensor_from_i32(const int32_t* data, const size_t* dims, size_t ndim, DType dtype);

/**
 * @brief 释放张量
 */
void tensor_free(Tensor* tensor);

/**
 * @brief 创建张量的深拷贝
 */
Tensor* tensor_clone(const Tensor* tensor);

/**
 * @brief 创建张量的视图 (共享数据)
 */
Tensor* tensor_view(const Tensor* tensor);

/* ============================================================================
 * 属性访问
 * ============================================================================ */

/**
 * @brief 获取形状
 */
static inline const Shape* tensor_shape(const Tensor* t) {
    return &t->shape;
}

/**
 * @brief 获取维度数
 */
static inline size_t tensor_ndim(const Tensor* t) {
    return t->shape.ndim;
}

/**
 * @brief 获取元素总数
 */
static inline size_t tensor_numel(const Tensor* t) {
    return shape_numel(&t->shape);
}

/**
 * @brief 获取数据类型
 */
static inline DType tensor_dtype(const Tensor* t) {
    return t->dtype;
}

/**
 * @brief 获取设备
 */
static inline Device tensor_device(const Tensor* t) {
    return t->device;
}

/**
 * @brief 获取数据指针
 */
void* tensor_data_ptr(Tensor* t);
const void* tensor_data_ptr_const(const Tensor* t);

/**
 * @brief 检查是否为连续存储
 */
bool tensor_is_contiguous(const Tensor* t);

/* ============================================================================
 * 数据访问
 * ============================================================================ */

/**
 * @brief 获取指定索引的值 (返回 f32)
 */
float tensor_get_f32(const Tensor* t, size_t index);

/**
 * @brief 设置指定索引的值 (f32)
 */
void tensor_set_f32(Tensor* t, size_t index, float value);

/**
 * @brief 获取指定多维索引的值
 */
float tensor_get_at(const Tensor* t, const size_t* indices);

/**
 * @brief 设置指定多维索引的值
 */
void tensor_set_at(Tensor* t, const size_t* indices, float value);

/**
 * @brief 复制数据到 f32 数组
 */
int tensor_to_f32(const Tensor* t, float* out, size_t out_size);

/**
 * @brief 从 f32 数组设置数据
 */
int tensor_set_data_f32(Tensor* t, const float* data, size_t data_size);

/* ============================================================================
 * 形状操作
 * ============================================================================ */

/**
 * @brief 重塑张量
 * @param new_dims 新维度，-1 表示自动推断
 */
Tensor* tensor_reshape(const Tensor* t, const ssize_t* new_dims, size_t new_ndim);

/**
 * @brief 转置张量
 */
Tensor* tensor_transpose(const Tensor* t, size_t dim1, size_t dim2);

/**
 * @brief 置换维度
 */
Tensor* tensor_permute(const Tensor* t, const size_t* dims, size_t ndim);

/**
 * @brief 压缩维度
 */
Tensor* tensor_squeeze(const Tensor* t, int dim);

/**
 * @brief 扩展维度
 */
Tensor* tensor_unsqueeze(const Tensor* t, size_t dim);

/**
 * @brief 获取连续存储副本
 */
Tensor* tensor_contiguous(const Tensor* t);

/**
 * @brief 切片
 * @param ranges 切片范围 [(start, end), ...]
 */
Tensor* tensor_slice(const Tensor* t, const size_t* starts, const size_t* ends, size_t n_ranges);

/**
 * @brief 索引
 */
Tensor* tensor_index(const Tensor* t, const size_t* indices, size_t n_indices);

/* ============================================================================
 * 工具函数
 * ============================================================================ */

/**
 * @brief 打印张量信息
 */
void tensor_print_info(const Tensor* t);

/**
 * @brief 打印张量数据
 */
void tensor_print(const Tensor* t, size_t max_elements);

/**
 * @brief 比较两个张量是否相等
 */
bool tensor_equals(const Tensor* a, const Tensor* b, float tolerance);

/* ============================================================================
 * 设备相关
 * ============================================================================ */

/**
 * @brief 创建 CPU 设备
 */
static inline Device device_cpu(void) {
    Device d = { DEVICE_CPU, 0 };
    return d;
}

/**
 * @brief 创建 CUDA 设备
 */
static inline Device device_cuda(int id) {
    Device d = { DEVICE_CUDA, id };
    return d;
}

#ifdef __cplusplus
}
#endif

#endif /* MYLLM_TENSOR_H */
