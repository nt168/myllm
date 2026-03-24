/**
 * @file shape.h
 * @brief 张量形状定义 - 对应 phyllm/src/tensor/shape.rs
 */

#ifndef MYLLM_SHAPE_H
#define MYLLM_SHAPE_H

#include <stddef.h>
#include <stdbool.h>
#include <stdint.h>
#include <sys/types.h>  /* for ssize_t */

#ifdef __cplusplus
extern "C" {
#endif

/** 最大维度数 */
#define MYLLM_MAX_NDIM 8

/**
 * @brief 张量形状结构
 */
typedef struct {
    size_t dims[MYLLM_MAX_NDIM];    /**< 维度大小数组 */
    size_t ndim;                     /**< 维度数量 */
} Shape;

/**
 * @brief 创建标量形状 (0维)
 */
Shape shape_scalar(void);

/**
 * @brief 从数组创建形状
 */
Shape shape_new(const size_t* dims, size_t ndim);

/**
 * @brief 从可变参数创建形状
 */
Shape shape_from_dims(size_t ndim, ...);

/**
 * @brief 获取维度数
 */
static inline size_t shape_ndim(const Shape* s) {
    return s->ndim;
}

/**
 * @brief 获取指定维度大小
 */
static inline size_t shape_dim(const Shape* s, size_t axis) {
    return (axis < s->ndim) ? s->dims[axis] : 0;
}

/**
 * @brief 获取维度数组
 */
static inline const size_t* shape_dims(const Shape* s) {
    return s->dims;
}

/**
 * @brief 获取元素总数
 */
size_t shape_numel(const Shape* s);

/**
 * @brief 检查是否为空张量
 */
bool shape_is_empty(const Shape* s);

/**
 * @brief 检查是否为标量
 */
static inline bool shape_is_scalar(const Shape* s) {
    return s->ndim == 0;
}

/**
 * @brief 计算连续存储的步幅 (C-contiguous, row-major)
 */
void shape_strides(const Shape* s, size_t* strides);

/**
 * @brief 计算多维索引的线性偏移
 */
size_t shape_offset(const Shape* s, const size_t* indices);

/**
 * @brief 广播两个形状
 * @return 成功返回0，失败返回-1
 */
int shape_broadcast(const Shape* a, const Shape* b, Shape* result);

/**
 * @brief 重塑形状
 * @param new_dims 新维度数组，-1 表示自动推断
 * @return 成功返回0，失败返回-1
 */
int shape_reshape(const Shape* s, const ssize_t* new_dims, size_t new_ndim, Shape* result);

/**
 * @brief 压缩形状 (移除大小为1的维度)
 */
Shape shape_squeeze(const Shape* s, int dim);

/**
 * @brief 扩展形状 (在指定位置添加大小为1的维度)
 */
Shape shape_unsqueeze(const Shape* s, size_t dim);

/**
 * @brief 转置两个维度
 */
Shape shape_transpose(const Shape* s, size_t dim1, size_t dim2);

/**
 * @brief 复制形状
 */
Shape shape_clone(const Shape* s);

/**
 * @brief 比较两个形状是否相等
 */
bool shape_equals(const Shape* a, const Shape* b);

/**
 * @brief 打印形状
 */
void shape_print(const Shape* s);

/**
 * @brief 广播两个形状
 */
int shape_broadcast(const Shape* a, const Shape* b, Shape* result);

/**
 * @brief 计算多维索引的线性偏移
 */
size_t shape_offset(const Shape* s, const size_t* indices);

#ifdef __cplusplus
}
#endif

#endif /* MYLLM_SHAPE_H */
