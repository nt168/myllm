/**
 * @file kv.h
 * @brief KV 缓存模块 - 用于高效的 Transformer 推理
 *
 * KV 缓存存储 Transformer 层的 Key 和 Value 张量，
 * 实现高效的自回归生成而无需重新计算。
 *
 * 内存布局: [batch, num_heads, max_seq_len, head_dim]
 */

#ifndef MYLLM_KV_H
#define MYLLM_KV_H

#include "tensor/tensor.h"
#include <stdbool.h>
#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

/* ============================================================================
 * KV 缓存结构
 * ============================================================================ */

/**
 * @brief KV 缓存结构体
 *
 * 存储多个 token 的 Key 和 Value 用于注意力计算。
 * 支持增量更新和批量操作。
 */
typedef struct KVCache {
    float* k_cache;         /**< Key 缓存 [batch, heads, max_seq, head_dim] */
    float* v_cache;         /**< Value 缓存 [batch, heads, max_seq, head_dim] */
    size_t max_seq_len;     /**< 最大序列长度 */
    size_t num_heads;       /**< 注意力头数 */
    size_t head_dim;        /**< 每个头的维度 */
    size_t batch_size;      /**< 批大小 */
    size_t current_len;     /**< 当前缓存长度 (token 数) */
    DType dtype;            /**< 数据类型 */
} KVCache;

/* ============================================================================
 * 生命周期管理
 * ============================================================================ */

/**
 * @brief 创建新的 KV 缓存
 *
 * @param max_seq_len 最大序列长度
 * @param num_heads 注意力头数
 * @param head_dim 每个头的维度
 * @param batch_size 批大小
 * @param dtype 数据类型
 * @return 成功返回 KVCache 指针，失败返回 NULL
 */
KVCache* kv_cache_new(
    size_t max_seq_len,
    size_t num_heads,
    size_t head_dim,
    size_t batch_size,
    DType dtype
);

/**
 * @brief 释放 KV 缓存
 *
 * @param cache KV 缓存指针
 */
void kv_cache_free(KVCache* cache);

/**
 * @brief 克隆 KV 缓存
 *
 * @param cache 源缓存
 * @return 成功返回新的 KVCache 指针，失败返回 NULL
 */
KVCache* kv_cache_clone(const KVCache* cache);

/* ============================================================================
 * 状态查询
 * ============================================================================ */

/**
 * @brief 检查缓存是否为空
 *
 * @param cache KV 缓存指针
 * @return 为空返回 true，否则返回 false
 */
bool kv_cache_is_empty(const KVCache* cache);

/**
 * @brief 获取当前缓存长度
 *
 * @param cache KV 缓存指针
 * @return 当前存储的 token 数
 */
size_t kv_cache_len(const KVCache* cache);

/**
 * @brief 获取缓存容量
 *
 * @param cache KV 缓存指针
 * @return 最大可存储的 token 数
 */
size_t kv_cache_capacity(const KVCache* cache);

/**
 * @brief 获取剩余可用空间
 *
 * @param cache KV 缓存指针
 * @return 还可以存储的 token 数
 */
size_t kv_cache_available(const KVCache* cache);

/* ============================================================================
 * 数据操作
 * ============================================================================ */

/**
 * @brief 追加单个 token 的 K, V 到缓存
 *
 * 输入张量形状应为 [batch, heads, 1, head_dim] 或 [heads, 1, head_dim]
 *
 * @param cache KV 缓存指针
 * @param k Key 张量
 * @param v Value 张量
 * @return 成功返回 MYLLM_OK，失败返回错误码
 */
int kv_cache_append(KVCache* cache, const Tensor* k, const Tensor* v);

/**
 * @brief 批量追加多个 token 的 K, V 到缓存
 *
 * 输入张量形状应为 [batch, heads, seq_len, head_dim]
 *
 * @param cache KV 缓存指针
 * @param k Key 张量
 * @param v Value 张量
 * @return 成功返回 MYLLM_OK，失败返回错误码
 */
int kv_cache_append_batch(KVCache* cache, const Tensor* k, const Tensor* v);

/**
 * @brief 获取缓存的 K, V 张量（到当前位置）
 *
 * 返回的张量形状为 [batch, heads, current_len, head_dim]
 * 注意：调用者负责释放返回的张量
 *
 * @param cache KV 缓存指针
 * @param k_out 输出 Key 张量的指针
 * @param v_out 输出 Value 张量的指针
 * @return 成功返回 MYLLM_OK，失败返回错误码
 */
int kv_cache_get(const KVCache* cache, Tensor** k_out, Tensor** v_out);

/**
 * @brief 获取指定范围的 K, V 张量
 *
 * 返回的张量形状为 [batch, heads, len, head_dim]
 *
 * @param cache KV 缓存指针
 * @param start 起始位置
 * @param len 长度
 * @param k_out 输出 Key 张量的指针
 * @param v_out 输出 Value 张量的指针
 * @return 成功返回 MYLLM_OK，失败返回错误码
 */
int kv_cache_get_slice(
    const KVCache* cache,
    size_t start,
    size_t len,
    Tensor** k_out,
    Tensor** v_out
);

/**
 * @brief 获取最后一个 token 的 K, V
 *
 * 返回的张量形状为 [batch, heads, 1, head_dim]
 *
 * @param cache KV 缓存指针
 * @param k_out 输出 Key 张量的指针
 * @param v_out 输出 Value 张量的指针
 * @return 成功返回 MYLLM_OK，失败返回错误码
 */
int kv_cache_get_last(const KVCache* cache, Tensor** k_out, Tensor** v_out);

/**
 * @brief 重置缓存（清空所有数据）
 *
 * @param cache KV 缓存指针
 */
void kv_cache_reset(KVCache* cache);

/* ============================================================================
 * 高级操作
 * ============================================================================ */

/**
 * @brief 删除指定位置之前的所有 token（滚动缓存）
 *
 * 保留从 keep_start 到当前位置的数据
 *
 * @param cache KV 缓存指针
 * @param keep_start 保留的起始位置
 * @return 成功返回 MYLLM_OK，失败返回错误码
 */
int kv_cache_discard_prefix(KVCache* cache, size_t keep_start);

/**
 * @brief 获取缓存内存使用量（字节）
 *
 * @param cache KV 缓存指针
 * @return 内存使用字节数
 */
size_t kv_cache_memory_usage(const KVCache* cache);

/* ============================================================================
 * 错误码
 * ============================================================================ */

#define MYLLM_OK                    0   /**< 成功 */
#define MYLLM_ERROR_NULL_POINTER    -1  /**< 空指针 */
#define MYLLM_ERROR_CACHE_OVERFLOW  -2  /**< 缓存溢出 */
#define MYLLM_ERROR_INVALID_SHAPE   -3  /**< 无效形状 */
#define MYLLM_ERROR_OUT_OF_MEMORY   -4  /**< 内存不足 */
#define MYLLM_ERROR_INVALID_RANGE   -5  /**< 无效范围 */

#ifdef __cplusplus
}
#endif

#endif /* MYLLM_KV_H */
