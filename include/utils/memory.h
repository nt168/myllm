/**
 * @file memory.h
 * @brief 内存工具
 *
 * 提供内存分配跟踪和对齐分配功能
 */

#ifndef MYLLM_UTILS_MEMORY_H
#define MYLLM_UTILS_MEMORY_H

#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

/* ============================================================================
 * 内存统计
 * ============================================================================ */

/**
 * @brief 内存统计结构
 */
typedef struct {
    size_t current_bytes;       ///< 当前分配字节数
    size_t peak_bytes;          ///< 峰值分配字节数
    size_t total_allocs;        ///< 总分配次数
    size_t total_frees;         ///< 总释放次数
    size_t active_allocs;       ///< 活跃分配数
} MemoryStats;

/**
 * @brief 启用内存跟踪
 * @return 0 成功，-1 失败
 */
int memory_tracking_enable(void);

/**
 * @brief 禁用内存跟踪
 */
void memory_tracking_disable(void);

/**
 * @brief 检查内存跟踪是否启用
 * @return 1 启用，0 禁用
 */
int memory_tracking_is_enabled(void);

/**
 * @brief 获取内存统计
 * @param stats 统计输出
 */
void memory_get_stats(MemoryStats* stats);

/**
 * @brief 重置内存统计
 */
void memory_reset_stats(void);

/**
 * @brief 打印内存统计
 */
void memory_print_stats(void);

/* ============================================================================
 * 跟踪内存分配
 * ============================================================================ */

/**
 * @brief 跟踪分配内存
 * @param size 字节数
 * @param file 源文件
 * @param line 行号
 * @return 分配的内存指针
 */
void* memory_tracked_malloc(size_t size, const char* file, int line);

/**
 * @brief 跟踪释放内存
 * @param ptr 内存指针
 */
void memory_tracked_free(void* ptr);

/**
 * @brief 跟踪重新分配内存
 * @param ptr 原指针
 * @param size 新大小
 * @param file 源文件
 * @param line 行号
 * @return 新指针
 */
void* memory_tracked_realloc(void* ptr, size_t size, const char* file, int line);

/**
 * @brief 跟踪分配并清零内存
 * @param count 元素数量
 * @param size 每个元素大小
 * @param file 源文件
 * @param line 行号
 * @return 分配的内存指针
 */
void* memory_tracked_calloc(size_t count, size_t size, const char* file, int line);

/* 便捷宏 */
#define MYLLM_MALLOC(size) \
    memory_tracked_malloc(size, __FILE__, __LINE__)

#define MYLLM_FREE(ptr) \
    memory_tracked_free(ptr)

#define MYLLM_REALLOC(ptr, size) \
    memory_tracked_realloc(ptr, size, __FILE__, __LINE__)

#define MYLLM_CALLOC(count, size) \
    memory_tracked_calloc(count, size, __FILE__, __LINE__)

/* ============================================================================
 * 对齐分配
 * ============================================================================ */

/**
 * @brief 分配对齐内存
 * @param alignment 对齐字节数 (必须是 2 的幂)
 * @param size 字节数
 * @return 分配的内存指针，失败返回 NULL
 */
void* memory_aligned_alloc(size_t alignment, size_t size);

/**
 * @brief 释放对齐内存
 * @param ptr 内存指针
 */
void memory_aligned_free(void* ptr);

/**
 * @brief 检查指针对齐
 * @param ptr 指针
 * @param alignment 对齐字节数
 * @return 1 对齐，0 未对齐
 */
int memory_is_aligned(const void* ptr, size_t alignment);

/**
 * @brief 计算对齐后的大小
 * @param size 原始大小
 * @param alignment 对齐字节数
 * @return 对齐后大小
 */
size_t memory_align_size(size_t size, size_t alignment);

/* ============================================================================
 * 内存工具
 * ============================================================================ */

/**
 * @brief 安全内存复制
 * @param dest 目标地址
 * @param src 源地址
 * @param n 字节数
 * @return 0 成功，-1 失败
 */
int memory_safe_copy(void* dest, const void* src, size_t n);

/**
 * @brief 安全内存设置
 * @param dest 目标地址
 * @param c 填充值
 * @param n 字节数
 * @return 0 成功，-1 失败
 */
int memory_safe_set(void* dest, int c, size_t n);

/**
 * @brief 比较内存块
 * @param ptr1 内存块1
 * @param ptr2 内存块2
 * @param n 字节数
 * @return 0 相等，非零不相等
 */
int memory_compare(const void* ptr1, const void* ptr2, size_t n);

/**
 * @brief 格式化字节大小
 * @param bytes 字节数
 * @param buf 输出缓冲区
 * @param buf_size 缓冲区大小
 * @return 格式化字符串长度
 */
int memory_format_size(size_t bytes, char* buf, size_t buf_size);

#ifdef __cplusplus
}
#endif

#endif /* MYLLM_UTILS_MEMORY_H */
