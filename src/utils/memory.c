/**
 * @file memory.c
 * @brief 内存工具实现
 */

#include "utils/memory.h"
#include "utils/log.h"
#include <stdlib.h>
#include <string.h>
#include <stdio.h>
#include <stdint.h>

/* ============================================================================
 * 内存统计
 * ============================================================================ */

static struct {
    MemoryStats stats;
    int enabled;
    int initialized;
} g_memory = {0};

int memory_tracking_enable(void) {
    g_memory.enabled = 1;
    g_memory.initialized = 1;
    return 0;
}

void memory_tracking_disable(void) {
    g_memory.enabled = 0;
}

int memory_tracking_is_enabled(void) {
    return g_memory.enabled;
}

void memory_get_stats(MemoryStats* stats) {
    if (stats) {
        *stats = g_memory.stats;
    }
}

void memory_reset_stats(void) {
    memset(&g_memory.stats, 0, sizeof(g_memory.stats));
}

void memory_print_stats(void) {
    char current[32], peak[32];
    memory_format_size(g_memory.stats.current_bytes, current, sizeof(current));
    memory_format_size(g_memory.stats.peak_bytes, peak, sizeof(peak));

    printf("\n=== Memory Statistics ===\n");
    printf("Current: %s\n", current);
    printf("Peak:    %s\n", peak);
    printf("Allocs:  %zu\n", g_memory.stats.total_allocs);
    printf("Frees:   %zu\n", g_memory.stats.total_frees);
    printf("Active:  %zu\n", g_memory.stats.active_allocs);
    printf("=========================\n\n");
}

/* ============================================================================
 * 跟踪内存分配
 * ============================================================================ */

/* 分配头部，用于跟踪大小 */
typedef struct {
    size_t size;
    const char* file;
    int line;
} AllocHeader;

#define HEADER_SIZE (sizeof(AllocHeader))
#define ALIGN_UP(x, a) (((x) + (a) - 1) & ~((a) - 1))

void* memory_tracked_malloc(size_t size, const char* file, int line) {
    if (size == 0) {
        return NULL;
    }

    /* 分配额外空间存储头部 */
    size_t total_size = ALIGN_UP(HEADER_SIZE, 16) + size;
    void* ptr = malloc(total_size);

    if (!ptr) {
        LOG_ERROR("Memory allocation failed: %zu bytes at %s:%d", size, file, line);
        return NULL;
    }

    /* 存储头部信息 */
    AllocHeader* header = (AllocHeader*)ptr;
    header->size = size;
    header->file = file;
    header->line = line;

    /* 更新统计 */
    if (g_memory.enabled) {
        g_memory.stats.current_bytes += size;
        if (g_memory.stats.current_bytes > g_memory.stats.peak_bytes) {
            g_memory.stats.peak_bytes = g_memory.stats.current_bytes;
        }
        g_memory.stats.total_allocs++;
        g_memory.stats.active_allocs++;
    }

    /* 返回用户指针 */
    return (char*)ptr + ALIGN_UP(HEADER_SIZE, 16);
}

void memory_tracked_free(void* ptr) {
    if (!ptr) return;

    /* 获取头部 */
    void* real_ptr = (char*)ptr - ALIGN_UP(HEADER_SIZE, 16);
    AllocHeader* header = (AllocHeader*)real_ptr;

    /* 更新统计 */
    if (g_memory.enabled) {
        g_memory.stats.current_bytes -= header->size;
        g_memory.stats.total_frees++;
        g_memory.stats.active_allocs--;
    }

    free(real_ptr);
}

void* memory_tracked_realloc(void* ptr, size_t size, const char* file, int line) {
    if (!ptr) {
        return memory_tracked_malloc(size, file, line);
    }

    if (size == 0) {
        memory_tracked_free(ptr);
        return NULL;
    }

    /* 获取旧头部 */
    void* real_ptr = (char*)ptr - ALIGN_UP(HEADER_SIZE, 16);
    AllocHeader* old_header = (AllocHeader*)real_ptr;
    size_t old_size = old_header->size;

    /* 分配新内存 */
    size_t total_size = ALIGN_UP(HEADER_SIZE, 16) + size;
    void* new_ptr = realloc(real_ptr, total_size);

    if (!new_ptr) {
        LOG_ERROR("Memory reallocation failed: %zu bytes at %s:%d", size, file, line);
        return NULL;
    }

    /* 更新头部 */
    AllocHeader* new_header = (AllocHeader*)new_ptr;
    new_header->size = size;
    new_header->file = file;
    new_header->line = line;

    /* 更新统计 */
    if (g_memory.enabled) {
        if (size > old_size) {
            g_memory.stats.current_bytes += (size - old_size);
        } else {
            g_memory.stats.current_bytes -= (old_size - size);
        }
        if (g_memory.stats.current_bytes > g_memory.stats.peak_bytes) {
            g_memory.stats.peak_bytes = g_memory.stats.current_bytes;
        }
    }

    return (char*)new_ptr + ALIGN_UP(HEADER_SIZE, 16);
}

void* memory_tracked_calloc(size_t count, size_t size, const char* file, int line) {
    size_t total = count * size;
    void* ptr = memory_tracked_malloc(total, file, line);
    if (ptr) {
        memset(ptr, 0, total);
    }
    return ptr;
}

/* ============================================================================
 * 对齐分配
 * ============================================================================ */

void* memory_aligned_alloc(size_t alignment, size_t size) {
    if (alignment == 0 || (alignment & (alignment - 1)) != 0) {
        return NULL;  /* 对齐必须是 2 的幂 */
    }

#ifdef _WIN32
    return _aligned_malloc(size, alignment);
#else
    void* ptr = NULL;
    if (posix_memalign(&ptr, alignment, size) != 0) {
        return NULL;
    }
    return ptr;
#endif
}

void memory_aligned_free(void* ptr) {
    if (!ptr) return;

#ifdef _WIN32
    _aligned_free(ptr);
#else
    free(ptr);
#endif
}

int memory_is_aligned(const void* ptr, size_t alignment) {
    return ((uintptr_t)ptr & (alignment - 1)) == 0;
}

size_t memory_align_size(size_t size, size_t alignment) {
    return (size + alignment - 1) & ~(alignment - 1);
}

/* ============================================================================
 * 内存工具
 * ============================================================================ */

int memory_safe_copy(void* dest, const void* src, size_t n) {
    if (!dest || !src) return -1;
    if (n == 0) return 0;
    memcpy(dest, src, n);
    return 0;
}

int memory_safe_set(void* dest, int c, size_t n) {
    if (!dest) return -1;
    if (n == 0) return 0;
    memset(dest, c, n);
    return 0;
}

int memory_compare(const void* ptr1, const void* ptr2, size_t n) {
    return memcmp(ptr1, ptr2, n);
}

int memory_format_size(size_t bytes, char* buf, size_t buf_size) {
    const char* units[] = {"B", "KB", "MB", "GB", "TB"};
    int unit_idx = 0;
    double size = (double)bytes;

    while (size >= 1024.0 && unit_idx < 4) {
        size /= 1024.0;
        unit_idx++;
    }

    return snprintf(buf, buf_size, "%.2f %s", size, units[unit_idx]);
}
