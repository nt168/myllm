/**
 * @file kv.c
 * @brief KV 缓存模块实现
 */

#include "kv/kv.h"
#include <stdlib.h>
#include <string.h>
#include <stdio.h>

/* ============================================================================
 * 内部辅助函数
 * ============================================================================ */

/**
 * @brief 计算形状的元素数量
 */
static size_t compute_numel(const Shape* shape) {
    if (shape->ndim == 0) return 1;
    size_t numel = 1;
    for (size_t i = 0; i < shape->ndim; i++) {
        numel *= shape->dims[i];
    }
    return numel;
}

/**
 * @brief 创建简单的张量结构（用于输出）
 */
static Tensor* create_tensor_4d(size_t d0, size_t d1, size_t d2, size_t d3, DType dtype) {
    Tensor* t = (Tensor*)malloc(sizeof(Tensor));
    if (!t) return NULL;

    /* 使用 shape_new 创建形状 */
    size_t dims[4] = {d0, d1, d2, d3};
    t->shape = shape_new(dims, 4);

    /* 设置步幅 */
    t->strides[0] = d1 * d2 * d3;
    t->strides[1] = d2 * d3;
    t->strides[2] = d3;
    t->strides[3] = 1;

    t->dtype = dtype;
    t->offset = 0;
    t->device.type = DEVICE_CPU;
    t->device.id = 0;
    t->owns_data = true;

    size_t numel = d0 * d1 * d2 * d3;
    size_t data_size = numel * sizeof(float);  /* 目前只支持 F32 */
    t->data = calloc(1, data_size);
    if (!t->data) {
        free(t);
        return NULL;
    }

    return t;
}

/**
 * @brief 释放简单张量
 */
static void free_tensor(Tensor* t) {
    if (!t) return;
    if (t->owns_data && t->data) {
        free(t->data);
    }
    free(t);
}

/* ============================================================================
 * 生命周期管理
 * ============================================================================ */

KVCache* kv_cache_new(
    size_t max_seq_len,
    size_t num_heads,
    size_t head_dim,
    size_t batch_size,
    DType dtype
) {
    KVCache* cache = (KVCache*)malloc(sizeof(KVCache));
    if (!cache) {
        return NULL;
    }

    cache->max_seq_len = max_seq_len;
    cache->num_heads = num_heads;
    cache->head_dim = head_dim;
    cache->batch_size = batch_size;
    cache->current_len = 0;
    cache->dtype = dtype;

    /* 分配 K 和 V 缓存: [batch, heads, max_seq, head_dim] */
    size_t cache_size = batch_size * num_heads * max_seq_len * head_dim;
    cache->k_cache = (float*)calloc(cache_size, sizeof(float));
    cache->v_cache = (float*)calloc(cache_size, sizeof(float));

    if (!cache->k_cache || !cache->v_cache) {
        free(cache->k_cache);
        free(cache->v_cache);
        free(cache);
        return NULL;
    }

    return cache;
}

void kv_cache_free(KVCache* cache) {
    if (!cache) {
        return;
    }

    free(cache->k_cache);
    free(cache->v_cache);
    free(cache);
}

KVCache* kv_cache_clone(const KVCache* cache) {
    if (!cache) {
        return NULL;
    }

    KVCache* clone = kv_cache_new(
        cache->max_seq_len,
        cache->num_heads,
        cache->head_dim,
        cache->batch_size,
        cache->dtype
    );

    if (!clone) {
        return NULL;
    }

    /* 复制数据 */
    size_t data_size = cache->batch_size * cache->num_heads *
                       cache->max_seq_len * cache->head_dim * sizeof(float);
    memcpy(clone->k_cache, cache->k_cache, data_size);
    memcpy(clone->v_cache, cache->v_cache, data_size);
    clone->current_len = cache->current_len;

    return clone;
}

/* ============================================================================
 * 状态查询
 * ============================================================================ */

bool kv_cache_is_empty(const KVCache* cache) {
    return cache ? cache->current_len == 0 : true;
}

size_t kv_cache_len(const KVCache* cache) {
    return cache ? cache->current_len : 0;
}

size_t kv_cache_capacity(const KVCache* cache) {
    return cache ? cache->max_seq_len : 0;
}

size_t kv_cache_available(const KVCache* cache) {
    return cache ? (cache->max_seq_len - cache->current_len) : 0;
}

/* ============================================================================
 * 数据操作
 * ============================================================================ */

int kv_cache_append(KVCache* cache, const Tensor* k, const Tensor* v) {
    if (!cache || !k || !v) {
        return MYLLM_ERROR_NULL_POINTER;
    }

    if (cache->current_len >= cache->max_seq_len) {
        fprintf(stderr, "KVCache: overflow, current_len=%zu, max_seq_len=%zu\n",
                cache->current_len, cache->max_seq_len);
        return MYLLM_ERROR_CACHE_OVERFLOW;
    }

    /* 支持 3D 和 4D 输入张量 */
    bool is_3d = (k->shape.ndim == 3);
    bool is_4d = (k->shape.ndim == 4);

    if (!is_3d && !is_4d) {
        fprintf(stderr, "KVCache: expected 3D or 4D tensors, got K=%zuD\n", k->shape.ndim);
        return MYLLM_ERROR_INVALID_SHAPE;
    }

    size_t k_seq_len, k_total_dim;
    if (is_3d) {
        /* 3D: [batch, seq, num_heads * head_dim] 或 [seq, num_heads * head_dim] */
        if (k->shape.dims[k->shape.ndim - 1] != cache->num_heads * cache->head_dim) {
            fprintf(stderr, "KVCache: 3D tensor last dim mismatch, expected %zu, got %zu\n",
                    cache->num_heads * cache->head_dim, k->shape.dims[k->shape.ndim - 1]);
            return MYLLM_ERROR_INVALID_SHAPE;
        }
        k_seq_len = k->shape.dims[k->shape.ndim - 2];
        k_total_dim = k->shape.dims[k->shape.ndim - 1];
    } else {
        /* 4D: [batch, heads, seq, head_dim] */
        if (k->shape.dims[2] != 1 || v->shape.dims[2] != 1) {
            fprintf(stderr, "KVCache: expected seq_len=1 for 4D append, got K=%zu\n",
                    k->shape.dims[2]);
            return MYLLM_ERROR_INVALID_SHAPE;
        }
        k_seq_len = 1;
    }

    size_t pos = cache->current_len;
    size_t batch = cache->batch_size;
    size_t heads = cache->num_heads;
    size_t dim = cache->head_dim;

    const float* k_data = (const float*)k->data;
    const float* v_data = (const float*)v->data;

    if (!k_data || !v_data) {
        return MYLLM_ERROR_NULL_POINTER;
    }

    if (is_3d) {
        /* 3D 输入: [batch, seq, heads * dim] -> 缓存: [batch, heads, max_seq, dim] */
        for (size_t b = 0; b < batch; b++) {
            for (size_t h = 0; h < heads; h++) {
                size_t cache_idx = ((b * heads + h) * cache->max_seq_len + pos) * dim;
                /* 3D 源数据: [b, 0, h * dim ... h * dim + dim - 1] */
                size_t src_idx = b * k_total_dim + h * dim;

                memcpy(&cache->k_cache[cache_idx], &k_data[src_idx], dim * sizeof(float));
                memcpy(&cache->v_cache[cache_idx], &v_data[src_idx], dim * sizeof(float));
            }
        }
    } else {
        /* 4D 输入: [batch, heads, 1, dim] -> 缓存: [batch, heads, max_seq, dim] */
        for (size_t b = 0; b < batch; b++) {
            for (size_t h = 0; h < heads; h++) {
                size_t cache_idx = ((b * heads + h) * cache->max_seq_len + pos) * dim;
                size_t src_idx = ((b * heads + h) * 1) * dim;

                memcpy(&cache->k_cache[cache_idx], &k_data[src_idx], dim * sizeof(float));
                memcpy(&cache->v_cache[cache_idx], &v_data[src_idx], dim * sizeof(float));
            }
        }
    }

    cache->current_len++;
    return MYLLM_OK;
}

int kv_cache_append_batch(KVCache* cache, const Tensor* k, const Tensor* v) {
    if (!cache || !k || !v) {
        return MYLLM_ERROR_NULL_POINTER;
    }

    /* 支持 3D 输入张量: [batch, seq, heads * dim] */
    if (k->shape.ndim < 3) {
        fprintf(stderr, "KVCache: expected at least 3D tensor, got %zuD\n", k->shape.ndim);
        return MYLLM_ERROR_INVALID_SHAPE;
    }

    size_t seq_len = k->shape.dims[k->shape.ndim - 2];
    size_t total_dim = k->shape.dims[k->shape.ndim - 1];

    /* 验证维度 */
    if (total_dim != cache->num_heads * cache->head_dim) {
        fprintf(stderr, "KVCache: tensor dim mismatch, expected %zu, got %zu\n",
                cache->num_heads * cache->head_dim, total_dim);
        return MYLLM_ERROR_INVALID_SHAPE;
    }

    if (cache->current_len + seq_len > cache->max_seq_len) {
        fprintf(stderr, "KVCache: batch overflow, current_len=%zu, seq_len=%zu, max_seq_len=%zu\n",
                cache->current_len, seq_len, cache->max_seq_len);
        return MYLLM_ERROR_CACHE_OVERFLOW;
    }

    size_t pos = cache->current_len;
    size_t batch = cache->batch_size;
    size_t heads = cache->num_heads;
    size_t dim = cache->head_dim;

    const float* k_data = (const float*)k->data;
    const float* v_data = (const float*)v->data;

    if (!k_data || !v_data) {
        return MYLLM_ERROR_NULL_POINTER;
    }

    /* 批量复制: 3D [batch, seq, heads * dim] -> 缓存 [batch, heads, max_seq, dim] */
    for (size_t b = 0; b < batch; b++) {
        for (size_t h = 0; h < heads; h++) {
            for (size_t s = 0; s < seq_len; s++) {
                size_t cache_idx = ((b * heads + h) * cache->max_seq_len + pos + s) * dim;
                /* 3D 源: [b, s, h * dim ... h * dim + dim - 1] */
                size_t src_idx = (b * seq_len + s) * total_dim + h * dim;

                memcpy(&cache->k_cache[cache_idx], &k_data[src_idx], dim * sizeof(float));
                memcpy(&cache->v_cache[cache_idx], &v_data[src_idx], dim * sizeof(float));
            }
        }
    }

    cache->current_len += seq_len;
    return MYLLM_OK;
}

int kv_cache_get(const KVCache* cache, Tensor** k_out, Tensor** v_out) {
    if (!cache || !k_out || !v_out) {
        return MYLLM_ERROR_NULL_POINTER;
    }

    if (cache->current_len == 0) {
        *k_out = NULL;
        *v_out = NULL;
        return MYLLM_OK;
    }

    return kv_cache_get_slice(cache, 0, cache->current_len, k_out, v_out);
}

int kv_cache_get_slice(
    const KVCache* cache,
    size_t start,
    size_t len,
    Tensor** k_out,
    Tensor** v_out
) {
    if (!cache || !k_out || !v_out) {
        return MYLLM_ERROR_NULL_POINTER;
    }

    if (start + len > cache->current_len) {
        fprintf(stderr, "KVCache: invalid slice range, start=%zu, len=%zu, current_len=%zu\n",
                start, len, cache->current_len);
        return MYLLM_ERROR_INVALID_RANGE;
    }

    size_t batch = cache->batch_size;
    size_t heads = cache->num_heads;
    size_t dim = cache->head_dim;

    /* 创建输出张量: [batch, heads, len, head_dim] */
    *k_out = create_tensor_4d(batch, heads, len, dim, cache->dtype);
    *v_out = create_tensor_4d(batch, heads, len, dim, cache->dtype);

    if (!*k_out || !*v_out) {
        free_tensor(*k_out);
        free_tensor(*v_out);
        *k_out = NULL;
        *v_out = NULL;
        return MYLLM_ERROR_OUT_OF_MEMORY;
    }

    /* 复制数据 */
    float* k_out_data = (float*)(*k_out)->data;
    float* v_out_data = (float*)(*v_out)->data;

    for (size_t b = 0; b < batch; b++) {
        for (size_t h = 0; h < heads; h++) {
            for (size_t s = 0; s < len; s++) {
                size_t cache_idx = ((b * heads + h) * cache->max_seq_len + start + s) * dim;
                size_t out_idx = ((b * heads + h) * len + s) * dim;

                memcpy(&k_out_data[out_idx], &cache->k_cache[cache_idx], dim * sizeof(float));
                memcpy(&v_out_data[out_idx], &cache->v_cache[cache_idx], dim * sizeof(float));
            }
        }
    }

    return MYLLM_OK;
}

int kv_cache_get_last(const KVCache* cache, Tensor** k_out, Tensor** v_out) {
    if (!cache || !k_out || !v_out) {
        return MYLLM_ERROR_NULL_POINTER;
    }

    if (cache->current_len == 0) {
        fprintf(stderr, "KVCache: cannot get last from empty cache\n");
        return MYLLM_ERROR_INVALID_RANGE;
    }

    return kv_cache_get_slice(cache, cache->current_len - 1, 1, k_out, v_out);
}

void kv_cache_reset(KVCache* cache) {
    if (!cache) {
        return;
    }

    cache->current_len = 0;

    /* 可选：清零缓存数据 */
    size_t cache_size = cache->batch_size * cache->num_heads *
                        cache->max_seq_len * cache->head_dim;
    memset(cache->k_cache, 0, cache_size * sizeof(float));
    memset(cache->v_cache, 0, cache_size * sizeof(float));
}

/* ============================================================================
 * 高级操作
 * ============================================================================ */

int kv_cache_discard_prefix(KVCache* cache, size_t keep_start) {
    if (!cache) {
        return MYLLM_ERROR_NULL_POINTER;
    }

    if (keep_start >= cache->current_len) {
        return MYLLM_OK;
    }

    size_t new_len = cache->current_len - keep_start;
    size_t batch = cache->batch_size;
    size_t heads = cache->num_heads;
    size_t dim = cache->head_dim;

    for (size_t b = 0; b < batch; b++) {
        for (size_t h = 0; h < heads; h++) {
            for (size_t s = 0; s < new_len; s++) {
                size_t src_idx = ((b * heads + h) * cache->max_seq_len + keep_start + s) * dim;
                size_t dst_idx = ((b * heads + h) * cache->max_seq_len + s) * dim;

                memmove(&cache->k_cache[dst_idx], &cache->k_cache[src_idx], dim * sizeof(float));
                memmove(&cache->v_cache[dst_idx], &cache->v_cache[src_idx], dim * sizeof(float));
            }
        }
    }

    cache->current_len = new_len;
    return MYLLM_OK;
}

size_t kv_cache_memory_usage(const KVCache* cache) {
    if (!cache) {
        return 0;
    }

    size_t element_size = sizeof(float);
    size_t total_elements = cache->batch_size * cache->num_heads *
                            cache->max_seq_len * cache->head_dim * 2;

    return total_elements * element_size + sizeof(KVCache);
}
