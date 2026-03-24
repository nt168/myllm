/**
 * @file test_kv.c
 * @brief KV 缓存模块单元测试
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include "kv/kv.h"
#include "tensor/shape.h"
#include "common/test_common.h"

/* ============================================================================
 * 测试辅助函数
 * ============================================================================ */

/**
 * @brief 创建简单的 4D 张量用于测试
 */
static Tensor* create_test_tensor_4d(size_t d0, size_t d1, size_t d2, size_t d3) {
    Tensor* t = (Tensor*)malloc(sizeof(Tensor));
    if (!t) return NULL;
    memset(t, 0, sizeof(Tensor));

    /* 使用 shape_new 创建形状 */
    size_t dims[4] = {d0, d1, d2, d3};
    t->shape = shape_new(dims, 4);

    /* 设置步幅 */
    t->strides[0] = d1 * d2 * d3;
    t->strides[1] = d2 * d3;
    t->strides[2] = d3;
    t->strides[3] = 1;

    t->dtype = DTYPE_F32;
    t->offset = 0;
    t->device.type = DEVICE_CPU;
    t->device.id = 0;
    t->owns_data = true;

    size_t data_size = shape_numel(&t->shape) * sizeof(float);
    t->data = calloc(1, data_size);
    if (!t->data) {
        free(t);
        return NULL;
    }

    /* 填充测试数据 */
    float* data = (float*)t->data;
    size_t numel = shape_numel(&t->shape);
    for (size_t i = 0; i < numel; i++) {
        data[i] = (float)i * 0.1f;
    }

    return t;
}

/**
 * @brief 释放测试张量
 */
static void free_test_tensor(Tensor* t) {
    if (!t) return;
    if (t->owns_data && t->data) {
        free(t->data);
    }
    free(t);
}

/* ============================================================================
 * 生命周期测试
 * ============================================================================ */

static void test_kv_cache_creation(void) {
    KVCache* cache = kv_cache_new(128, 8, 64, 1, DTYPE_F32);
    ASSERT_NOT_NULL(cache, "Cache should not be NULL");
    ASSERT_EQ(kv_cache_capacity(cache), 128, "Capacity should be 128");
    ASSERT_EQ(kv_cache_len(cache), 0, "Length should be 0");
    ASSERT_TRUE(kv_cache_is_empty(cache), "Cache should be empty");

    kv_cache_free(cache);
    TEST_PASS();
}

static void test_kv_cache_free_null(void) {
    /* 应该不会崩溃 */
    kv_cache_free(NULL);
    TEST_PASS();
}

static void test_kv_cache_clone(void) {
    KVCache* cache = kv_cache_new(128, 4, 32, 1, DTYPE_F32);
    ASSERT_NOT_NULL(cache, "Cache should not be NULL");

    /* 添加一些数据 */
    Tensor* k = create_test_tensor_4d(1, 4, 1, 32);
    Tensor* v = create_test_tensor_4d(1, 4, 1, 32);

    kv_cache_append(cache, k, v);
    kv_cache_append(cache, k, v);

    /* 克隆 */
    KVCache* clone = kv_cache_clone(cache);
    ASSERT_NOT_NULL(clone, "Clone should not be NULL");
    ASSERT_EQ(kv_cache_len(clone), 2, "Clone length should be 2");

    free_test_tensor(k);
    free_test_tensor(v);
    kv_cache_free(cache);
    kv_cache_free(clone);
    TEST_PASS();
}

/* ============================================================================
 * 状态查询测试
 * ============================================================================ */

static void test_kv_cache_available(void) {
    KVCache* cache = kv_cache_new(100, 4, 32, 1, DTYPE_F32);
    ASSERT_NOT_NULL(cache, "Cache should not be NULL");
    ASSERT_EQ(kv_cache_available(cache), 100, "Available should be 100");

    /* 添加一个 token */
    Tensor* k = create_test_tensor_4d(1, 4, 1, 32);
    Tensor* v = create_test_tensor_4d(1, 4, 1, 32);

    kv_cache_append(cache, k, v);
    ASSERT_EQ(kv_cache_available(cache), 99, "Available should be 99");

    free_test_tensor(k);
    free_test_tensor(v);
    kv_cache_free(cache);
    TEST_PASS();
}

/* ============================================================================
 * 追加操作测试
 * ============================================================================ */

static void test_kv_cache_append_single(void) {
    KVCache* cache = kv_cache_new(128, 4, 32, 1, DTYPE_F32);
    ASSERT_NOT_NULL(cache, "Cache should not be NULL");

    /* 创建 K, V 张量 [1, 4, 1, 32] */
    Tensor* k = create_test_tensor_4d(1, 4, 1, 32);
    Tensor* v = create_test_tensor_4d(1, 4, 1, 32);
    ASSERT_NOT_NULL(k, "K tensor should not be NULL");
    ASSERT_NOT_NULL(v, "V tensor should not be NULL");

    int ret = kv_cache_append(cache, k, v);
    ASSERT_EQ(ret, MYLLM_OK, "Append should succeed");
    ASSERT_EQ(kv_cache_len(cache), 1, "Length should be 1");
    ASSERT_FALSE(kv_cache_is_empty(cache), "Cache should not be empty");

    free_test_tensor(k);
    free_test_tensor(v);
    kv_cache_free(cache);
    TEST_PASS();
}

static void test_kv_cache_append_multiple(void) {
    KVCache* cache = kv_cache_new(128, 4, 32, 1, DTYPE_F32);
    ASSERT_NOT_NULL(cache, "Cache should not be NULL");

    Tensor* k = create_test_tensor_4d(1, 4, 1, 32);
    Tensor* v = create_test_tensor_4d(1, 4, 1, 32);

    /* 添加 5 个 token */
    for (int i = 0; i < 5; i++) {
        int ret = kv_cache_append(cache, k, v);
        ASSERT_EQ(ret, MYLLM_OK, "Append should succeed");
    }
    ASSERT_EQ(kv_cache_len(cache), 5, "Length should be 5");

    free_test_tensor(k);
    free_test_tensor(v);
    kv_cache_free(cache);
    TEST_PASS();
}

static void test_kv_cache_append_batch(void) {
    KVCache* cache = kv_cache_new(128, 4, 32, 1, DTYPE_F32);
    ASSERT_NOT_NULL(cache, "Cache should not be NULL");

    /* 创建批量 K, V 张量 [1, 4, 10, 32] */
    Tensor* k = create_test_tensor_4d(1, 4, 10, 32);
    Tensor* v = create_test_tensor_4d(1, 4, 10, 32);
    ASSERT_NOT_NULL(k, "K tensor should not be NULL");
    ASSERT_NOT_NULL(v, "V tensor should not be NULL");

    int ret = kv_cache_append_batch(cache, k, v);
    ASSERT_EQ(ret, MYLLM_OK, "Batch append should succeed");
    ASSERT_EQ(kv_cache_len(cache), 10, "Length should be 10");

    free_test_tensor(k);
    free_test_tensor(v);
    kv_cache_free(cache);
    TEST_PASS();
}

static void test_kv_cache_overflow(void) {
    KVCache* cache = kv_cache_new(5, 4, 32, 1, DTYPE_F32);
    ASSERT_NOT_NULL(cache, "Cache should not be NULL");

    Tensor* k = create_test_tensor_4d(1, 4, 1, 32);
    Tensor* v = create_test_tensor_4d(1, 4, 1, 32);

    /* 填满缓存 */
    for (int i = 0; i < 5; i++) {
        kv_cache_append(cache, k, v);
    }
    ASSERT_EQ(kv_cache_len(cache), 5, "Length should be 5");

    /* 尝试溢出 */
    int ret = kv_cache_append(cache, k, v);
    ASSERT_EQ(ret, MYLLM_ERROR_CACHE_OVERFLOW, "Should return overflow error");

    free_test_tensor(k);
    free_test_tensor(v);
    kv_cache_free(cache);
    TEST_PASS();
}

/* ============================================================================
 * 获取操作测试
 * ============================================================================ */

static void test_kv_cache_get(void) {
    KVCache* cache = kv_cache_new(128, 4, 32, 1, DTYPE_F32);
    ASSERT_NOT_NULL(cache, "Cache should not be NULL");

    /* 添加数据 */
    Tensor* k = create_test_tensor_4d(1, 4, 1, 32);
    Tensor* v = create_test_tensor_4d(1, 4, 1, 32);
    kv_cache_append(cache, k, v);

    /* 获取数据 */
    Tensor *k_out = NULL, *v_out = NULL;
    int ret = kv_cache_get(cache, &k_out, &v_out);
    ASSERT_EQ(ret, MYLLM_OK, "Get should succeed");
    ASSERT_NOT_NULL(k_out, "K output should not be NULL");
    ASSERT_NOT_NULL(v_out, "V output should not be NULL");

    /* 验证形状 */
    ASSERT_EQ(k_out->shape.ndim, 4, "K output should be 4D");
    ASSERT_EQ(k_out->shape.dims[0], 1, "Batch should be 1");
    ASSERT_EQ(k_out->shape.dims[1], 4, "Heads should be 4");
    ASSERT_EQ(k_out->shape.dims[2], 1, "Seq len should be 1");
    ASSERT_EQ(k_out->shape.dims[3], 32, "Head dim should be 32");

    free_test_tensor(k);
    free_test_tensor(v);
    free_test_tensor(k_out);
    free_test_tensor(v_out);
    kv_cache_free(cache);
    TEST_PASS();
}

static void test_kv_cache_get_slice(void) {
    KVCache* cache = kv_cache_new(128, 4, 32, 1, DTYPE_F32);
    ASSERT_NOT_NULL(cache, "Cache should not be NULL");

    /* 添加 5 个 token */
    Tensor* k = create_test_tensor_4d(1, 4, 1, 32);
    Tensor* v = create_test_tensor_4d(1, 4, 1, 32);
    for (int i = 0; i < 5; i++) {
        kv_cache_append(cache, k, v);
    }

    /* 获取切片 [1:3] */
    Tensor *k_out = NULL, *v_out = NULL;
    int ret = kv_cache_get_slice(cache, 1, 2, &k_out, &v_out);
    ASSERT_EQ(ret, MYLLM_OK, "Get slice should succeed");
    ASSERT_NOT_NULL(k_out, "K output should not be NULL");

    ASSERT_EQ(k_out->shape.dims[2], 2, "Slice length should be 2");

    free_test_tensor(k);
    free_test_tensor(v);
    free_test_tensor(k_out);
    free_test_tensor(v_out);
    kv_cache_free(cache);
    TEST_PASS();
}

static void test_kv_cache_get_last(void) {
    KVCache* cache = kv_cache_new(128, 4, 32, 1, DTYPE_F32);
    ASSERT_NOT_NULL(cache, "Cache should not be NULL");

    /* 添加 3 个 token */
    Tensor* k = create_test_tensor_4d(1, 4, 1, 32);
    Tensor* v = create_test_tensor_4d(1, 4, 1, 32);
    for (int i = 0; i < 3; i++) {
        kv_cache_append(cache, k, v);
    }

    /* 获取最后一个 */
    Tensor *k_out = NULL, *v_out = NULL;
    int ret = kv_cache_get_last(cache, &k_out, &v_out);
    ASSERT_EQ(ret, MYLLM_OK, "Get last should succeed");
    ASSERT_NOT_NULL(k_out, "K output should not be NULL");

    ASSERT_EQ(k_out->shape.dims[2], 1, "Last should have length 1");

    free_test_tensor(k);
    free_test_tensor(v);
    free_test_tensor(k_out);
    free_test_tensor(v_out);
    kv_cache_free(cache);
    TEST_PASS();
}

/* ============================================================================
 * 重置操作测试
 * ============================================================================ */

static void test_kv_cache_reset(void) {
    KVCache* cache = kv_cache_new(128, 4, 32, 1, DTYPE_F32);
    ASSERT_NOT_NULL(cache, "Cache should not be NULL");

    /* 添加数据 */
    Tensor* k = create_test_tensor_4d(1, 4, 1, 32);
    Tensor* v = create_test_tensor_4d(1, 4, 1, 32);
    for (int i = 0; i < 5; i++) {
        kv_cache_append(cache, k, v);
    }
    ASSERT_EQ(kv_cache_len(cache), 5, "Length should be 5");

    /* 重置 */
    kv_cache_reset(cache);
    ASSERT_EQ(kv_cache_len(cache), 0, "Length should be 0 after reset");
    ASSERT_TRUE(kv_cache_is_empty(cache), "Cache should be empty after reset");

    /* 可以重新添加 */
    kv_cache_append(cache, k, v);
    ASSERT_EQ(kv_cache_len(cache), 1, "Length should be 1 after re-add");

    free_test_tensor(k);
    free_test_tensor(v);
    kv_cache_free(cache);
    TEST_PASS();
}

/* ============================================================================
 * 内存使用测试
 * ============================================================================ */

static void test_kv_cache_memory_usage(void) {
    /* 1 batch, 8 heads, 128 seq, 64 dim */
    KVCache* cache = kv_cache_new(128, 8, 64, 1, DTYPE_F32);
    ASSERT_NOT_NULL(cache, "Cache should not be NULL");

    /* 预期: 1 * 8 * 128 * 64 * 2 (K+V) * 4 (float) = 524288 bytes */
    size_t expected = 1 * 8 * 128 * 64 * 2 * sizeof(float) + sizeof(KVCache);
    size_t usage = kv_cache_memory_usage(cache);
    ASSERT_EQ(usage, expected, "Memory usage should match expected");

    kv_cache_free(cache);
    TEST_PASS();
}

/* ============================================================================
 * 主函数
 * ============================================================================ */

int main(void) {
    printf("========================================\n");
    printf("KV Cache Module Unit Tests\n");
    printf("========================================\n\n");

    /* 生命周期测试 */
    printf("[Lifecycle Tests]\n");
    RUN_TEST(test_kv_cache_creation);
    RUN_TEST(test_kv_cache_free_null);
    RUN_TEST(test_kv_cache_clone);
    printf("\n");

    /* 状态查询测试 */
    printf("[State Query Tests]\n");
    RUN_TEST(test_kv_cache_available);
    printf("\n");

    /* 追加操作测试 */
    printf("[Append Tests]\n");
    RUN_TEST(test_kv_cache_append_single);
    RUN_TEST(test_kv_cache_append_multiple);
    RUN_TEST(test_kv_cache_append_batch);
    RUN_TEST(test_kv_cache_overflow);
    printf("\n");

    /* 获取操作测试 */
    printf("[Get Tests]\n");
    RUN_TEST(test_kv_cache_get);
    RUN_TEST(test_kv_cache_get_slice);
    RUN_TEST(test_kv_cache_get_last);
    printf("\n");

    /* 重置测试 */
    printf("[Reset Tests]\n");
    RUN_TEST(test_kv_cache_reset);
    printf("\n");

    /* 内存测试 */
    printf("[Memory Tests]\n");
    RUN_TEST(test_kv_cache_memory_usage);
    printf("\n");

    TEST_SUMMARY();
    return TEST_RETURN();
}
