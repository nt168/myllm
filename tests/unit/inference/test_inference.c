/**
 * @file test_inference.c
 * @brief 推理引擎模块单元测试
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <assert.h>
#include "inference/engine.h"

/* 测试计数器 */
static int tests_passed = 0;
static int tests_total = 0;

/* 测试宏 */
#define TEST_START(name) \
    do { \
        printf("  [TEST] %s... ", name); \
        tests_total++; \
    } while(0)

#define TEST_PASS() \
    do { \
        printf("PASSED\n"); \
        tests_passed++; \
    } while(0)

#define TEST_FAIL(msg) \
    do { \
        printf("FAILED: %s\n", msg); \
    } while(0)

#define ASSERT_TRUE(cond, msg) \
    do { \
        if (!(cond)) { \
            TEST_FAIL(msg); \
            return; \
        } \
    } while(0)

#define ASSERT_FALSE(cond, msg) ASSERT_TRUE(!(cond), msg)
#define ASSERT_EQ(a, b, msg) ASSERT_TRUE((a) == (b), msg)
#define ASSERT_NE(a, b, msg) ASSERT_TRUE((a) != (b), msg)
#define ASSERT_NOT_NULL(p, msg) ASSERT_TRUE((p) != NULL, msg)
#define ASSERT_NULL(p, msg) ASSERT_TRUE((p) == NULL, msg)

/* ============================================================================
 * 生成配置测试
 * ============================================================================ */

static void test_generate_config_default(void) {
    TEST_START("GenerateConfig default values");

    GenerateConfig config = generate_config_default();

    ASSERT_EQ(config.max_tokens, ENGINE_DEFAULT_MAX_TOKENS, "max_tokens default");
    ASSERT_TRUE(config.temperature > 0.0f, "temperature > 0");
    ASSERT_TRUE(config.top_p > 0.0f && config.top_p <= 1.0f, "top_p in (0,1]");
    ASSERT_TRUE(config.top_k > 0, "top_k > 0");
    ASSERT_TRUE(config.repetition_penalty >= 1.0f, "repetition_penalty >= 1.0");
    ASSERT_NULL(config.stop_tokens, "stop_tokens is NULL");
    ASSERT_EQ(config.num_stop_tokens, 0, "num_stop_tokens is 0");

    TEST_PASS();
}

static void test_generate_config_values(void) {
    TEST_START("GenerateConfig custom values");

    GenerateConfig config = generate_config_default();
    config.max_tokens = 512;
    config.temperature = 0.7f;
    config.top_k = 100;
    config.top_p = 0.95f;

    ASSERT_EQ(config.max_tokens, 512, "max_tokens custom");
    ASSERT_TRUE(config.temperature > 0.69f && config.temperature < 0.71f, "temperature custom");
    ASSERT_EQ(config.top_k, 100, "top_k custom");
    ASSERT_TRUE(config.top_p > 0.94f && config.top_p < 0.96f, "top_p custom");

    TEST_PASS();
}

/* ============================================================================
 * 生成结果测试
 * ============================================================================ */

static void test_generate_result_free_null(void) {
    TEST_START("GenerateResult free with NULL");

    /* 测试 NULL 指针安全 */
    generate_result_free(NULL);

    TEST_PASS();
}

static void test_generate_result_free_with_data(void) {
    TEST_START("GenerateResult free with data");

    GenerateResult result = {0};

    /* 分配一些数据 */
    result.tokens = (int32_t*)malloc(10 * sizeof(int32_t));
    result.num_tokens = 10;
    result.text = strdup("Hello, world!");
    result.text_len = 13;

    for (int i = 0; i < 10; i++) {
        result.tokens[i] = i;
    }

    /* 释放 */
    generate_result_free(&result);

    ASSERT_NULL(result.tokens, "tokens freed");
    ASSERT_NULL(result.text, "text freed");
    ASSERT_EQ(result.num_tokens, 0, "num_tokens reset");

    TEST_PASS();
}

/* ============================================================================
 * 引擎创建/释放测试
 * ============================================================================ */

static void test_engine_new_null(void) {
    TEST_START("Engine new with NULL path");

    InferenceEngine* engine = engine_new(NULL);
    ASSERT_NULL(engine, "engine_new returns NULL for NULL path");

    TEST_PASS();
}

static void test_engine_new_with_components_null(void) {
    TEST_START("Engine new with NULL components");

    InferenceEngine* engine = engine_new_with_components(NULL, NULL, NULL);
    ASSERT_NULL(engine, "engine_new_with_components returns NULL");

    TEST_PASS();
}

static void test_engine_free_null(void) {
    TEST_START("Engine free with NULL");

    /* 测试 NULL 指针安全 */
    engine_free(NULL);

    TEST_PASS();
}

/* ============================================================================
 * 引擎属性测试
 * ============================================================================ */

static void test_engine_get_config_null(void) {
    TEST_START("Engine get config with NULL");

    const ModelConfig* config = engine_get_config(NULL);
    ASSERT_NULL(config, "get_config returns NULL for NULL engine");

    TEST_PASS();
}

static void test_engine_vocab_size_null(void) {
    TEST_START("Engine vocab size with NULL");

    size_t vocab = engine_vocab_size(NULL);
    ASSERT_EQ(vocab, 0, "vocab_size returns 0 for NULL");

    TEST_PASS();
}

static void test_engine_num_layers_null(void) {
    TEST_START("Engine num layers with NULL");

    size_t layers = engine_num_layers(NULL);
    ASSERT_EQ(layers, 0, "num_layers returns 0 for NULL");

    TEST_PASS();
}

static void test_engine_hidden_size_null(void) {
    TEST_START("Engine hidden size with NULL");

    size_t hidden = engine_hidden_size(NULL);
    ASSERT_EQ(hidden, 0, "hidden_size returns 0 for NULL");

    TEST_PASS();
}

/* ============================================================================
 * 编码/解码测试
 * ============================================================================ */

static void test_engine_encode_null(void) {
    TEST_START("Engine encode with NULL");

    int32_t ids[10];
    int result = engine_encode(NULL, "test", true, ids, 10);
    ASSERT_EQ(result, -1, "encode returns -1 for NULL engine");

    TEST_PASS();
}

static void test_engine_decode_null(void) {
    TEST_START("Engine decode with NULL");

    int32_t ids[] = {1, 2, 3};
    char out[100];
    int result = engine_decode(NULL, ids, 3, true, out, 100);
    ASSERT_EQ(result, -1, "decode returns -1 for NULL engine");

    TEST_PASS();
}

static void test_engine_special_ids_null(void) {
    TEST_START("Engine special IDs with NULL");

    ASSERT_EQ(engine_bos_id(NULL), -1, "bos_id returns -1 for NULL");
    ASSERT_EQ(engine_eos_id(NULL), -1, "eos_id returns -1 for NULL");
    ASSERT_EQ(engine_pad_id(NULL), -1, "pad_id returns -1 for NULL");

    TEST_PASS();
}

/* ============================================================================
 * 推理接口测试
 * ============================================================================ */

static void test_engine_prefill_null(void) {
    TEST_START("Engine prefill with NULL");

    int32_t tokens[] = {1, 2, 3};
    Tensor* result = engine_prefill(NULL, tokens, 3);
    ASSERT_NULL(result, "prefill returns NULL for NULL engine");

    TEST_PASS();
}

static void test_engine_prefill_empty(void) {
    TEST_START("Engine prefill with empty tokens");

    /* NULL 引擎应该返回 NULL */
    Tensor* result = engine_prefill(NULL, NULL, 0);
    ASSERT_NULL(result, "prefill returns NULL for NULL engine");

    TEST_PASS();
}

static void test_engine_decode_step_null(void) {
    TEST_START("Engine decode step with NULL");

    Tensor* result = engine_decode_step(NULL, 1);
    ASSERT_NULL(result, "decode_step returns NULL for NULL engine");

    TEST_PASS();
}

static void test_engine_cache_len_null(void) {
    TEST_START("Engine cache len with NULL");

    size_t len = engine_cache_len(NULL);
    ASSERT_EQ(len, 0, "cache_len returns 0 for NULL");

    TEST_PASS();
}

/* ============================================================================
 * 生成接口测试
 * ============================================================================ */

static void test_engine_generate_null(void) {
    TEST_START("Engine generate with NULL");

    GenerateConfig config = generate_config_default();
    GenerateResult result = engine_generate(NULL, "Hello", &config);

    ASSERT_NULL(result.tokens, "generate returns empty result for NULL engine");
    ASSERT_NULL(result.text, "generate returns empty text for NULL engine");

    TEST_PASS();
}

static void test_engine_generate_from_tokens_null(void) {
    TEST_START("Engine generate from tokens with NULL");

    int32_t tokens[] = {1, 2, 3};
    GenerateConfig config = generate_config_default();
    GenerateResult result = engine_generate_from_tokens(NULL, tokens, 3, &config);

    ASSERT_NULL(result.tokens, "generate_from_tokens returns empty result for NULL");

    TEST_PASS();
}

static void test_engine_generate_stream_null(void) {
    TEST_START("Engine generate stream with NULL");

    GenerateConfig config = generate_config_default();
    size_t count = engine_generate_stream(NULL, "Hello", &config, NULL, NULL);
    ASSERT_EQ(count, 0, "generate_stream returns 0 for NULL engine");

    TEST_PASS();
}

/* ============================================================================
 * 模拟生成测试
 * ============================================================================ */

/* 简单的回调计数器 */
typedef struct {
    size_t count;
    int32_t last_token;
    char last_text[256];
} CallbackState;

static bool test_callback(int32_t token, const char* text, void* user_data) {
    CallbackState* state = (CallbackState*)user_data;
    state->count++;
    state->last_token = token;
    if (text) {
        strncpy(state->last_text, text, sizeof(state->last_text) - 1);
    }
    return true;  /* 继续生成 */
}

static void test_engine_stream_callback(void) {
    TEST_START("Engine stream callback");

    /* 由于 InferenceEngine 是不透明类型，无法直接创建实例 */
    /* 这里只测试 NULL 情况，完整测试需要使用 engine_new */

    GenerateConfig config = generate_config_default();
    config.max_tokens = 5;

    CallbackState state = {0};

    /* 测试 NULL 引擎不会崩溃 */
    size_t count = engine_generate_stream(NULL, "test", &config, test_callback, &state);

    /* NULL 引擎应该返回 0 */
    ASSERT_EQ(count, 0, "generate_stream returns 0 for NULL");

    TEST_PASS();
}

/* ============================================================================
 * 主函数
 * ============================================================================ */

int main(void) {
    printf("\n========================================\n");
    printf("Inference Engine Module Unit Tests\n");
    printf("========================================\n\n");

    printf("[Config Tests]\n");
    test_generate_config_default();
    test_generate_config_values();

    printf("\n[Result Tests]\n");
    test_generate_result_free_null();
    test_generate_result_free_with_data();

    printf("\n[Engine Lifecycle Tests]\n");
    test_engine_new_null();
    test_engine_new_with_components_null();
    test_engine_free_null();

    printf("\n[Engine Property Tests]\n");
    test_engine_get_config_null();
    test_engine_vocab_size_null();
    test_engine_num_layers_null();
    test_engine_hidden_size_null();

    printf("\n[Encode/Decode Tests]\n");
    test_engine_encode_null();
    test_engine_decode_null();
    test_engine_special_ids_null();

    printf("\n[Inference Tests]\n");
    test_engine_prefill_null();
    test_engine_prefill_empty();
    test_engine_decode_step_null();
    test_engine_cache_len_null();

    printf("\n[Generation Tests]\n");
    test_engine_generate_null();
    test_engine_generate_from_tokens_null();
    test_engine_generate_stream_null();
    test_engine_stream_callback();

    printf("\n========================================\n");
    printf("Tests run: %d\n", tests_total);
    printf("Passed: %d\n", tests_passed);
    printf("Failed: %d\n", tests_total - tests_passed);
    printf("========================================\n\n");

    return (tests_passed == tests_total) ? 0 : 1;
}
