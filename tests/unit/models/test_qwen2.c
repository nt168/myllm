/**
 * @file test_qwen2.c
 * @brief Qwen2 模型单元测试
 *
 * 测试内容:
 * - 模型创建 (有/无缓存)
 * - 属性访问器
 * - KV 缓存操作
 * - Prefill 和 Decode
 * - 前向传播
 */

#include <stdio.h>
#include <stdlib.h>
#include "models/qwen2/model.h"
#include "common/test_common.h"

/* ============================================================================
 * 测试配置
 * ============================================================================ */

/**
 * @brief 创建测试用配置
 */
static Qwen2Config create_test_config(void) {
    Qwen2Config config;
    qwen2_config_init(&config);

    config.base.hidden_size = 64;
    config.base.intermediate_size = 128;
    config.base.num_attention_heads = 2;
    config.base.num_hidden_layers = 2;
    config.base.vocab_size = 1000;
    config.base.max_position_embeddings = 128;
    config.base.head_dim = 32;
    config.base.num_key_value_heads = 2;
    config.base.rope_theta = 10000.0;
    config.base.rms_norm_eps = 1e-6f;
    config.base.tie_word_embeddings = true;

    /* Qwen2 特有 */
    config.sliding_window = 4096;
    config.use_sliding_window = false;

    return config;
}

/* ============================================================================
 * 模型创建测试
 * ============================================================================ */

static void test_qwen2_model_new_without_cache(void) {
    Qwen2Config config = create_test_config();
    Qwen2Model* model = qwen2_model_new(&config);

    ASSERT_NOT_NULL(model, "Model should not be NULL");
    ASSERT_FALSE(qwen2_model_has_cache(model), "Model should not have cache");
    ASSERT_EQ(qwen2_model_cache_len(model), 0, "Cache len should be 0");

    qwen2_model_free(model);
    TEST_PASS();
}

static void test_qwen2_model_new_with_cache(void) {
    Qwen2Config config = create_test_config();
    Qwen2Model* model = qwen2_model_new_with_cache(&config, 1);

    ASSERT_NOT_NULL(model, "Model should not be NULL");
    ASSERT_TRUE(qwen2_model_has_cache(model), "Model should have cache");
    ASSERT_EQ(qwen2_model_cache_len(model), 0, "Cache len should be 0");

    qwen2_model_free(model);
    TEST_PASS();
}

/* ============================================================================
 * 模型属性测试
 * ============================================================================ */

static void test_qwen2_model_num_layers(void) {
    Qwen2Config config = create_test_config();
    Qwen2Model* model = qwen2_model_new(&config);

    ASSERT_EQ(qwen2_model_num_layers(model), 2, "Num layers should be 2");

    qwen2_model_free(model);
    TEST_PASS();
}

static void test_qwen2_model_vocab_size(void) {
    Qwen2Config config = create_test_config();
    Qwen2Model* model = qwen2_model_new(&config);

    ASSERT_EQ(qwen2_model_vocab_size(model), 1000, "Vocab size should be 1000");

    qwen2_model_free(model);
    TEST_PASS();
}

static void test_qwen2_model_name(void) {
    Qwen2Config config = create_test_config();
    Qwen2Model* model = qwen2_model_new(&config);

    const char* name = qwen2_model_name(model);
    ASSERT_NOT_NULL(name, "Model name should not be NULL");
    ASSERT_TRUE(strcmp(name, "Qwen2Model") == 0, "Model name should be 'Qwen2Model'");

    qwen2_model_free(model);
    TEST_PASS();
}

/* ============================================================================
 * Prefill 测试
 * ============================================================================ */

static void test_qwen2_model_prefill_without_cache_error(void) {
    Qwen2Config config = create_test_config();
    Qwen2Model* model = qwen2_model_new(&config);

    int32_t tokens[] = {1, 2, 3};
    Tensor* result = qwen2_model_prefill(model, tokens, 3);

    ASSERT_NULL(result, "Prefill without cache should return NULL");

    qwen2_model_free(model);
    TEST_PASS();
}

static void test_qwen2_model_prefill_with_cache(void) {
    Qwen2Config config = create_test_config();
    Qwen2Model* model = qwen2_model_new_with_cache(&config, 1);

    int32_t tokens[] = {1, 2, 3, 4};
    Tensor* logits = qwen2_model_prefill(model, tokens, 4);

    ASSERT_NOT_NULL(logits, "Prefill should return logits");
    ASSERT_EQ(qwen2_model_cache_len(model), 4, "Cache len should be 4 after prefill");

    qwen2_model_free(model);
    TEST_PASS();
}

static void test_qwen2_model_prefill_single_token(void) {
    Qwen2Config config = create_test_config();
    Qwen2Model* model = qwen2_model_new_with_cache(&config, 1);

    int32_t tokens[] = {42};
    Tensor* logits = qwen2_model_prefill(model, tokens, 1);

    ASSERT_NOT_NULL(logits, "Prefill should return logits");
    ASSERT_EQ(qwen2_model_cache_len(model), 1, "Cache len should be 1");

    qwen2_model_free(model);
    TEST_PASS();
}

/* ============================================================================
 * Decode 测试
 * ============================================================================ */

static void test_qwen2_model_decode_without_cache_error(void) {
    Qwen2Config config = create_test_config();
    Qwen2Model* model = qwen2_model_new(&config);

    Tensor* result = qwen2_model_decode_step(model, 1, 0);

    ASSERT_NULL(result, "Decode without cache should return NULL");

    qwen2_model_free(model);
    TEST_PASS();
}

static void test_qwen2_model_decode_step(void) {
    Qwen2Config config = create_test_config();
    Qwen2Model* model = qwen2_model_new_with_cache(&config, 1);

    /* First prefill */
    int32_t tokens[] = {1, 2};
    qwen2_model_prefill(model, tokens, 2);
    ASSERT_EQ(qwen2_model_cache_len(model), 2, "Cache len should be 2");

    /* Then decode */
    Tensor* logits = qwen2_model_decode_step(model, 3, 2);
    ASSERT_NOT_NULL(logits, "Decode should return logits");
    ASSERT_EQ(qwen2_model_cache_len(model), 3, "Cache len should be 3");

    qwen2_model_free(model);
    TEST_PASS();
}

static void test_qwen2_model_multiple_decode_steps(void) {
    Qwen2Config config = create_test_config();
    Qwen2Model* model = qwen2_model_new_with_cache(&config, 1);

    /* Prefill */
    int32_t tokens[] = {1, 2, 3};
    qwen2_model_prefill(model, tokens, 3);
    ASSERT_EQ(qwen2_model_cache_len(model), 3, "Cache len should be 3");

    /* Multiple decode steps */
    for (int i = 0; i < 3; i++) {
        Tensor* logits = qwen2_model_decode_step(model, 10 + i, 3 + i);
        ASSERT_NOT_NULL(logits, "Decode should return logits");
    }
    ASSERT_EQ(qwen2_model_cache_len(model), 6, "Cache len should be 6");

    qwen2_model_free(model);
    TEST_PASS();
}

/* ============================================================================
 * 缓存操作测试
 * ============================================================================ */

static void test_qwen2_model_reset_cache(void) {
    Qwen2Config config = create_test_config();
    Qwen2Model* model = qwen2_model_new_with_cache(&config, 1);

    /* Prefill */
    int32_t tokens[] = {1, 2, 3};
    qwen2_model_prefill(model, tokens, 3);
    ASSERT_EQ(qwen2_model_cache_len(model), 3, "Cache len should be 3");

    /* Reset */
    qwen2_model_reset_cache(model);
    ASSERT_EQ(qwen2_model_cache_len(model), 0, "Cache len should be 0 after reset");

    /* Can prefill again */
    int32_t tokens2[] = {4, 5};
    qwen2_model_prefill(model, tokens2, 2);
    ASSERT_EQ(qwen2_model_cache_len(model), 2, "Cache len should be 2");

    qwen2_model_free(model);
    TEST_PASS();
}

/* ============================================================================
 * 完整生成周期测试
 * ============================================================================ */

static void test_qwen2_model_full_generation_cycle(void) {
    Qwen2Config config = create_test_config();
    Qwen2Model* model = qwen2_model_new_with_cache(&config, 1);

    /* Prefill with prompt */
    int32_t prompt[] = {1, 2, 3, 4};
    qwen2_model_prefill(model, prompt, 4);
    ASSERT_EQ(qwen2_model_cache_len(model), 4, "Cache len should be 4");

    /* Generate 3 tokens */
    for (int i = 0; i < 3; i++) {
        Tensor* logits = qwen2_model_decode_step(model, 10 + i, 4 + i);
        ASSERT_NOT_NULL(logits, "Decode should return logits");
    }
    ASSERT_EQ(qwen2_model_cache_len(model), 7, "Cache len should be 7");

    /* Reset and start new generation */
    qwen2_model_reset_cache(model);
    ASSERT_EQ(qwen2_model_cache_len(model), 0, "Cache len should be 0");

    int32_t new_prompt[] = {100, 200};
    qwen2_model_prefill(model, new_prompt, 2);
    ASSERT_EQ(qwen2_model_cache_len(model), 2, "Cache len should be 2");

    qwen2_model_free(model);
    TEST_PASS();
}

/* ============================================================================
 * 主函数
 * ============================================================================ */

int main(void) {
    printf("========================================\n");
    printf("Qwen2 Model Unit Tests\n");
    printf("========================================\n\n");

    /* 模型创建测试 */
    printf("[Model Creation Tests]\n");
    RUN_TEST(test_qwen2_model_new_without_cache);
    RUN_TEST(test_qwen2_model_new_with_cache);
    printf("\n");

    /* 属性测试 */
    printf("[Model Property Tests]\n");
    RUN_TEST(test_qwen2_model_num_layers);
    RUN_TEST(test_qwen2_model_vocab_size);
    RUN_TEST(test_qwen2_model_name);
    printf("\n");

    /* Prefill 测试 */
    printf("[Prefill Tests]\n");
    RUN_TEST(test_qwen2_model_prefill_without_cache_error);
    RUN_TEST(test_qwen2_model_prefill_with_cache);
    RUN_TEST(test_qwen2_model_prefill_single_token);
    printf("\n");

    /* Decode 测试 */
    printf("[Decode Tests]\n");
    RUN_TEST(test_qwen2_model_decode_without_cache_error);
    RUN_TEST(test_qwen2_model_decode_step);
    RUN_TEST(test_qwen2_model_multiple_decode_steps);
    printf("\n");

    /* 缓存操作测试 */
    printf("[Cache Operation Tests]\n");
    RUN_TEST(test_qwen2_model_reset_cache);
    printf("\n");

    /* 完整周期测试 */
    printf("[Full Cycle Tests]\n");
    RUN_TEST(test_qwen2_model_full_generation_cycle);
    printf("\n");

    TEST_SUMMARY();
    return TEST_RETURN();
}
