/**
 * @file test_qwen3.c
 * @brief Qwen3 模型单元测试
 *
 * 测试内容:
 * - 模型创建 (有/无缓存)
 * - 属性访问器
 * - KV 缓存操作
 * - Prefill 和 Decode
 * - 前向传播
 * - tie_word_embeddings 配置
 */

#include <stdio.h>
#include <stdlib.h>
#include "models/qwen3/model.h"
#include "common/test_common.h"

/* ============================================================================
 * 测试配置
 * ============================================================================ */

/**
 * @brief 创建测试用配置
 */
static Qwen3Config create_test_config(void) {
    Qwen3Config config;
    qwen3_config_init(&config);

    config.base.hidden_size = 128;
    config.base.intermediate_size = 384;
    config.base.num_attention_heads = 8;
    config.base.num_hidden_layers = 2;
    config.base.vocab_size = 1000;
    config.base.max_position_embeddings = 128;
    config.base.head_dim = 16;
    config.base.num_key_value_heads = 4;
    config.base.rope_theta = 1000000.0;
    config.base.rms_norm_eps = 1e-6f;
    config.base.tie_word_embeddings = true;

    /* Qwen3 特有 */
    config.sliding_window = 0;
    config.use_sliding_window = false;

    return config;
}

/* ============================================================================
 * 模型创建测试
 * ============================================================================ */

static void test_qwen3_model_new_without_cache(void) {
    Qwen3Config config = create_test_config();
    Qwen3Model* model = qwen3_model_new(&config);

    ASSERT_NOT_NULL(model, "Model should not be NULL");
    ASSERT_FALSE(qwen3_model_has_cache(model), "Model should not have cache");
    ASSERT_EQ(qwen3_model_cache_len(model), 0, "Cache len should be 0");

    qwen3_model_free(model);
    TEST_PASS();
}

static void test_qwen3_model_new_with_cache(void) {
    Qwen3Config config = create_test_config();
    Qwen3Model* model = qwen3_model_new_with_cache(&config, 1);

    ASSERT_NOT_NULL(model, "Model should not be NULL");
    ASSERT_TRUE(qwen3_model_has_cache(model), "Model should have cache");
    ASSERT_EQ(qwen3_model_cache_len(model), 0, "Cache len should be 0");

    qwen3_model_free(model);
    TEST_PASS();
}

/* ============================================================================
 * 模型属性测试
 * ============================================================================ */

static void test_qwen3_model_num_layers(void) {
    Qwen3Config config = create_test_config();
    Qwen3Model* model = qwen3_model_new(&config);

    ASSERT_EQ(qwen3_model_num_layers(model), 2, "Num layers should be 2");

    qwen3_model_free(model);
    TEST_PASS();
}

static void test_qwen3_model_vocab_size(void) {
    Qwen3Config config = create_test_config();
    Qwen3Model* model = qwen3_model_new(&config);

    ASSERT_EQ(qwen3_model_vocab_size(model), 1000, "Vocab size should be 1000");

    qwen3_model_free(model);
    TEST_PASS();
}

static void test_qwen3_model_name(void) {
    Qwen3Config config = create_test_config();
    Qwen3Model* model = qwen3_model_new(&config);

    const char* name = qwen3_model_name(model);
    ASSERT_NOT_NULL(name, "Model name should not be NULL");
    ASSERT_TRUE(strcmp(name, "Qwen3Model") == 0, "Model name should be 'Qwen3Model'");

    qwen3_model_free(model);
    TEST_PASS();
}

/* ============================================================================
 * 层访问器测试
 * ============================================================================ */

static void test_qwen3_model_layer_access(void) {
    Qwen3Config config = create_test_config();
    Qwen3Model* model = qwen3_model_new_with_cache(&config, 1);

    ASSERT_NOT_NULL(qwen3_model_layer(model, 0), "Layer 0 should exist");
    ASSERT_NOT_NULL(qwen3_model_layer(model, 1), "Layer 1 should exist");
    ASSERT_NULL(qwen3_model_layer(model, 2), "Layer 2 should not exist");

    qwen3_model_free(model);
    TEST_PASS();
}

static void test_qwen3_model_embed_tokens_access(void) {
    Qwen3Config config = create_test_config();
    Qwen3Model* model = qwen3_model_new(&config);

    Embedding* embed = qwen3_model_embed_tokens(model);
    ASSERT_NOT_NULL(embed, "Embedding should not be NULL");

    qwen3_model_free(model);
    TEST_PASS();
}

static void test_qwen3_model_norm_access(void) {
    Qwen3Config config = create_test_config();
    Qwen3Model* model = qwen3_model_new(&config);

    RMSNorm* norm = qwen3_model_norm(model);
    ASSERT_NOT_NULL(norm, "Norm should not be NULL");

    qwen3_model_free(model);
    TEST_PASS();
}

/* ============================================================================
 * tie_word_embeddings 测试
 * ============================================================================ */

static void test_qwen3_model_tied_embeddings(void) {
    Qwen3Config config = create_test_config();
    config.base.tie_word_embeddings = true;
    Qwen3Model* model = qwen3_model_new(&config);

    Linear* lm_head = qwen3_model_lm_head(model);
    ASSERT_NULL(lm_head, "lm_head should be NULL when tie_word_embeddings=true");
    ASSERT_TRUE(qwen3_model_tie_word_embeddings(model), "tie_word_embeddings should be true");

    qwen3_model_free(model);
    TEST_PASS();
}

static void test_qwen3_model_untied_embeddings(void) {
    Qwen3Config config = create_test_config();
    config.base.tie_word_embeddings = false;
    Qwen3Model* model = qwen3_model_new(&config);

    Linear* lm_head = qwen3_model_lm_head(model);
    ASSERT_NOT_NULL(lm_head, "lm_head should not be NULL when tie_word_embeddings=false");
    ASSERT_FALSE(qwen3_model_tie_word_embeddings(model), "tie_word_embeddings should be false");

    qwen3_model_free(model);
    TEST_PASS();
}

/* ============================================================================
 * Prefill 测试
 * ============================================================================ */

static void test_qwen3_model_prefill_without_cache_error(void) {
    Qwen3Config config = create_test_config();
    Qwen3Model* model = qwen3_model_new(&config);

    int32_t tokens[] = {1, 2, 3};
    Tensor* result = qwen3_model_prefill(model, tokens, 3);

    ASSERT_NULL(result, "Prefill without cache should return NULL");

    qwen3_model_free(model);
    TEST_PASS();
}

static void test_qwen3_model_prefill_with_cache(void) {
    Qwen3Config config = create_test_config();
    Qwen3Model* model = qwen3_model_new_with_cache(&config, 1);

    int32_t tokens[] = {1, 2, 3, 4};
    Tensor* logits = qwen3_model_prefill(model, tokens, 4);

    ASSERT_NOT_NULL(logits, "Prefill should return logits");
    ASSERT_EQ(qwen3_model_cache_len(model), 4, "Cache len should be 4 after prefill");

    qwen3_model_free(model);
    TEST_PASS();
}

static void test_qwen3_model_prefill_single_token(void) {
    Qwen3Config config = create_test_config();
    Qwen3Model* model = qwen3_model_new_with_cache(&config, 1);

    int32_t tokens[] = {42};
    Tensor* logits = qwen3_model_prefill(model, tokens, 1);

    ASSERT_NOT_NULL(logits, "Prefill should return logits");
    ASSERT_EQ(qwen3_model_cache_len(model), 1, "Cache len should be 1");

    qwen3_model_free(model);
    TEST_PASS();
}

/* ============================================================================
 * Decode 测试
 * ============================================================================ */

static void test_qwen3_model_decode_without_cache_error(void) {
    Qwen3Config config = create_test_config();
    Qwen3Model* model = qwen3_model_new(&config);

    Tensor* result = qwen3_model_decode_step(model, 1, 0);

    ASSERT_NULL(result, "Decode without cache should return NULL");

    qwen3_model_free(model);
    TEST_PASS();
}

static void test_qwen3_model_decode_step(void) {
    Qwen3Config config = create_test_config();
    Qwen3Model* model = qwen3_model_new_with_cache(&config, 1);

    /* First prefill */
    int32_t tokens[] = {1, 2};
    qwen3_model_prefill(model, tokens, 2);
    ASSERT_EQ(qwen3_model_cache_len(model), 2, "Cache len should be 2");

    /* Then decode */
    Tensor* logits = qwen3_model_decode_step(model, 3, 2);
    ASSERT_NOT_NULL(logits, "Decode should return logits");
    ASSERT_EQ(qwen3_model_cache_len(model), 3, "Cache len should be 3");

    qwen3_model_free(model);
    TEST_PASS();
}

static void test_qwen3_model_multiple_decode_steps(void) {
    Qwen3Config config = create_test_config();
    Qwen3Model* model = qwen3_model_new_with_cache(&config, 1);

    /* Prefill */
    int32_t tokens[] = {1, 2, 3};
    qwen3_model_prefill(model, tokens, 3);
    ASSERT_EQ(qwen3_model_cache_len(model), 3, "Cache len should be 3");

    /* Multiple decode steps */
    for (int i = 0; i < 3; i++) {
        Tensor* logits = qwen3_model_decode_step(model, 10 + i, 3 + i);
        ASSERT_NOT_NULL(logits, "Decode should return logits");
    }
    ASSERT_EQ(qwen3_model_cache_len(model), 6, "Cache len should be 6");

    qwen3_model_free(model);
    TEST_PASS();
}

/* ============================================================================
 * 缓存操作测试
 * ============================================================================ */

static void test_qwen3_model_reset_cache(void) {
    Qwen3Config config = create_test_config();
    Qwen3Model* model = qwen3_model_new_with_cache(&config, 1);

    /* Prefill */
    int32_t tokens[] = {1, 2, 3};
    qwen3_model_prefill(model, tokens, 3);
    ASSERT_EQ(qwen3_model_cache_len(model), 3, "Cache len should be 3");

    /* Reset */
    qwen3_model_reset_cache(model);
    ASSERT_EQ(qwen3_model_cache_len(model), 0, "Cache len should be 0 after reset");

    /* Can prefill again */
    int32_t tokens2[] = {4, 5};
    qwen3_model_prefill(model, tokens2, 2);
    ASSERT_EQ(qwen3_model_cache_len(model), 2, "Cache len should be 2");

    qwen3_model_free(model);
    TEST_PASS();
}

/* ============================================================================
 * 完整生成周期测试
 * ============================================================================ */

static void test_qwen3_model_full_generation_cycle(void) {
    Qwen3Config config = create_test_config();
    Qwen3Model* model = qwen3_model_new_with_cache(&config, 1);

    /* Prefill with prompt */
    int32_t prompt[] = {1, 2, 3, 4};
    qwen3_model_prefill(model, prompt, 4);
    ASSERT_EQ(qwen3_model_cache_len(model), 4, "Cache len should be 4");

    /* Generate 3 tokens */
    for (int i = 0; i < 3; i++) {
        Tensor* logits = qwen3_model_decode_step(model, 10 + i, 4 + i);
        ASSERT_NOT_NULL(logits, "Decode should return logits");
    }
    ASSERT_EQ(qwen3_model_cache_len(model), 7, "Cache len should be 7");

    /* Reset and start new generation */
    qwen3_model_reset_cache(model);
    ASSERT_EQ(qwen3_model_cache_len(model), 0, "Cache len should be 0");

    int32_t new_prompt[] = {100, 200};
    qwen3_model_prefill(model, new_prompt, 2);
    ASSERT_EQ(qwen3_model_cache_len(model), 2, "Cache len should be 2");

    qwen3_model_free(model);
    TEST_PASS();
}

/* ============================================================================
 * GQA 测试 (Grouped Query Attention)
 * ============================================================================ */

static void test_qwen3_model_gqa_config(void) {
    Qwen3Config config = create_test_config();
    /* num_heads=8, num_kv_heads=4 -> GQA with group size 2 */
    Qwen3Model* model = qwen3_model_new(&config);

    ASSERT_NOT_NULL(model, "Model should not be NULL");
    ASSERT_EQ(qwen3_model_num_layers(model), 2, "Num layers should be 2");

    qwen3_model_free(model);
    TEST_PASS();
}

/* ============================================================================
 * 主函数
 * ============================================================================ */

int main(void) {
    printf("========================================\n");
    printf("Qwen3 Model Unit Tests\n");
    printf("========================================\n\n");

    /* 模型创建测试 */
    printf("[Model Creation Tests]\n");
    RUN_TEST(test_qwen3_model_new_without_cache);
    RUN_TEST(test_qwen3_model_new_with_cache);
    printf("\n");

    /* 属性测试 */
    printf("[Model Property Tests]\n");
    RUN_TEST(test_qwen3_model_num_layers);
    RUN_TEST(test_qwen3_model_vocab_size);
    RUN_TEST(test_qwen3_model_name);
    printf("\n");

    /* 层访问器测试 */
    printf("[Layer Accessor Tests]\n");
    RUN_TEST(test_qwen3_model_layer_access);
    RUN_TEST(test_qwen3_model_embed_tokens_access);
    RUN_TEST(test_qwen3_model_norm_access);
    printf("\n");

    /* tie_word_embeddings 测试 */
    printf("[Tie Embeddings Tests]\n");
    RUN_TEST(test_qwen3_model_tied_embeddings);
    RUN_TEST(test_qwen3_model_untied_embeddings);
    printf("\n");

    /* Prefill 测试 */
    printf("[Prefill Tests]\n");
    RUN_TEST(test_qwen3_model_prefill_without_cache_error);
    RUN_TEST(test_qwen3_model_prefill_with_cache);
    RUN_TEST(test_qwen3_model_prefill_single_token);
    printf("\n");

    /* Decode 测试 */
    printf("[Decode Tests]\n");
    RUN_TEST(test_qwen3_model_decode_without_cache_error);
    RUN_TEST(test_qwen3_model_decode_step);
    RUN_TEST(test_qwen3_model_multiple_decode_steps);
    printf("\n");

    /* 缓存操作测试 */
    printf("[Cache Operation Tests]\n");
    RUN_TEST(test_qwen3_model_reset_cache);
    printf("\n");

    /* 完整周期测试 */
    printf("[Full Cycle Tests]\n");
    RUN_TEST(test_qwen3_model_full_generation_cycle);
    printf("\n");

    /* GQA 测试 */
    printf("[GQA Tests]\n");
    RUN_TEST(test_qwen3_model_gqa_config);
    printf("\n");

    TEST_SUMMARY();
    return TEST_RETURN();
}
