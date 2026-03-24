/**
 * @file test_nn.c
 * @brief 神经网络模块单元测试
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include "nn/nn.h"
#include "common/test_common.h"

/* ============================================================================
 * Linear 测试
 * ============================================================================ */

static void test_linear_creation(void) {
    NN_Linear* linear = nn_linear_new(4, 3, true, DTYPE_F32);
    ASSERT_NOT_NULL(linear, "Linear should not be NULL");
    ASSERT_EQ(nn_linear_in_features(linear), 4, "Linear in_features should be 4");
    ASSERT_EQ(nn_linear_out_features(linear), 3, "Linear out_features should be 3");
    ASSERT_NOT_NULL(nn_linear_weight(linear), "Linear weight should not be NULL");
    ASSERT_NOT_NULL(nn_linear_bias(linear), "Linear bias should not be NULL");

    nn_linear_free(linear);
    TEST_PASS();
}

static void test_linear_no_bias(void) {
    NN_Linear* linear = nn_linear_new(4, 3, false, DTYPE_F32);
    ASSERT_NOT_NULL(linear, "Linear should not be NULL");
    ASSERT_NULL(nn_linear_bias(linear), "Linear bias should be NULL");

    nn_linear_free(linear);
    TEST_PASS();
}

static void test_linear_forward_1d(void) {
    NN_Linear* linear = nn_linear_new(4, 3, false, DTYPE_F32);
    ASSERT_NOT_NULL(linear, "Linear should not be NULL");

    /* 创建输入 [4] */
    float input_data[] = {1.0f, 2.0f, 3.0f, 4.0f};
    size_t input_dims[1] = {4};
    Tensor* input = tensor_from_f32(input_data, input_dims, 1, DTYPE_F32);
    ASSERT_NOT_NULL(input, "Input should not be NULL");

    Tensor* output = nn_linear_forward(linear, input);
    ASSERT_NOT_NULL(output, "Output should not be NULL");

    /* 输出形状应该是 [3] */
    const Shape* output_shape = tensor_shape(output);
    ASSERT_EQ(output_shape->ndim, 1, "Output should be 1D");
    ASSERT_EQ(output_shape->dims[0], 3, "Output dim should be 3");

    tensor_free(input);
    tensor_free(output);
    nn_linear_free(linear);
    TEST_PASS();
}

static void test_linear_forward_2d(void) {
    NN_Linear* linear = nn_linear_new(4, 3, false, DTYPE_F32);
    ASSERT_NOT_NULL(linear, "Linear should not be NULL");

    /* 创建输入 [2, 4] */
    float input_data[] = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f, 7.0f, 8.0f};
    size_t input_dims[2] = {2, 4};
    Tensor* input = tensor_from_f32(input_data, input_dims, 2, DTYPE_F32);
    ASSERT_NOT_NULL(input, "Input should not be NULL");

    Tensor* output = nn_linear_forward(linear, input);
    ASSERT_NOT_NULL(output, "Output should not be NULL");

    /* 输出形状应该是 [2, 3] */
    const Shape* output_shape = tensor_shape(output);
    ASSERT_EQ(output_shape->ndim, 2, "Output should be 2D");
    ASSERT_EQ(output_shape->dims[0], 2, "Output dim 0 should be 2");
    ASSERT_EQ(output_shape->dims[1], 3, "Output dim 1 should be 3");

    tensor_free(input);
    tensor_free(output);
    nn_linear_free(linear);
    TEST_PASS();
}

/* ============================================================================
 * Embedding 测试
 * ============================================================================ */

static void test_embedding_creation(void) {
    NN_Embedding* embedding = nn_embedding_new(1000, 64, DTYPE_F32);
    ASSERT_NOT_NULL(embedding, "Embedding should not be NULL");
    ASSERT_EQ(nn_embedding_vocab_size(embedding), 1000, "Vocab size should be 1000");
    ASSERT_EQ(nn_embedding_hidden_dim(embedding), 64, "Hidden dim should be 64");

    nn_embedding_free(embedding);
    TEST_PASS();
}

static void test_embedding_forward(void) {
    NN_Embedding* embedding = nn_embedding_new(100, 32, DTYPE_F32);
    ASSERT_NOT_NULL(embedding, "Embedding should not be NULL");

    int32_t token_ids[] = {1, 5, 10};
    Tensor* output = nn_embedding_forward(embedding, token_ids, 3);

    ASSERT_NOT_NULL(output, "Embedding output should not be NULL");
    const Shape* output_shape = tensor_shape(output);
    ASSERT_EQ(output_shape->ndim, 2, "Output should be 2D");
    ASSERT_EQ(output_shape->dims[0], 3, "Output dim 0 should be 3 (seq_len)");
    ASSERT_EQ(output_shape->dims[1], 32, "Output dim 1 should be 32 (hidden_dim)");

    tensor_free(output);
    nn_embedding_free(embedding);
    TEST_PASS();
}

/* ============================================================================
 * RMSNorm 测试
 * ============================================================================ */

static void test_rmsnorm_creation(void) {
    NN_RMSNorm* rmsnorm = nn_rmsnorm_new(64, 1e-5f, DTYPE_F32);
    ASSERT_NOT_NULL(rmsnorm, "RMSNorm should not be NULL");
    ASSERT_EQ(nn_rmsnorm_normalized_shape(rmsnorm), 64, "Normalized shape should be 64");

    nn_rmsnorm_free(rmsnorm);
    TEST_PASS();
}

static void test_rmsnorm_forward_1d(void) {
    NN_RMSNorm* rmsnorm = nn_rmsnorm_new(4, 1e-5f, DTYPE_F32);
    ASSERT_NOT_NULL(rmsnorm, "RMSNorm should not be NULL");

    float input_data[] = {1.0f, 2.0f, 3.0f, 4.0f};
    size_t input_dims[1] = {4};
    Tensor* input = tensor_from_f32(input_data, input_dims, 1, DTYPE_F32);

    Tensor* output = nn_rmsnorm_forward(rmsnorm, input);
    ASSERT_NOT_NULL(output, "RMSNorm output should not be NULL");

    /* 验证 RMS ≈ 1 */
    float sum_sq = 0.0f;
    for (size_t i = 0; i < 4; i++) {
        float val = tensor_get_f32(output, i);
        sum_sq += val * val;
    }
    float rms = sqrtf(sum_sq / 4.0f);
    ASSERT_FLOAT_EQ(rms, 1.0f, 1e-5, "RMS should be ~1");

    tensor_free(input);
    tensor_free(output);
    nn_rmsnorm_free(rmsnorm);
    TEST_PASS();
}

static void test_rmsnorm_forward_2d(void) {
    NN_RMSNorm* rmsnorm = nn_rmsnorm_new(4, 1e-5f, DTYPE_F32);
    ASSERT_NOT_NULL(rmsnorm, "RMSNorm should not be NULL");

    float input_data[] = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f, 7.0f, 8.0f};
    size_t input_dims[2] = {2, 4};
    Tensor* input = tensor_from_f32(input_data, input_dims, 2, DTYPE_F32);

    Tensor* output = nn_rmsnorm_forward(rmsnorm, input);
    ASSERT_NOT_NULL(output, "RMSNorm output should not be NULL");

    /* 每行的 RMS 应该 ≈ 1 */
    for (size_t row = 0; row < 2; row++) {
        float sum_sq = 0.0f;
        for (size_t col = 0; col < 4; col++) {
            float val = tensor_get_f32(output, row * 4 + col);
            sum_sq += val * val;
        }
        float rms = sqrtf(sum_sq / 4.0f);
        ASSERT_FLOAT_EQ(rms, 1.0f, 1e-5, "Row RMS should be ~1");
    }

    tensor_free(input);
    tensor_free(output);
    nn_rmsnorm_free(rmsnorm);
    TEST_PASS();
}

/* ============================================================================
 * MLP 测试
 * ============================================================================ */

static void test_mlp_creation(void) {
    NN_MLP* mlp = nn_mlp_new(64, 256, DTYPE_F32);
    ASSERT_NOT_NULL(mlp, "MLP should not be NULL");
    ASSERT_EQ(nn_mlp_hidden_dim(mlp), 64, "Hidden dim should be 64");
    ASSERT_EQ(nn_mlp_intermediate_dim(mlp), 256, "Intermediate dim should be 256");

    nn_mlp_free(mlp);
    TEST_PASS();
}

static void test_mlp_forward(void) {
    NN_MLP* mlp = nn_mlp_new(32, 64, DTYPE_F32);
    ASSERT_NOT_NULL(mlp, "MLP should not be NULL");

    float input_data[32] = {0};
    for (size_t i = 0; i < 32; i++) {
        input_data[i] = (float)i * 0.1f;
    }
    size_t input_dims[2] = {2, 32};
    Tensor* input = tensor_from_f32(input_data, input_dims, 2, DTYPE_F32);

    Tensor* output = nn_mlp_forward(mlp, input);
    ASSERT_NOT_NULL(output, "MLP output should not be NULL");

    const Shape* output_shape = tensor_shape(output);
    ASSERT_EQ(output_shape->ndim, 2, "Output should be 2D");
    ASSERT_EQ(output_shape->dims[0], 2, "Output dim 0 should be 2");
    ASSERT_EQ(output_shape->dims[1], 32, "Output dim 1 should be 32 (hidden_dim)");

    tensor_free(input);
    tensor_free(output);
    nn_mlp_free(mlp);
    TEST_PASS();
}

/* ============================================================================
 * 主函数
 * ============================================================================ */

int main(void) {
    printf("========================================\n");
    printf("NN Module Unit Tests\n");
    printf("========================================\n\n");

    /* Linear 测试 */
    printf("[Linear Tests]\n");
    RUN_TEST(test_linear_creation);
    RUN_TEST(test_linear_no_bias);
    RUN_TEST(test_linear_forward_1d);
    RUN_TEST(test_linear_forward_2d);
    printf("\n");

    /* Embedding 测试 */
    printf("[Embedding Tests]\n");
    RUN_TEST(test_embedding_creation);
    RUN_TEST(test_embedding_forward);
    printf("\n");

    /* RMSNorm 测试 */
    printf("[RMSNorm Tests]\n");
    RUN_TEST(test_rmsnorm_creation);
    RUN_TEST(test_rmsnorm_forward_1d);
    RUN_TEST(test_rmsnorm_forward_2d);
    printf("\n");

    /* MLP 测试 */
    printf("[MLP Tests]\n");
    RUN_TEST(test_mlp_creation);
    RUN_TEST(test_mlp_forward);
    printf("\n");

    TEST_SUMMARY();
    return TEST_RETURN();
}
