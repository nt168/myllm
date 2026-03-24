/**
 * @file test_ops.c
 * @brief 运算模块单元测试
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include "ops/ops.h"
#include "common/test_common.h"

/* ============================================================================
 * Elementwise 测试
 * ============================================================================ */

static void test_add_simple(void) {
    float a_data[] = {1.0f, 2.0f, 3.0f, 4.0f};
    float b_data[] = {1.0f, 2.0f, 3.0f, 4.0f};
    size_t dims[] = {2, 2};

    Tensor* a = tensor_from_f32(a_data, dims, 2, DTYPE_F32);
    Tensor* b = tensor_from_f32(b_data, dims, 2, DTYPE_F32);

    Tensor* result = ops_add(a, b);
    ASSERT_NOT_NULL(result, "Add result should not be NULL");

    float expected[] = {2.0f, 4.0f, 6.0f, 8.0f};
    for (size_t i = 0; i < 4; i++) {
        ASSERT_FLOAT_EQ(tensor_get_f32(result, i), expected[i], 1e-6, "Add element mismatch");
    }

    tensor_free(a);
    tensor_free(b);
    tensor_free(result);
    TEST_PASS();
}

static void test_add_broadcast(void) {
    float a_data[] = {1.0f, 2.0f, 3.0f, 4.0f};
    float b_data[] = {1.0f, 1.0f};
    size_t dims_a[] = {2, 2};
    size_t dims_b[] = {1, 2};

    Tensor* a = tensor_from_f32(a_data, dims_a, 2, DTYPE_F32);
    Tensor* b = tensor_from_f32(b_data, dims_b, 2, DTYPE_F32);

    Tensor* result = ops_add(a, b);
    ASSERT_NOT_NULL(result, "Broadcast add result should not be NULL");

    float expected[] = {2.0f, 3.0f, 4.0f, 5.0f};
    for (size_t i = 0; i < 4; i++) {
        ASSERT_FLOAT_EQ(tensor_get_f32(result, i), expected[i], 1e-6, "Broadcast add element mismatch");
    }

    tensor_free(a);
    tensor_free(b);
    tensor_free(result);
    TEST_PASS();
}

static void test_mul_simple(void) {
    float a_data[] = {1.0f, 2.0f, 3.0f, 4.0f};
    float b_data[] = {2.0f, 2.0f, 2.0f, 2.0f};
    size_t dims[] = {2, 2};

    Tensor* a = tensor_from_f32(a_data, dims, 2, DTYPE_F32);
    Tensor* b = tensor_from_f32(b_data, dims, 2, DTYPE_F32);

    Tensor* result = ops_mul(a, b);
    ASSERT_NOT_NULL(result, "Mul result should not be NULL");

    float expected[] = {2.0f, 4.0f, 6.0f, 8.0f};
    for (size_t i = 0; i < 4; i++) {
        ASSERT_FLOAT_EQ(tensor_get_f32(result, i), expected[i], 1e-6, "Mul element mismatch");
    }

    tensor_free(a);
    tensor_free(b);
    tensor_free(result);
    TEST_PASS();
}

static void test_sub_simple(void) {
    float a_data[] = {5.0f, 4.0f, 3.0f, 2.0f};
    float b_data[] = {1.0f, 2.0f, 1.0f, 2.0f};
    size_t dims[] = {2, 2};

    Tensor* a = tensor_from_f32(a_data, dims, 2, DTYPE_F32);
    Tensor* b = tensor_from_f32(b_data, dims, 2, DTYPE_F32);

    Tensor* result = ops_sub(a, b);
    ASSERT_NOT_NULL(result, "Sub result should not be NULL");

    float expected[] = {4.0f, 2.0f, 2.0f, 0.0f};
    for (size_t i = 0; i < 4; i++) {
        ASSERT_FLOAT_EQ(tensor_get_f32(result, i), expected[i], 1e-6, "Sub element mismatch");
    }

    tensor_free(a);
    tensor_free(b);
    tensor_free(result);
    TEST_PASS();
}

static void test_div_simple(void) {
    float a_data[] = {10.0f, 8.0f, 6.0f, 4.0f};
    float b_data[] = {2.0f, 4.0f, 2.0f, 1.0f};
    size_t dims[] = {2, 2};

    Tensor* a = tensor_from_f32(a_data, dims, 2, DTYPE_F32);
    Tensor* b = tensor_from_f32(b_data, dims, 2, DTYPE_F32);

    Tensor* result = ops_div(a, b);
    ASSERT_NOT_NULL(result, "Div result should not be NULL");

    float expected[] = {5.0f, 2.0f, 3.0f, 4.0f};
    for (size_t i = 0; i < 4; i++) {
        ASSERT_FLOAT_EQ(tensor_get_f32(result, i), expected[i], 1e-6, "Div element mismatch");
    }

    tensor_free(a);
    tensor_free(b);
    tensor_free(result);
    TEST_PASS();
}

/* ============================================================================
 * Activation 测试
 * ============================================================================ */

static void test_gelu_basic(void) {
    float data[] = {0.0f, 1.0f, -1.0f, 2.0f};
    size_t dims[] = {4};

    Tensor* input = tensor_from_f32(data, dims, 1, DTYPE_F32);
    Tensor* result = ops_gelu(input);

    ASSERT_NOT_NULL(result, "GELU result should not be NULL");

    /* GELU(0) ≈ 0 */
    ASSERT_FLOAT_EQ(tensor_get_f32(result, 0), 0.0f, 1e-5, "GELU(0) should be ~0");

    /* GELU(1) ≈ 0.841 */
    ASSERT_FLOAT_EQ(tensor_get_f32(result, 1), 0.841f, 0.01f, "GELU(1) should be ~0.841");

    /* GELU(-1) ≈ -0.159 */
    ASSERT_FLOAT_EQ(tensor_get_f32(result, 2), -0.159f, 0.01f, "GELU(-1) should be ~-0.159");

    tensor_free(input);
    tensor_free(result);
    TEST_PASS();
}

static void test_silu_basic(void) {
    float data[] = {0.0f, 1.0f, -1.0f, 2.0f};
    size_t dims[] = {4};

    Tensor* input = tensor_from_f32(data, dims, 1, DTYPE_F32);
    Tensor* result = ops_silu(input);

    ASSERT_NOT_NULL(result, "SiLU result should not be NULL");

    /* SiLU(0) = 0 */
    ASSERT_FLOAT_EQ(tensor_get_f32(result, 0), 0.0f, 1e-5, "SiLU(0) should be 0");

    /* SiLU(1) ≈ 0.731 */
    ASSERT_FLOAT_EQ(tensor_get_f32(result, 1), 0.731f, 0.01f, "SiLU(1) should be ~0.731");

    tensor_free(input);
    tensor_free(result);
    TEST_PASS();
}

static void test_relu_basic(void) {
    float data[] = {-2.0f, -1.0f, 0.0f, 1.0f, 2.0f};
    size_t dims[] = {5};

    Tensor* input = tensor_from_f32(data, dims, 1, DTYPE_F32);
    Tensor* result = ops_relu(input);

    ASSERT_NOT_NULL(result, "ReLU result should not be NULL");

    ASSERT_FLOAT_EQ(tensor_get_f32(result, 0), 0.0f, 1e-6, "ReLU(-2) should be 0");
    ASSERT_FLOAT_EQ(tensor_get_f32(result, 1), 0.0f, 1e-6, "ReLU(-1) should be 0");
    ASSERT_FLOAT_EQ(tensor_get_f32(result, 2), 0.0f, 1e-6, "ReLU(0) should be 0");
    ASSERT_FLOAT_EQ(tensor_get_f32(result, 3), 1.0f, 1e-6, "ReLU(1) should be 1");
    ASSERT_FLOAT_EQ(tensor_get_f32(result, 4), 2.0f, 1e-6, "ReLU(2) should be 2");

    tensor_free(input);
    tensor_free(result);
    TEST_PASS();
}

/* ============================================================================
 * Normalization 测试
 * ============================================================================ */

static void test_layernorm_1d(void) {
    float data[] = {1.0f, 2.0f, 3.0f, 4.0f};
    size_t dims[] = {4};

    Tensor* input = tensor_from_f32(data, dims, 1, DTYPE_F32);
    Tensor* result = ops_layernorm(input, 4, NULL, NULL, 1e-5f);

    ASSERT_NOT_NULL(result, "LayerNorm result should not be NULL");

    /* 均值应该接近 0 */
    float mean = 0.0f;
    for (size_t i = 0; i < 4; i++) {
        mean += tensor_get_f32(result, i);
    }
    mean /= 4.0f;
    ASSERT_FLOAT_EQ(mean, 0.0f, 1e-5, "LayerNorm mean should be ~0");

    tensor_free(input);
    tensor_free(result);
    TEST_PASS();
}

static void test_layernorm_2d(void) {
    float data[] = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f, 7.0f, 8.0f};
    size_t dims[] = {2, 4};

    Tensor* input = tensor_from_f32(data, dims, 2, DTYPE_F32);
    Tensor* result = ops_layernorm(input, 4, NULL, NULL, 1e-5f);

    ASSERT_NOT_NULL(result, "LayerNorm 2D result should not be NULL");

    /* 每行的均值应该接近 0 */
    for (size_t row = 0; row < 2; row++) {
        float mean = 0.0f;
        for (size_t col = 0; col < 4; col++) {
            mean += tensor_get_f32(result, row * 4 + col);
        }
        mean /= 4.0f;
        ASSERT_FLOAT_EQ(mean, 0.0f, 1e-5, "LayerNorm row mean should be ~0");
    }

    tensor_free(input);
    tensor_free(result);
    TEST_PASS();
}

static void test_rmsnorm_basic(void) {
    float data[] = {1.0f, 2.0f, 3.0f, 4.0f};
    size_t dims[] = {4};

    Tensor* input = tensor_from_f32(data, dims, 1, DTYPE_F32);
    Tensor* result = ops_rmsnorm(input, 4, NULL, 1e-5f);

    ASSERT_NOT_NULL(result, "RMSNorm result should not be NULL");

    /* RMS 应该接近 1 */
    float sum_sq = 0.0f;
    for (size_t i = 0; i < 4; i++) {
        float val = tensor_get_f32(result, i);
        sum_sq += val * val;
    }
    float rms = sqrtf(sum_sq / 4.0f);
    ASSERT_FLOAT_EQ(rms, 1.0f, 1e-5, "RMSNorm rms should be ~1");

    tensor_free(input);
    tensor_free(result);
    TEST_PASS();
}

/* ============================================================================
 * MatMul 测试
 * ============================================================================ */

static void test_matmul_2d(void) {
    float a_data[] = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f};  /* [2, 3] */
    float b_data[] = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f};  /* [3, 2] */
    size_t dims_a[] = {2, 3};
    size_t dims_b[] = {3, 2};

    Tensor* a = tensor_from_f32(a_data, dims_a, 2, DTYPE_F32);
    Tensor* b = tensor_from_f32(b_data, dims_b, 2, DTYPE_F32);

    Tensor* result = ops_matmul(a, b);
    ASSERT_NOT_NULL(result, "MatMul result should not be NULL");

    /* 结果形状应该是 [2, 2] */
    ASSERT_EQ(tensor_ndim(result), 2, "MatMul result ndim should be 2");
    ASSERT_EQ(tensor_shape(result)->dims[0], 2, "MatMul result dim0 should be 2");
    ASSERT_EQ(tensor_shape(result)->dims[1], 2, "MatMul result dim1 should be 2");

    /* 验证计算结果 */
    /* [1,2,3] x [1,2; 3,4; 5,6] = [22, 28] */
    /* [4,5,6] x [1,2; 3,4; 5,6] = [49, 64] */
    ASSERT_FLOAT_EQ(tensor_get_f32(result, 0), 22.0f, 1e-4, "MatMul element 0");
    ASSERT_FLOAT_EQ(tensor_get_f32(result, 1), 28.0f, 1e-4, "MatMul element 1");
    ASSERT_FLOAT_EQ(tensor_get_f32(result, 2), 49.0f, 1e-4, "MatMul element 2");
    ASSERT_FLOAT_EQ(tensor_get_f32(result, 3), 64.0f, 1e-4, "MatMul element 3");

    tensor_free(a);
    tensor_free(b);
    tensor_free(result);
    TEST_PASS();
}

static void test_matmul_3d(void) {
    /* 2 个 batch，每个 [2, 3] */
    float a_data[] = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f,
                      1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f};
    float b_data[] = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f,
                      1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f};
    size_t dims_a[] = {2, 2, 3};
    size_t dims_b[] = {2, 3, 2};

    Tensor* a = tensor_from_f32(a_data, dims_a, 3, DTYPE_F32);
    Tensor* b = tensor_from_f32(b_data, dims_b, 3, DTYPE_F32);

    Tensor* result = ops_matmul(a, b);
    ASSERT_NOT_NULL(result, "Batch MatMul result should not be NULL");

    /* 结果形状应该是 [2, 2, 2] */
    ASSERT_EQ(tensor_ndim(result), 3, "Batch MatMul result ndim should be 3");
    ASSERT_EQ(tensor_shape(result)->dims[0], 2, "Batch MatMul result dim0 should be 2");
    ASSERT_EQ(tensor_shape(result)->dims[1], 2, "Batch MatMul result dim1 should be 2");
    ASSERT_EQ(tensor_shape(result)->dims[2], 2, "Batch MatMul result dim2 should be 2");

    tensor_free(a);
    tensor_free(b);
    tensor_free(result);
    TEST_PASS();
}

/* ============================================================================
 * 主函数
 * ============================================================================ */

int main(void) {
    printf("========================================\n");
    printf("Ops Module Unit Tests\n");
    printf("========================================\n\n");

    /* Elementwise 测试 */
    printf("[Elementwise Tests]\n");
    RUN_TEST(test_add_simple);
    RUN_TEST(test_add_broadcast);
    RUN_TEST(test_mul_simple);
    RUN_TEST(test_sub_simple);
    RUN_TEST(test_div_simple);
    printf("\n");

    /* Activation 测试 */
    printf("[Activation Tests]\n");
    RUN_TEST(test_gelu_basic);
    RUN_TEST(test_silu_basic);
    RUN_TEST(test_relu_basic);
    printf("\n");

    /* Normalization 测试 */
    printf("[Normalization Tests]\n");
    RUN_TEST(test_layernorm_1d);
    RUN_TEST(test_layernorm_2d);
    RUN_TEST(test_rmsnorm_basic);
    printf("\n");

    /* MatMul 测试 */
    printf("[MatMul Tests]\n");
    RUN_TEST(test_matmul_2d);
    RUN_TEST(test_matmul_3d);
    printf("\n");

    TEST_SUMMARY();
    return TEST_RETURN();
}
