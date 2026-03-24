/**
 * @file test_tensor.c
 * @brief 张量模块单元测试
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include "tensor.h"
#include "common/test_common.h"

/* ============================================================================
 * DType 测试
 * ============================================================================ */

static void test_dtype_size(void) {
    ASSERT_EQ(dtype_size(DTYPE_F16), 2, "F16 size should be 2");
    ASSERT_EQ(dtype_size(DTYPE_F32), 4, "F32 size should be 4");
    ASSERT_EQ(dtype_size(DTYPE_BF16), 2, "BF16 size should be 2");
    ASSERT_EQ(dtype_size(DTYPE_I32), 4, "I32 size should be 4");
    ASSERT_EQ(dtype_size(DTYPE_I64), 8, "I64 size should be 8");
    TEST_PASS();
}

static void test_dtype_is_float(void) {
    ASSERT_TRUE(dtype_is_float(DTYPE_F16), "F16 is float");
    ASSERT_TRUE(dtype_is_float(DTYPE_F32), "F32 is float");
    ASSERT_TRUE(dtype_is_float(DTYPE_BF16), "BF16 is float");
    ASSERT_FALSE(dtype_is_float(DTYPE_I32), "I32 is not float");
    ASSERT_FALSE(dtype_is_float(DTYPE_I64), "I64 is not float");
    TEST_PASS();
}

static void test_dtype_name(void) {
    ASSERT_TRUE(strcmp(dtype_name(DTYPE_F16), "f16") == 0, "F16 name");
    ASSERT_TRUE(strcmp(dtype_name(DTYPE_F32), "f32") == 0, "F32 name");
    ASSERT_TRUE(strcmp(dtype_name(DTYPE_BF16), "bf16") == 0, "BF16 name");
    ASSERT_TRUE(strcmp(dtype_name(DTYPE_I32), "i32") == 0, "I32 name");
    ASSERT_TRUE(strcmp(dtype_name(DTYPE_I64), "i64") == 0, "I64 name");
    TEST_PASS();
}

/* ============================================================================
 * Shape 测试
 * ============================================================================ */

static void test_shape_scalar(void) {
    Shape s = shape_scalar();
    ASSERT_EQ(s.ndim, 0, "Scalar ndim should be 0");
    ASSERT_TRUE(shape_is_scalar(&s), "Should be scalar");
    ASSERT_EQ(shape_numel(&s), 1, "Scalar numel should be 1");
    TEST_PASS();
}

static void test_shape_new(void) {
    size_t dims[] = {2, 3, 4};
    Shape s = shape_new(dims, 3);

    ASSERT_EQ(s.ndim, 3, "ndim should be 3");
    ASSERT_EQ(shape_dim(&s, 0), 2, "dim 0 should be 2");
    ASSERT_EQ(shape_dim(&s, 1), 3, "dim 1 should be 3");
    ASSERT_EQ(shape_dim(&s, 2), 4, "dim 2 should be 4");
    ASSERT_EQ(shape_numel(&s), 24, "numel should be 24");
    TEST_PASS();
}

static void test_shape_strides(void) {
    size_t dims[] = {2, 3, 4};
    Shape s = shape_new(dims, 3);
    size_t strides[3];
    shape_strides(&s, strides);

    ASSERT_EQ(strides[0], 12, "stride[0] should be 12");
    ASSERT_EQ(strides[1], 4, "stride[1] should be 4");
    ASSERT_EQ(strides[2], 1, "stride[2] should be 1");
    TEST_PASS();
}

static void test_shape_reshape(void) {
    size_t dims[] = {2, 3, 4};
    Shape s = shape_new(dims, 3);

    ssize_t new_dims[] = {6, 4};
    Shape result;
    ASSERT_EQ(shape_reshape(&s, new_dims, 2, &result), 0, "reshape should succeed");
    ASSERT_EQ(result.ndim, 2, "new ndim should be 2");
    ASSERT_EQ(result.dims[0], 6, "new dim 0 should be 6");
    ASSERT_EQ(result.dims[1], 4, "new dim 1 should be 4");
    TEST_PASS();
}

static void test_shape_reshape_infer(void) {
    size_t dims[] = {2, 3, 4};
    Shape s = shape_new(dims, 3);

    ssize_t new_dims[] = {-1, 4};
    Shape result;
    ASSERT_EQ(shape_reshape(&s, new_dims, 2, &result), 0, "reshape with -1 should succeed");
    ASSERT_EQ(result.dims[0], 6, "inferred dim should be 6");
    ASSERT_EQ(result.dims[1], 4, "dim 1 should be 4");
    TEST_PASS();
}

static void test_shape_squeeze(void) {
    size_t dims[] = {1, 2, 1, 3, 1};
    Shape s = shape_new(dims, 5);

    Shape result = shape_squeeze(&s, -1);  /* -1 表示移除所有大小为1的维度 */
    ASSERT_EQ(result.ndim, 2, "squeezed ndim should be 2");
    ASSERT_EQ(result.dims[0], 2, "squeezed dim 0 should be 2");
    ASSERT_EQ(result.dims[1], 3, "squeezed dim 1 should be 3");
    TEST_PASS();
}

static void test_shape_unsqueeze(void) {
    size_t dims[] = {2, 3};
    Shape s = shape_new(dims, 2);

    Shape result = shape_unsqueeze(&s, 0);
    ASSERT_EQ(result.ndim, 3, "unsqueezed ndim should be 3");
    ASSERT_EQ(result.dims[0], 1, "unsqueezed dim 0 should be 1");
    ASSERT_EQ(result.dims[1], 2, "unsqueezed dim 1 should be 2");
    ASSERT_EQ(result.dims[2], 3, "unsqueezed dim 2 should be 3");
    TEST_PASS();
}

static void test_shape_transpose(void) {
    size_t dims[] = {2, 3, 4};
    Shape s = shape_new(dims, 3);

    Shape result = shape_transpose(&s, 0, 2);
    ASSERT_EQ(result.dims[0], 4, "transposed dim 0 should be 4");
    ASSERT_EQ(result.dims[1], 3, "transposed dim 1 should be 3");
    ASSERT_EQ(result.dims[2], 2, "transposed dim 2 should be 2");
    TEST_PASS();
}

/* ============================================================================
 * Tensor 创建测试
 * ============================================================================ */

static void test_tensor_zeros(void) {
    size_t dims[] = {2, 3};
    Tensor* t = tensor_zeros(dims, 2, DTYPE_F32);

    ASSERT_NOT_NULL(t, "Tensor should not be NULL");
    ASSERT_EQ(tensor_ndim(t), 2, "ndim should be 2");
    ASSERT_EQ(tensor_numel(t), 6, "numel should be 6");

    for (size_t i = 0; i < 6; i++) {
        ASSERT_FLOAT_EQ(tensor_get_f32(t, i), 0.0f, 1e-6, "Element should be 0");
    }

    tensor_free(t);
    TEST_PASS();
}

static void test_tensor_ones(void) {
    size_t dims[] = {2, 3};
    Tensor* t = tensor_ones(dims, 2, DTYPE_F32);

    ASSERT_NOT_NULL(t, "Tensor should not be NULL");

    for (size_t i = 0; i < 6; i++) {
        ASSERT_FLOAT_EQ(tensor_get_f32(t, i), 1.0f, 1e-6, "Element should be 1");
    }

    tensor_free(t);
    TEST_PASS();
}

static void test_tensor_from_f32(void) {
    float data[] = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f};
    size_t dims[] = {2, 3};

    Tensor* t = tensor_from_f32(data, dims, 2, DTYPE_F32);
    ASSERT_NOT_NULL(t, "Tensor should not be NULL");

    for (size_t i = 0; i < 6; i++) {
        ASSERT_FLOAT_EQ(tensor_get_f32(t, i), data[i], 1e-6, "Element mismatch");
    }

    tensor_free(t);
    TEST_PASS();
}

static void test_tensor_from_i32(void) {
    int32_t data[] = {1, 2, 3, 4, 5, 6};
    size_t dims[] = {2, 3};

    Tensor* t = tensor_from_i32(data, dims, 2, DTYPE_I32);
    ASSERT_NOT_NULL(t, "Tensor should not be NULL");

    for (size_t i = 0; i < 6; i++) {
        ASSERT_FLOAT_EQ(tensor_get_f32(t, i), (float)data[i], 1e-6, "Element mismatch");
    }

    tensor_free(t);
    TEST_PASS();
}

/* ============================================================================
 * Tensor 操作测试
 * ============================================================================ */

static void test_tensor_set_get(void) {
    size_t dims[] = {2, 3};
    Tensor* t = tensor_zeros(dims, 2, DTYPE_F32);

    tensor_set_f32(t, 0, 1.5f);
    tensor_set_f32(t, 5, 3.5f);

    ASSERT_FLOAT_EQ(tensor_get_f32(t, 0), 1.5f, 1e-6, "Element 0 should be 1.5");
    ASSERT_FLOAT_EQ(tensor_get_f32(t, 5), 3.5f, 1e-6, "Element 5 should be 3.5");

    tensor_free(t);
    TEST_PASS();
}

static void test_tensor_clone(void) {
    float data[] = {1.0f, 2.0f, 3.0f, 4.0f};
    size_t dims[] = {2, 2};

    Tensor* t1 = tensor_from_f32(data, dims, 2, DTYPE_F32);
    Tensor* t2 = tensor_clone(t1);

    ASSERT_NOT_NULL(t2, "Clone should not be NULL");
    ASSERT_TRUE(tensor_equals(t1, t2, 1e-6), "Cloned tensor should equal original");

    /* 修改原始张量，克隆应该不受影响 */
    tensor_set_f32(t1, 0, 99.0f);
    ASSERT_FLOAT_EQ(tensor_get_f32(t2, 0), 1.0f, 1e-6, "Clone should be independent");

    tensor_free(t1);
    tensor_free(t2);
    TEST_PASS();
}

static void test_tensor_reshape(void) {
    float data[] = {1, 2, 3, 4, 5, 6};
    size_t dims[] = {2, 3};

    Tensor* t = tensor_from_f32(data, dims, 2, DTYPE_F32);

    ssize_t new_dims[] = {3, 2};
    Tensor* reshaped = tensor_reshape(t, new_dims, 2);

    ASSERT_NOT_NULL(reshaped, "Reshape should not be NULL");
    ASSERT_EQ(tensor_ndim(reshaped), 2, "Reshaped ndim should be 2");
    ASSERT_EQ(tensor_shape(reshaped)->dims[0], 3, "Reshaped dim 0 should be 3");
    ASSERT_EQ(tensor_shape(reshaped)->dims[1], 2, "Reshaped dim 1 should be 2");
    ASSERT_EQ(tensor_numel(reshaped), 6, "Reshaped numel should be 6");

    tensor_free(t);
    tensor_free(reshaped);
    TEST_PASS();
}

static void test_tensor_squeeze(void) {
    float data[] = {1, 2, 3};
    size_t dims[] = {1, 3, 1};

    Tensor* t = tensor_from_f32(data, dims, 3, DTYPE_F32);
    Tensor* squeezed = tensor_squeeze(t, -1);

    ASSERT_NOT_NULL(squeezed, "Squeeze should not be NULL");
    ASSERT_EQ(tensor_ndim(squeezed), 1, "Squeezed ndim should be 1");
    ASSERT_EQ(tensor_shape(squeezed)->dims[0], 3, "Squeezed dim should be 3");

    tensor_free(t);
    tensor_free(squeezed);
    TEST_PASS();
}

static void test_tensor_unsqueeze(void) {
    float data[] = {1, 2, 3};
    size_t dims[] = {3};

    Tensor* t = tensor_from_f32(data, dims, 1, DTYPE_F32);
    Tensor* unsqueezed = tensor_unsqueeze(t, 0);

    ASSERT_NOT_NULL(unsqueezed, "Unsqueeze should not be NULL");
    ASSERT_EQ(tensor_ndim(unsqueezed), 2, "Unsqueezed ndim should be 2");
    ASSERT_EQ(tensor_shape(unsqueezed)->dims[0], 1, "Unsqueezed dim 0 should be 1");
    ASSERT_EQ(tensor_shape(unsqueezed)->dims[1], 3, "Unsqueezed dim 1 should be 3");

    tensor_free(t);
    tensor_free(unsqueezed);
    TEST_PASS();
}

static void test_tensor_transpose(void) {
    float data[] = {1, 2, 3, 4, 5, 6};
    size_t dims[] = {2, 3};

    Tensor* t = tensor_from_f32(data, dims, 2, DTYPE_F32);
    Tensor* transposed = tensor_transpose(t, 0, 1);

    ASSERT_NOT_NULL(transposed, "Transpose should not be NULL");
    ASSERT_EQ(tensor_ndim(transposed), 2, "Transposed ndim should be 2");
    ASSERT_EQ(tensor_shape(transposed)->dims[0], 3, "Transposed dim 0 should be 3");
    ASSERT_EQ(tensor_shape(transposed)->dims[1], 2, "Transposed dim 1 should be 2");

    /* 验证数据: 原始 [1,2,3,4,5,6] -> 转置 [[1,4],[2,5],[3,6]] */
    ASSERT_FLOAT_EQ(tensor_get_f32(transposed, 0), 1.0f, 1e-6, "Transposed element 0");
    ASSERT_FLOAT_EQ(tensor_get_f32(transposed, 1), 4.0f, 1e-6, "Transposed element 1");
    ASSERT_FLOAT_EQ(tensor_get_f32(transposed, 2), 2.0f, 1e-6, "Transposed element 2");

    tensor_free(t);
    tensor_free(transposed);
    TEST_PASS();
}

static void test_tensor_is_contiguous(void) {
    float data[] = {1, 2, 3, 4, 5, 6};
    size_t dims[] = {2, 3};

    Tensor* t = tensor_from_f32(data, dims, 2, DTYPE_F32);
    ASSERT_TRUE(tensor_is_contiguous(t), "New tensor should be contiguous");

    Tensor* transposed = tensor_transpose(t, 0, 1);
    ASSERT_FALSE(tensor_is_contiguous(transposed), "Transposed tensor should not be contiguous");

    Tensor* contiguous = tensor_contiguous(transposed);
    ASSERT_TRUE(tensor_is_contiguous(contiguous), "Contiguous copy should be contiguous");

    tensor_free(t);
    tensor_free(transposed);
    tensor_free(contiguous);
    TEST_PASS();
}

/* ============================================================================
 * 类型转换测试
 * ============================================================================ */

static void test_tensor_f16_conversion(void) {
    float data[] = {1.0f, 2.0f, 0.5f, -1.0f};
    size_t dims[] = {4};

    Tensor* t = tensor_from_f32(data, dims, 1, DTYPE_F16);
    ASSERT_NOT_NULL(t, "F16 tensor should not be NULL");
    ASSERT_EQ(tensor_dtype(t), DTYPE_F16, "Dtype should be F16");

    /* 验证转换精度 */
    for (size_t i = 0; i < 4; i++) {
        float recovered = tensor_get_f32(t, i);
        ASSERT_FLOAT_EQ(recovered, data[i], 0.01f, "F16 conversion should be accurate");
    }

    tensor_free(t);
    TEST_PASS();
}

static void test_tensor_i32_conversion(void) {
    int32_t data[] = {1, 2, 3, 4, 5, 6};
    size_t dims[] = {2, 3};

    Tensor* t = tensor_from_i32(data, dims, 2, DTYPE_I32);
    ASSERT_NOT_NULL(t, "I32 tensor should not be NULL");
    ASSERT_EQ(tensor_dtype(t), DTYPE_I32, "Dtype should be I32");

    for (size_t i = 0; i < 6; i++) {
        float val = tensor_get_f32(t, i);
        ASSERT_FLOAT_EQ(val, (float)data[i], 1e-6, "I32 conversion");
    }

    tensor_free(t);
    TEST_PASS();
}

/* ============================================================================
 * 主函数
 * ============================================================================ */

int main(void) {
    printf("========================================\n");
    printf("Tensor Module Unit Tests\n");
    printf("========================================\n\n");

    /* DType 测试 */
    printf("[DType Tests]\n");
    RUN_TEST(test_dtype_size);
    RUN_TEST(test_dtype_is_float);
    RUN_TEST(test_dtype_name);
    printf("\n");

    /* Shape 测试 */
    printf("[Shape Tests]\n");
    RUN_TEST(test_shape_scalar);
    RUN_TEST(test_shape_new);
    RUN_TEST(test_shape_strides);
    RUN_TEST(test_shape_reshape);
    RUN_TEST(test_shape_reshape_infer);
    RUN_TEST(test_shape_squeeze);
    RUN_TEST(test_shape_unsqueeze);
    RUN_TEST(test_shape_transpose);
    printf("\n");

    /* Tensor 创建测试 */
    printf("[Tensor Creation Tests]\n");
    RUN_TEST(test_tensor_zeros);
    RUN_TEST(test_tensor_ones);
    RUN_TEST(test_tensor_from_f32);
    RUN_TEST(test_tensor_from_i32);
    printf("\n");

    /* Tensor 操作测试 */
    printf("[Tensor Operation Tests]\n");
    RUN_TEST(test_tensor_set_get);
    RUN_TEST(test_tensor_clone);
    RUN_TEST(test_tensor_reshape);
    RUN_TEST(test_tensor_squeeze);
    RUN_TEST(test_tensor_unsqueeze);
    RUN_TEST(test_tensor_transpose);
    RUN_TEST(test_tensor_is_contiguous);
    printf("\n");

    /* 类型转换测试 */
    printf("[Type Conversion Tests]\n");
    RUN_TEST(test_tensor_f16_conversion);
    RUN_TEST(test_tensor_i32_conversion);
    printf("\n");

    TEST_SUMMARY();
    return TEST_RETURN();
}
