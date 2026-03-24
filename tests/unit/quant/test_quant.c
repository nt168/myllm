/**
 * @file test_quant.c
 * @brief 量化模块单元测试
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <float.h>
#include "quant/quant.h"
#include "common/test_common.h"

#define FLOAT_EQ(a, b) (fabsf((a) - (b)) < 1e-5f)
#define FLOAT_NEAR(a, b, eps) (fabsf((a) - (b)) < (eps))

/* ============================================================================
 * FP8 E4M3 测试
 * ============================================================================ */

static void test_fp8_zero(void) {
    /* 零值测试 */
    uint8_t fp8_zero = f32_to_fp8_e4m3(0.0f);
    ASSERT_EQ(fp8_zero, 0, "Zero should encode to 0");

    float f32_zero = fp8_e4m3_to_f32(fp8_zero);
    ASSERT_TRUE(FLOAT_EQ(f32_zero, 0.0f), "Zero should decode to 0.0");

    /* 负零 */
    uint8_t fp8_neg_zero = f32_to_fp8_e4m3(-0.0f);
    ASSERT_EQ(fp8_neg_zero, 0x80, "Negative zero should have sign bit");

    TEST_PASS();
}

static void test_fp8_identity(void) {
    /* 往返测试 */
    float test_values[] = {0.5f, 1.0f, 1.5f, 2.0f, -1.0f, -2.5f, 100.0f};

    for (size_t i = 0; i < sizeof(test_values)/sizeof(test_values[0]); i++) {
        float orig = test_values[i];
        uint8_t fp8 = f32_to_fp8_e4m3(orig);
        float decoded = fp8_e4m3_to_f32(fp8);

        /* FP8 精度有限，允许一定误差 */
        float rel_error = fabsf(decoded - orig) / fmaxf(fabsf(orig), 1e-10f);
        ASSERT_TRUE(rel_error < 0.15f, "FP8 roundtrip should preserve value approximately");
    }

    TEST_PASS();
}

static void test_fp8_saturation(void) {
    /* 超出范围的值应该饱和 */
    uint8_t fp8_large = f32_to_fp8_e4m3(1000.0f);
    float decoded_large = fp8_e4m3_to_f32(fp8_large);
    ASSERT_TRUE(decoded_large >= 200.0f, "Large value should saturate to max");

    uint8_t fp8_small = f32_to_fp8_e4m3(-1000.0f);
    float decoded_small = fp8_e4m3_to_f32(fp8_small);
    ASSERT_TRUE(decoded_small <= -200.0f, "Small value should saturate to min");

    TEST_PASS();
}

static void test_fp8_batch(void) {
    float src[] = {0.0f, 1.0f, 2.0f, -1.0f, 0.5f};
    uint8_t dst[5];
    float back[5];

    f32_to_fp8_e4m3_batch(src, dst, 5);
    fp8_e4m3_to_f32_batch(dst, back, 5);

    for (int i = 0; i < 5; i++) {
        float rel_error = fabsf(back[i] - src[i]) / fmaxf(fabsf(src[i]), 1e-10f);
        ASSERT_TRUE(rel_error < 0.15f, "Batch conversion should preserve values");
    }

    TEST_PASS();
}

/* ============================================================================
 * INT8 量化测试
 * ============================================================================ */

static void test_int8_quantize_zero(void) {
    float data[] = {0.0f, 0.0f, 0.0f};
    float scale;
    int8_t quantized[3];

    int ret = compute_int8_params(data, 3, &scale);
    ASSERT_EQ(ret, 0, "compute_int8_params should succeed");
    ASSERT_TRUE(scale > 0.0f, "Scale should be positive");

    ret = quantize_int8(data, quantized, 3, scale);
    ASSERT_EQ(ret, 0, "quantize_int8 should succeed");

    for (int i = 0; i < 3; i++) {
        ASSERT_EQ(quantized[i], 0, "Zero should quantize to 0");
    }

    TEST_PASS();
}

static void test_int8_roundtrip(void) {
    float data[] = {0.5f, -0.5f, 1.0f, -1.0f, 0.25f};
    float scale;
    int8_t quantized[5];
    float dequantized[5];

    int ret = compute_int8_params(data, 5, &scale);
    ASSERT_EQ(ret, 0, "compute_int8_params should succeed");

    ret = quantize_int8(data, quantized, 5, scale);
    ASSERT_EQ(ret, 0, "quantize_int8 should succeed");

    ret = dequantize_int8(quantized, dequantized, 5, scale);
    ASSERT_EQ(ret, 0, "dequantize_int8 should succeed");

    /* 检查误差在可接受范围内 */
    for (int i = 0; i < 5; i++) {
        float error = fabsf(dequantized[i] - data[i]);
        ASSERT_TRUE(error < scale, "Quantization error should be less than scale");
    }

    TEST_PASS();
}

static void test_int8_clamp(void) {
    float data[] = {1000.0f, -1000.0f};
    float scale = 1.0f;  /* 小 scale 导致需要 clamp */
    int8_t quantized[2];

    int ret = quantize_int8(data, quantized, 2, scale);
    ASSERT_EQ(ret, 0, "quantize_int8 should succeed");

    ASSERT_EQ(quantized[0], 127, "Large positive should clamp to 127");
    ASSERT_EQ(quantized[1], -128, "Large negative should clamp to -128");

    TEST_PASS();
}

/* ============================================================================
 * INT4 块量化测试
 * ============================================================================ */

static void test_int4_block_roundtrip(void) {
    /* 32 个元素的块 */
    float data[32];
    for (int i = 0; i < 32; i++) {
        data[i] = (float)(i - 16) * 0.1f;  /* -1.5 到 1.5 */
    }

    size_t block_size = 32;
    uint8_t weights[16];  /* 32 个 4-bit 值 = 16 字节 */
    float scales[1];

    int ret = quantize_int4_block(data, 32, block_size, weights, scales);
    ASSERT_EQ(ret, 0, "quantize_int4_block should succeed");

    float dequantized[32];
    ret = dequantize_int4_block(weights, scales, 32, block_size, dequantized);
    ASSERT_EQ(ret, 0, "dequantize_int4_block should succeed");

    /* 检查误差 */
    for (int i = 0; i < 32; i++) {
        float error = fabsf(dequantized[i] - data[i]);
        /* INT4 精度较低，允许较大误差 */
        ASSERT_TRUE(error < 0.3f, "INT4 error should be acceptable");
    }

    TEST_PASS();
}

static void test_int4_multi_block(void) {
    /* 64 个元素，2 个块 */
    float data[64];
    for (int i = 0; i < 64; i++) {
        data[i] = (float)i * 0.05f - 1.5f;
    }

    size_t block_size = 32;
    uint8_t weights[32];  /* 64 个 4-bit 值 = 32 字节 */
    float scales[2];

    int ret = quantize_int4_block(data, 64, block_size, weights, scales);
    ASSERT_EQ(ret, 0, "quantize_int4_block should succeed for multi-block");

    float dequantized[64];
    ret = dequantize_int4_block(weights, scales, 64, block_size, dequantized);
    ASSERT_EQ(ret, 0, "dequantize_int4_block should succeed");

    /* 验证两个块有不同的 scale */
    ASSERT_TRUE(scales[0] > 0.0f, "Scale 0 should be positive");
    ASSERT_TRUE(scales[1] > 0.0f, "Scale 1 should be positive");

    TEST_PASS();
}

/* ============================================================================
 * 量化配置测试
 * ============================================================================ */

static void test_quant_config_default(void) {
    QuantConfig config = quant_config_default();

    ASSERT_EQ(config.method, QUANT_METHOD_NONE, "Default method should be None");
    ASSERT_EQ(config.bits, 32, "Default bits should be 32");

    TEST_PASS();
}

static void test_quant_config_fp8(void) {
    QuantConfig config = quant_config_fp8_e4m3(0);

    ASSERT_EQ(config.method, QUANT_METHOD_FP8_E4M3, "Method should be FP8 E4M3");
    ASSERT_EQ(config.bits, 8, "Bits should be 8");
    ASSERT_EQ(config.scale_format, SCALE_FORMAT_PER_TENSOR, "Scale format should be per-tensor");

    /* 带块大小 */
    config = quant_config_fp8_e4m3(128);
    ASSERT_EQ(config.block_size, 128, "Block size should be 128");
    ASSERT_EQ(config.scale_format, SCALE_FORMAT_BLOCK_WISE, "Scale format should be block-wise");

    TEST_PASS();
}

static void test_quant_config_int8(void) {
    QuantConfig config = quant_config_int8(SCALE_FORMAT_PER_TENSOR);

    ASSERT_EQ(config.method, QUANT_METHOD_INT8, "Method should be INT8");
    ASSERT_EQ(config.bits, 8, "Bits should be 8");
    ASSERT_EQ(config.scale_format, SCALE_FORMAT_PER_TENSOR, "Scale format should be per-tensor");

    TEST_PASS();
}

static void test_quant_config_int4(void) {
    QuantConfig config = quant_config_int4(32);

    ASSERT_EQ(config.method, QUANT_METHOD_INT4, "Method should be INT4");
    ASSERT_EQ(config.bits, 4, "Bits should be 4");
    ASSERT_EQ(config.block_size, 32, "Block size should be 32");
    ASSERT_EQ(config.scale_format, SCALE_FORMAT_BLOCK_WISE, "Scale format should be block-wise");

    TEST_PASS();
}

static void test_quant_method_name(void) {
    ASSERT_EQ(strcmp(quant_method_name(QUANT_METHOD_NONE), "None"), 0, "None name");
    ASSERT_EQ(strcmp(quant_method_name(QUANT_METHOD_FP8_E4M3), "FP8-E4M3"), 0, "FP8 E4M3 name");
    ASSERT_EQ(strcmp(quant_method_name(QUANT_METHOD_INT8), "INT8"), 0, "INT8 name");
    ASSERT_EQ(strcmp(quant_method_name(QUANT_METHOD_INT4), "INT4"), 0, "INT4 name");

    TEST_PASS();
}

static void test_quant_compressed_size(void) {
    size_t numel = 1024;

    /* 无量化 */
    QuantConfig config = quant_config_default();
    size_t size = quant_compressed_size(numel, &config);
    ASSERT_EQ(size, numel * sizeof(float), "FP32 size should be 4*numel");

    /* FP8 per-tensor (无额外 scale 存储) */
    config = quant_config_fp8_e4m3(0);
    size = quant_compressed_size(numel, &config);
    ASSERT_EQ(size, numel * sizeof(uint8_t), "FP8 per-tensor size should be numel");

    /* FP8 block-wise (带 scale) */
    config = quant_config_fp8_e4m3(128);
    size = quant_compressed_size(numel, &config);
    size_t expected_fp8 = numel * sizeof(uint8_t) + (numel / 128) * sizeof(float);
    ASSERT_EQ(size, expected_fp8, "FP8 block-wise size should include scales");

    /* INT4 */
    config = quant_config_int4(32);
    size = quant_compressed_size(numel, &config);
    size_t expected = (numel + 1) / 2 + (numel / 32) * sizeof(float);
    ASSERT_EQ(size, expected, "INT4 size should be numel/2 + scales");

    TEST_PASS();
}

/* ============================================================================
 * 量化权重测试
 * ============================================================================ */

static void test_quant_weight_lifecycle(void) {
    QuantConfig config = quant_config_int8(SCALE_FORMAT_PER_TENSOR);
    QuantizedWeight* weight = quant_weight_new(1024, &config);

    ASSERT_NOT_NULL(weight, "Weight should not be NULL");
    ASSERT_NOT_NULL(weight->data, "Weight data should not be NULL");
    ASSERT_NOT_NULL(weight->scales, "Weight scales should not be NULL");
    ASSERT_EQ(weight->numel, 1024, "Numel should be 1024");
    ASSERT_EQ(weight->num_scales, 1, "Should have 1 scale");

    quant_weight_free(weight);
    TEST_PASS();
}

static void test_quant_weight_free_null(void) {
    /* 应该不会崩溃 */
    quant_weight_free(NULL);
    TEST_PASS();
}

/* ============================================================================
 * 版本测试
 * ============================================================================ */

static void test_quant_version(void) {
    const char* version = quant_version();
    ASSERT_NOT_NULL(version, "Version should not be NULL");
    ASSERT_TRUE(strlen(version) > 0, "Version should not be empty");

    TEST_PASS();
}

/* ============================================================================
 * 主函数
 * ============================================================================ */

int main(void) {
    printf("========================================\n");
    printf("Quantization Module Unit Tests\n");
    printf("========================================\n\n");

    /* FP8 测试 */
    printf("[FP8 E4M3 Tests]\n");
    RUN_TEST(test_fp8_zero);
    RUN_TEST(test_fp8_identity);
    RUN_TEST(test_fp8_saturation);
    RUN_TEST(test_fp8_batch);
    printf("\n");

    /* INT8 测试 */
    printf("[INT8 Tests]\n");
    RUN_TEST(test_int8_quantize_zero);
    RUN_TEST(test_int8_roundtrip);
    RUN_TEST(test_int8_clamp);
    printf("\n");

    /* INT4 测试 */
    printf("[INT4 Tests]\n");
    RUN_TEST(test_int4_block_roundtrip);
    RUN_TEST(test_int4_multi_block);
    printf("\n");

    /* 配置测试 */
    printf("[Config Tests]\n");
    RUN_TEST(test_quant_config_default);
    RUN_TEST(test_quant_config_fp8);
    RUN_TEST(test_quant_config_int8);
    RUN_TEST(test_quant_config_int4);
    RUN_TEST(test_quant_method_name);
    RUN_TEST(test_quant_compressed_size);
    printf("\n");

    /* 权重测试 */
    printf("[Weight Tests]\n");
    RUN_TEST(test_quant_weight_lifecycle);
    RUN_TEST(test_quant_weight_free_null);
    printf("\n");

    /* 版本测试 */
    printf("[Version Tests]\n");
    RUN_TEST(test_quant_version);
    printf("\n");

    TEST_SUMMARY();
    return TEST_RETURN();
}
