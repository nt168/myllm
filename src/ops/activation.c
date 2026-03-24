/**
 * @file activation.c
 * @brief 激活函数实现
 */

#include "ops/activation.h"
#include <stdlib.h>
#include <math.h>

/* GELU 常量 */
#define SQRT_2_OVER_PI 0.7978846f
#define GELU_COEFF 0.044715f

/* ============================================================================
 * 激活函数实现
 * ============================================================================ */

/**
 * @brief GELU 激活函数 (近似版本)
 *
 * 公式: 0.5 * x * (1 + tanh(sqrt(2/π) * (x + 0.044715 * x³)))
 */
Tensor* ops_gelu(const Tensor* input) {
    if (!input) {
        return NULL;
    }

    size_t numel = tensor_numel(input);
    float* output_data = (float*)malloc(numel * sizeof(float));
    if (!output_data) {
        return NULL;
    }

    /* 计算每个元素 */
    for (size_t i = 0; i < numel; i++) {
        float x = tensor_get_f32(input, i);
        float x3 = x * x * x;
        float inner = SQRT_2_OVER_PI * (x + GELU_COEFF * x3);
        output_data[i] = 0.5f * x * (1.0f + tanhf(inner));
    }

    /* 创建输出张量 */
    const Shape* shape = tensor_shape(input);
    Tensor* result = tensor_from_f32(output_data, shape->dims, shape->ndim, input->dtype);
    free(output_data);

    return result;
}

/**
 * @brief SiLU (Swish) 激活函数
 *
 * 公式: x * sigmoid(x) = x / (1 + exp(-x))
 */
Tensor* ops_silu(const Tensor* input) {
    if (!input) {
        return NULL;
    }

    size_t numel = tensor_numel(input);
    float* output_data = (float*)malloc(numel * sizeof(float));
    if (!output_data) {
        return NULL;
    }

    for (size_t i = 0; i < numel; i++) {
        float x = tensor_get_f32(input, i);
        float sigmoid = 1.0f / (1.0f + expf(-x));
        output_data[i] = x * sigmoid;
    }

    const Shape* shape = tensor_shape(input);
    Tensor* result = tensor_from_f32(output_data, shape->dims, shape->ndim, input->dtype);
    free(output_data);

    return result;
}

/**
 * @brief Sigmoid 激活函数
 *
 * 公式: 1 / (1 + exp(-x))
 */
Tensor* ops_sigmoid(const Tensor* input) {
    if (!input) {
        return NULL;
    }

    size_t numel = tensor_numel(input);
    float* output_data = (float*)malloc(numel * sizeof(float));
    if (!output_data) {
        return NULL;
    }

    for (size_t i = 0; i < numel; i++) {
        float x = tensor_get_f32(input, i);
        output_data[i] = 1.0f / (1.0f + expf(-x));
    }

    const Shape* shape = tensor_shape(input);
    Tensor* result = tensor_from_f32(output_data, shape->dims, shape->ndim, input->dtype);
    free(output_data);

    return result;
}

/**
 * @brief ReLU 激活函数
 *
 * 公式: max(0, x)
 */
Tensor* ops_relu(const Tensor* input) {
    if (!input) {
        return NULL;
    }

    size_t numel = tensor_numel(input);
    float* output_data = (float*)malloc(numel * sizeof(float));
    if (!output_data) {
        return NULL;
    }

    for (size_t i = 0; i < numel; i++) {
        float x = tensor_get_f32(input, i);
        output_data[i] = x > 0.0f ? x : 0.0f;
    }

    const Shape* shape = tensor_shape(input);
    Tensor* result = tensor_from_f32(output_data, shape->dims, shape->ndim, input->dtype);
    free(output_data);

    return result;
}
