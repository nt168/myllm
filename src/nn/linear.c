/**
 * @file linear.c
 * @brief 线性层实现
 */

#include "nn/linear.h"
#include "ops/ops.h"
#include <stdlib.h>
#include <stdio.h>
#include <string.h>

NN_Linear* nn_linear_new(size_t in_features, size_t out_features, bool use_bias, DType dtype) {
    NN_Linear* linear = (NN_Linear*)malloc(sizeof(NN_Linear));
    if (!linear) {
        return NULL;
    }

    linear->in_features = in_features;
    linear->out_features = out_features;

    /* 创建权重张量 [out_features, in_features] */
    size_t weight_dims[2] = { out_features, in_features };
    linear->weight = tensor_zeros(weight_dims, 2, dtype);
    if (!linear->weight) {
        free(linear);
        return NULL;
    }

    /* 创建偏置张量 (可选) */
    if (use_bias) {
        size_t bias_dims[1] = { out_features };
        linear->bias = tensor_zeros(bias_dims, 1, dtype);
        if (!linear->bias) {
            tensor_free(linear->weight);
            free(linear);
            return NULL;
        }
    } else {
        linear->bias = NULL;
    }

    return linear;
}

NN_Linear* nn_linear_from_weights(Tensor* weight, Tensor* bias) {
    if (!weight) {
        return NULL;
    }

    const Shape* weight_shape = tensor_shape(weight);
    if (weight_shape->ndim != 2) {
        fprintf(stderr, "NN_Linear: weight must be 2D, got %zuD\n", weight_shape->ndim);
        return NULL;
    }

    NN_Linear* linear = (NN_Linear*)malloc(sizeof(NN_Linear));
    if (!linear) {
        return NULL;
    }

    linear->out_features = weight_shape->dims[0];
    linear->in_features = weight_shape->dims[1];
    linear->weight = weight;

    /* 验证偏置形状 */
    if (bias) {
        const Shape* bias_shape = tensor_shape(bias);
        if (bias_shape->ndim != 1 || bias_shape->dims[0] != linear->out_features) {
            fprintf(stderr, "NN_Linear: bias shape mismatch, expected [%zu], got [%zu]\n",
                    linear->out_features, bias_shape->dims[0]);
            free(linear);
            return NULL;
        }
    }
    linear->bias = bias;

    return linear;
}

void nn_linear_free(NN_Linear* linear) {
    if (!linear) {
        return;
    }

    if (linear->weight) {
        tensor_free(linear->weight);
    }
    if (linear->bias) {
        tensor_free(linear->bias);
    }
    free(linear);
}

Tensor* nn_linear_forward(NN_Linear* linear, const Tensor* input) {
    if (!linear || !input) {
        return NULL;
    }

    const Shape* input_shape = tensor_shape(input);
    size_t input_ndim = input_shape->ndim;

    /* 支持 1D 和 2D 输入 */
    size_t in_dim;

    if (input_ndim == 1) {
        in_dim = input_shape->dims[0];
    } else if (input_ndim == 2) {
        in_dim = input_shape->dims[1];
    } else {
        fprintf(stderr, "NN_Linear: input must be 1D or 2D, got %zuD\n", input_ndim);
        return NULL;
    }

    if (in_dim != linear->in_features) {
        fprintf(stderr, "NN_Linear: input dimension %zu != expected %zu\n",
                in_dim, linear->in_features);
        return NULL;
    }

    /* 转置权重: [out_features, in_features] -> [in_features, out_features] */
    Tensor* weight_t = tensor_transpose(linear->weight, 0, 1);
    if (!weight_t) {
        return NULL;
    }

    /* 准备 2D 输入 */
    Tensor* input_2d = NULL;
    Tensor* input_to_use = NULL;

    if (input_ndim == 1) {
        /* 将 1D 输入 reshape 为 [1, in_features] */
        ssize_t new_dims[2] = { 1, (ssize_t)in_dim };
        input_2d = tensor_reshape(input, new_dims, 2);
        input_to_use = input_2d;
    } else {
        input_to_use = (Tensor*)input;
    }

    /* 矩阵乘法: [batch, in_features] @ [in_features, out_features] = [batch, out_features] */
    Tensor* output = ops_matmul(input_to_use, weight_t);

    if (input_2d) {
        tensor_free(input_2d);
    }
    tensor_free(weight_t);

    if (!output) {
        return NULL;
    }

    /* 添加偏置 */
    if (linear->bias) {
        Tensor* output_with_bias = ops_add(output, linear->bias);
        tensor_free(output);
        if (!output_with_bias) {
            return NULL;
        }
        output = output_with_bias;
    }

    /* 如果输入是 1D，将输出 reshape 回 [out_features] */
    if (input_ndim == 1) {
        ssize_t new_dims[1] = { (ssize_t)linear->out_features };
        Tensor* output_1d = tensor_reshape(output, new_dims, 1);
        tensor_free(output);
        output = output_1d;
    }

    return output;
}
