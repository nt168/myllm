/**
 * @file rmsnorm.c
 * @brief RMS 归一化层实现
 */

#include "nn/rmsnorm.h"
#include "ops/ops.h"
#include <stdlib.h>
#include <stdio.h>

NN_RMSNorm* nn_rmsnorm_new(size_t normalized_shape, float eps, DType dtype) {
    NN_RMSNorm* rmsnorm = (NN_RMSNorm*)malloc(sizeof(NN_RMSNorm));
    if (!rmsnorm) {
        return NULL;
    }

    rmsnorm->normalized_shape = normalized_shape;
    rmsnorm->eps = eps;

    /* 创建权重张量 [normalized_shape]，初始化为 1 */
    size_t weight_dims[1] = { normalized_shape };
    rmsnorm->weight = tensor_ones(weight_dims, 1, dtype);
    if (!rmsnorm->weight) {
        free(rmsnorm);
        return NULL;
    }

    return rmsnorm;
}

NN_RMSNorm* nn_rmsnorm_from_weights(Tensor* weight, float eps) {
    if (!weight) {
        return NULL;
    }

    const Shape* weight_shape = tensor_shape(weight);
    if (weight_shape->ndim != 1) {
        fprintf(stderr, "NN_RMSNorm: weight must be 1D, got %zuD\n", weight_shape->ndim);
        return NULL;
    }

    NN_RMSNorm* rmsnorm = (NN_RMSNorm*)malloc(sizeof(NN_RMSNorm));
    if (!rmsnorm) {
        return NULL;
    }

    rmsnorm->normalized_shape = weight_shape->dims[0];
    rmsnorm->eps = eps;
    rmsnorm->weight = weight;

    return rmsnorm;
}

void nn_rmsnorm_free(NN_RMSNorm* rmsnorm) {
    if (!rmsnorm) {
        return;
    }

    if (rmsnorm->weight) {
        tensor_free(rmsnorm->weight);
    }
    free(rmsnorm);
}

Tensor* nn_rmsnorm_forward(NN_RMSNorm* rmsnorm, const Tensor* input) {
    if (!rmsnorm || !input) {
        return NULL;
    }

    const Shape* input_shape = tensor_shape(input);
    size_t ndim = input_shape->ndim;

    if (ndim == 0) {
        fprintf(stderr, "NN_RMSNorm: input must have at least 1 dimension\n");
        return NULL;
    }

    /* 检查最后一维是否匹配 normalized_shape */
    size_t last_dim = input_shape->dims[ndim - 1];
    if (last_dim != rmsnorm->normalized_shape) {
        fprintf(stderr, "NN_RMSNorm: last dim %zu != normalized_shape %zu\n",
                last_dim, rmsnorm->normalized_shape);
        return NULL;
    }

    /* 获取权重数据 */
    float* weight_data = (float*)malloc(rmsnorm->normalized_shape * sizeof(float));
    if (!weight_data) {
        return NULL;
    }

    for (size_t i = 0; i < rmsnorm->normalized_shape; i++) {
        weight_data[i] = tensor_get_f32(rmsnorm->weight, i);
    }

    /* 使用 ops 模块的 rmsnorm */
    Tensor* output = ops_rmsnorm(input, rmsnorm->normalized_shape, weight_data, rmsnorm->eps);

    free(weight_data);
    return output;
}
