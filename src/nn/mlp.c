/**
 * @file mlp.c
 * @brief 前馈网络 (SwiGLU) 实现
 */

#include "nn/mlp.h"
#include "ops/ops.h"
#include <stdlib.h>
#include <stdio.h>

NN_MLP* nn_mlp_new(size_t hidden_dim, size_t intermediate_dim, DType dtype) {
    NN_MLP* mlp = (NN_MLP*)malloc(sizeof(NN_MLP));
    if (!mlp) {
        return NULL;
    }

    mlp->hidden_dim = hidden_dim;
    mlp->intermediate_dim = intermediate_dim;

    /* 创建三个线性层 (无偏置) */
    mlp->gate_proj = nn_linear_new(hidden_dim, intermediate_dim, false, dtype);
    if (!mlp->gate_proj) {
        free(mlp);
        return NULL;
    }

    mlp->up_proj = nn_linear_new(hidden_dim, intermediate_dim, false, dtype);
    if (!mlp->up_proj) {
        nn_linear_free(mlp->gate_proj);
        free(mlp);
        return NULL;
    }

    mlp->down_proj = nn_linear_new(intermediate_dim, hidden_dim, false, dtype);
    if (!mlp->down_proj) {
        nn_linear_free(mlp->gate_proj);
        nn_linear_free(mlp->up_proj);
        free(mlp);
        return NULL;
    }

    return mlp;
}

void nn_mlp_free(NN_MLP* mlp) {
    if (!mlp) {
        return;
    }

    if (mlp->gate_proj) {
        nn_linear_free(mlp->gate_proj);
    }
    if (mlp->up_proj) {
        nn_linear_free(mlp->up_proj);
    }
    if (mlp->down_proj) {
        nn_linear_free(mlp->down_proj);
    }
    free(mlp);
}

Tensor* nn_mlp_forward(NN_MLP* mlp, const Tensor* input) {
    if (!mlp || !input) {
        return NULL;
    }

    const Shape* input_shape = tensor_shape(input);
    size_t ndim = input_shape->ndim;

    /* 支持 1D, 2D, 3D 输入 */
    Tensor* input_2d = NULL;
    Tensor* input_to_use = NULL;
    size_t batch = 1;
    size_t seq_len = 1;
    size_t hidden = mlp->hidden_dim;

    if (ndim == 1) {
        /* 1D: [hidden_dim] */
        input_to_use = (Tensor*)input;
    } else if (ndim == 2) {
        /* 2D: [seq_len, hidden_dim] 或 [batch, hidden_dim] */
        input_to_use = (Tensor*)input;
        seq_len = input_shape->dims[0];
    } else if (ndim == 3) {
        /* 3D: [batch, seq_len, hidden_dim] -> reshape to [batch*seq_len, hidden_dim] */
        batch = input_shape->dims[0];
        seq_len = input_shape->dims[1];
        ssize_t new_dims[2] = { (ssize_t)(batch * seq_len), (ssize_t)hidden };
        input_2d = tensor_reshape(input, new_dims, 2);
        if (!input_2d) {
            return NULL;
        }
        input_to_use = input_2d;
    } else {
        fprintf(stderr, "NN_MLP: input must be 1D, 2D, or 3D, got %zuD\n", ndim);
        return NULL;
    }

    /* gate_proj: [N, hidden_dim] -> [N, intermediate_dim] */
    Tensor* gate = nn_linear_forward(mlp->gate_proj, input_to_use);
    if (!gate) {
        if (input_2d) tensor_free(input_2d);
        return NULL;
    }

    /* up_proj: [N, hidden_dim] -> [N, intermediate_dim] */
    Tensor* up = nn_linear_forward(mlp->up_proj, input_to_use);
    if (!up) {
        tensor_free(gate);
        if (input_2d) tensor_free(input_2d);
        return NULL;
    }

    /* 应用 SiLU 激活函数到 gate */
    Tensor* gate_activated = ops_silu(gate);
    tensor_free(gate);
    if (!gate_activated) {
        tensor_free(up);
        if (input_2d) tensor_free(input_2d);
        return NULL;
    }

    /* 逐元素乘法: gate_activated * up */
    Tensor* hidden_state = ops_mul(gate_activated, up);
    tensor_free(gate_activated);
    tensor_free(up);
    if (!hidden_state) {
        if (input_2d) tensor_free(input_2d);
        return NULL;
    }

    /* down_proj: [N, intermediate_dim] -> [N, hidden_dim] */
    Tensor* output = nn_linear_forward(mlp->down_proj, hidden_state);
    tensor_free(hidden_state);
    if (input_2d) tensor_free(input_2d);

    if (!output) {
        return NULL;
    }

    /* 如果输入是 3D，reshape 回原始形状 */
    if (ndim == 3) {
        ssize_t new_dims[3] = { (ssize_t)batch, (ssize_t)seq_len, (ssize_t)hidden };
        Tensor* output_3d = tensor_reshape(output, new_dims, 3);
        tensor_free(output);
        output = output_3d;
    }

    return output;
}
