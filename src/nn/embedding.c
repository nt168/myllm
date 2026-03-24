/**
 * @file embedding.c
 * @brief 词嵌入层实现
 */

#include "nn/embedding.h"
#include <stdlib.h>
#include <stdio.h>
#include <string.h>

NN_Embedding* nn_embedding_new(size_t vocab_size, size_t hidden_dim, DType dtype) {
    NN_Embedding* embedding = (NN_Embedding*)malloc(sizeof(NN_Embedding));
    if (!embedding) {
        return NULL;
    }

    embedding->vocab_size = vocab_size;
    embedding->hidden_dim = hidden_dim;

    /* 创建权重张量 [vocab_size, hidden_dim] */
    size_t weight_dims[2] = { vocab_size, hidden_dim };
    embedding->weight = tensor_zeros(weight_dims, 2, dtype);
    if (!embedding->weight) {
        free(embedding);
        return NULL;
    }

    return embedding;
}

NN_Embedding* nn_embedding_from_weights(Tensor* weight) {
    if (!weight) {
        return NULL;
    }

    const Shape* weight_shape = tensor_shape(weight);
    if (weight_shape->ndim != 2) {
        fprintf(stderr, "NN_Embedding: weight must be 2D, got %zuD\n", weight_shape->ndim);
        return NULL;
    }

    NN_Embedding* embedding = (NN_Embedding*)malloc(sizeof(NN_Embedding));
    if (!embedding) {
        return NULL;
    }

    embedding->vocab_size = weight_shape->dims[0];
    embedding->hidden_dim = weight_shape->dims[1];
    embedding->weight = weight;

    return embedding;
}

void nn_embedding_free(NN_Embedding* embedding) {
    if (!embedding) {
        return;
    }

    if (embedding->weight) {
        tensor_free(embedding->weight);
    }
    free(embedding);
}

Tensor* nn_embedding_forward(NN_Embedding* embedding, const int32_t* token_ids, size_t seq_len) {
    if (!embedding || !token_ids) {
        return NULL;
    }

    /* 验证 token IDs */
    for (size_t i = 0; i < seq_len; i++) {
        if (token_ids[i] < 0) {
            fprintf(stderr, "NN_Embedding: negative token ID %d\n", token_ids[i]);
            return NULL;
        }
        if ((size_t)token_ids[i] >= embedding->vocab_size) {
            fprintf(stderr, "NN_Embedding: token ID %d out of bounds (vocab_size=%zu)\n",
                    token_ids[i], embedding->vocab_size);
            return NULL;
        }
    }

    /* 创建输出张量 [seq_len, hidden_dim] */
    size_t output_dims[2] = { seq_len, embedding->hidden_dim };
    Tensor* output = tensor_zeros(output_dims, 2, tensor_dtype(embedding->weight));
    if (!output) {
        return NULL;
    }

    /* 逐个 token 查找嵌入向量 */
    for (size_t i = 0; i < seq_len; i++) {
        size_t token_id = (size_t)token_ids[i];

        /* 从权重中复制对应的行 */
        for (size_t j = 0; j < embedding->hidden_dim; j++) {
            float val = tensor_get_f32(embedding->weight, token_id * embedding->hidden_dim + j);
            tensor_set_f32(output, i * embedding->hidden_dim + j, val);
        }
    }

    return output;
}

Tensor* nn_embedding_forward_tensor(NN_Embedding* embedding, const Tensor* input) {
    if (!embedding || !input) {
        return NULL;
    }

    /* 输入必须是 I32 类型 */
    if (tensor_dtype(input) != DTYPE_I32) {
        fprintf(stderr, "NN_Embedding: input must be I32 type\n");
        return NULL;
    }

    const Shape* input_shape = tensor_shape(input);
    size_t input_ndim = input_shape->ndim;

    if (input_ndim == 1) {
        /* 1D 输入: [seq_len] */
        size_t seq_len = input_shape->dims[0];

        /* 提取 token IDs */
        int32_t* token_ids = (int32_t*)malloc(seq_len * sizeof(int32_t));
        if (!token_ids) {
            return NULL;
        }

        for (size_t i = 0; i < seq_len; i++) {
            token_ids[i] = (int32_t)tensor_get_f32(input, i);
        }

        Tensor* output = nn_embedding_forward(embedding, token_ids, seq_len);
        free(token_ids);

        return output;

    } else if (input_ndim == 2) {
        /* 2D 输入: [batch, seq_len] */
        size_t batch = input_shape->dims[0];
        size_t seq_len = input_shape->dims[1];

        /* 创建输出张量 [batch, seq_len, hidden_dim] */
        size_t output_dims[3] = { batch, seq_len, embedding->hidden_dim };
        Tensor* output = tensor_zeros(output_dims, 3, tensor_dtype(embedding->weight));
        if (!output) {
            return NULL;
        }

        /* 处理每个 batch */
        for (size_t b = 0; b < batch; b++) {
            for (size_t s = 0; s < seq_len; s++) {
                int32_t token_id = (int32_t)tensor_get_f32(input, b * seq_len + s);

                if (token_id < 0 || (size_t)token_id >= embedding->vocab_size) {
                    fprintf(stderr, "NN_Embedding: token ID %d out of bounds\n", token_id);
                    tensor_free(output);
                    return NULL;
                }

                /* 复制嵌入向量 */
                for (size_t h = 0; h < embedding->hidden_dim; h++) {
                    float val = tensor_get_f32(embedding->weight,
                                               (size_t)token_id * embedding->hidden_dim + h);
                    tensor_set_f32(output, (b * seq_len + s) * embedding->hidden_dim + h, val);
                }
            }
        }

        return output;

    } else {
        fprintf(stderr, "NN_Embedding: input must be 1D or 2D, got %zuD\n", input_ndim);
        return NULL;
    }
}
