/**
 * @file quant.c
 * @brief 量化模块实现
 */

#include "quant/quant.h"
#include "tensor/shape.h"
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <float.h>

#define QUANT_VERSION "0.1.0"

const char* quant_version(void) {
    return QUANT_VERSION;
}

/* ============================================================================
 * 量化权重管理
 * ============================================================================ */

QuantizedWeight* quant_weight_new(size_t numel, const QuantConfig* config) {
    if (!config || numel == 0) {
        return NULL;
    }

    QuantizedWeight* weight = (QuantizedWeight*)calloc(1, sizeof(QuantizedWeight));
    if (!weight) {
        return NULL;
    }

    weight->numel = numel;
    weight->config = *config;

    /* 计算数据大小 */
    weight->data_size = quant_compressed_size(numel, config);

    /* 分配数据存储 */
    weight->data = (uint8_t*)calloc(1, weight->data_size);
    if (!weight->data) {
        free(weight);
        return NULL;
    }

    /* 计算缩放因子数量 */
    if (config->scale_format == SCALE_FORMAT_BLOCK_WISE && config->block_size > 0) {
        weight->num_scales = (numel + config->block_size - 1) / config->block_size;
    } else if (config->scale_format == SCALE_FORMAT_PER_CHANNEL) {
        weight->num_scales = 1;  /* 简化：实际应该按通道数 */
    } else {
        weight->num_scales = 1;  /* Per-tensor */
    }

    /* 分配缩放因子存储 */
    if (weight->num_scales > 0) {
        weight->scales = (float*)calloc(weight->num_scales, sizeof(float));
        if (!weight->scales) {
            free(weight->data);
            free(weight);
            return NULL;
        }
    }

    return weight;
}

void quant_weight_free(QuantizedWeight* weight) {
    if (!weight) return;

    free(weight->data);
    free(weight->scales);
    free(weight->zero_points);
    free(weight);
}

/* ============================================================================
 * INT8 量化
 * ============================================================================ */

int compute_int8_params(const float* data, size_t numel, float* scale) {
    if (!data || !scale || numel == 0) {
        return -1;
    }

    /* 找最大绝对值 */
    float max_abs = 0.0f;
    for (size_t i = 0; i < numel; i++) {
        float abs_val = fabsf(data[i]);
        if (abs_val > max_abs) {
            max_abs = abs_val;
        }
    }

    /* 对称量化: scale = max_abs / 127 */
    if (max_abs > 0.0f) {
        *scale = max_abs / 127.0f;
    } else {
        *scale = 1.0f;
    }

    return 0;
}

int compute_int8_params_asymmetric(const float* data, size_t numel,
                                   float* scale, int8_t* zero_point) {
    if (!data || !scale || !zero_point || numel == 0) {
        return -1;
    }

    /* 找最小最大值 */
    float min_val = data[0];
    float max_val = data[0];
    for (size_t i = 1; i < numel; i++) {
        if (data[i] < min_val) min_val = data[i];
        if (data[i] > max_val) max_val = data[i];
    }

    /* 非对称量化 */
    float range = max_val - min_val;
    if (range > 0.0f) {
        *scale = range / 255.0f;
        int zp = (int)roundf(-min_val / (*scale) - 128.0f);
        if (zp < -128) zp = -128;
        if (zp > 127) zp = 127;
        *zero_point = (int8_t)zp;
    } else {
        *scale = 1.0f;
        *zero_point = 0;
    }

    return 0;
}

int quantize_int8(const float* src, int8_t* dst, size_t numel, float scale) {
    if (!src || !dst || numel == 0 || scale <= 0.0f) {
        return -1;
    }

    float inv_scale = 1.0f / scale;
    for (size_t i = 0; i < numel; i++) {
        float quantized = roundf(src[i] * inv_scale);
        /* 钳位到 [-128, 127] */
        if (quantized > 127.0f) quantized = 127.0f;
        if (quantized < -128.0f) quantized = -128.0f;
        dst[i] = (int8_t)quantized;
    }

    return 0;
}

int dequantize_int8(const int8_t* src, float* dst, size_t numel, float scale) {
    if (!src || !dst || numel == 0 || scale <= 0.0f) {
        return -1;
    }

    for (size_t i = 0; i < numel; i++) {
        dst[i] = (float)src[i] * scale;
    }

    return 0;
}

/* ============================================================================
 * 块量化参数计算
 * ============================================================================ */

int compute_block_params(const float* data, size_t numel, size_t block_size,
                         float* scales, size_t num_blocks) {
    if (!data || !scales || numel == 0 || block_size == 0) {
        return -1;
    }

    for (size_t b = 0; b < num_blocks; b++) {
        size_t start = b * block_size;
        size_t end = start + block_size;
        if (end > numel) end = numel;

        /* 找块内最大绝对值 */
        float max_abs = 0.0f;
        for (size_t i = start; i < end; i++) {
            float abs_val = fabsf(data[i]);
            if (abs_val > max_abs) {
                max_abs = abs_val;
            }
        }

        scales[b] = max_abs > 0.0f ? max_abs : 1.0f;
    }

    return 0;
}

/* ============================================================================
 * INT4 块量化 (Q4_0 风格)
 * ============================================================================ */

int quantize_int4_block(const float* src, size_t numel, size_t block_size,
                        uint8_t* weights, float* scales) {
    if (!src || !weights || !scales || numel == 0 || block_size == 0) {
        return -1;
    }

    size_t num_blocks = (numel + block_size - 1) / block_size;

    for (size_t b = 0; b < num_blocks; b++) {
        size_t start = b * block_size;
        size_t end = start + block_size;
        if (end > numel) end = numel;

        /* 找块内最大绝对值 */
        float max_abs = 0.0f;
        for (size_t i = start; i < end; i++) {
            float abs_val = fabsf(src[i]);
            if (abs_val > max_abs) {
                max_abs = abs_val;
            }
        }

        /* 缩放因子 */
        float scale = max_abs / 7.0f;  /* INT4 范围 [-8, 7]，用 7 避免溢出 */
        if (scale < 1e-10f) scale = 1.0f;
        scales[b] = scale;

        float inv_scale = 1.0f / scale;

        /* 量化并打包 */
        for (size_t i = start; i < end; i++) {
            float quantized = roundf(src[i] * inv_scale);
            /* 钳位到 [-8, 7] */
            if (quantized > 7.0f) quantized = 7.0f;
            if (quantized < -8.0f) quantized = -8.0f;

            int8_t q4 = (int8_t)quantized;
            /* 将 -8..7 映射到 0..15 */
            uint8_t u4 = (uint8_t)(q4 + 8);

            size_t idx = i - start;
            size_t byte_idx = (start + idx) / 2;
            if ((start + idx) & 1) {
                /* 奇数位置: 高 4 位 */
                weights[byte_idx] |= (u4 << 4);
            } else {
                /* 偶数位置: 低 4 位 */
                weights[byte_idx] = u4;
            }
        }
    }

    return 0;
}

int dequantize_int4_block(const uint8_t* weights, const float* scales,
                          size_t numel, size_t block_size, float* dst) {
    if (!weights || !scales || !dst || numel == 0 || block_size == 0) {
        return -1;
    }

    size_t num_blocks = (numel + block_size - 1) / block_size;

    for (size_t b = 0; b < num_blocks; b++) {
        size_t start = b * block_size;
        size_t end = start + block_size;
        if (end > numel) end = numel;

        float scale = scales[b];

        for (size_t i = start; i < end; i++) {
            size_t byte_idx = i / 2;
            uint8_t u4;

            if (i & 1) {
                /* 奇数位置: 高 4 位 */
                u4 = (weights[byte_idx] >> 4) & 0xF;
            } else {
                /* 偶数位置: 低 4 位 */
                u4 = weights[byte_idx] & 0xF;
            }

            /* 将 0..15 映射回 -8..7 */
            int8_t q4 = (int8_t)u4 - 8;
            dst[i] = (float)q4 * scale;
        }
    }

    return 0;
}

/* ============================================================================
 * 张量量化/反量化
 * ============================================================================ */

QuantizedWeight* quantize_tensor(const Tensor* src, const QuantConfig* config) {
    if (!src || !config || !src->data) {
        return NULL;
    }

    size_t numel = shape_numel(&src->shape);
    const float* data = (const float*)src->data;

    QuantizedWeight* weight = quant_weight_new(numel, config);
    if (!weight) {
        return NULL;
    }

    switch (config->method) {
        case QUANT_METHOD_FP8_E4M3:
            /* FP8 量化 */
            f32_to_fp8_e4m3_batch(data, weight->data, numel);
            if (config->scale_format == SCALE_FORMAT_BLOCK_WISE && config->block_size > 0) {
                compute_block_params(data, numel, config->block_size,
                                    weight->scales, weight->num_scales);
            } else {
                weight->scales[0] = 1.0f;
            }
            break;

        case QUANT_METHOD_INT8:
            /* INT8 量化 */
            if (compute_int8_params(data, numel, &weight->scales[0]) == 0) {
                quantize_int8(data, (int8_t*)weight->data, numel, weight->scales[0]);
            }
            break;

        case QUANT_METHOD_INT4:
            /* INT4 块量化 */
            quantize_int4_block(data, numel, config->block_size,
                               weight->data, weight->scales);
            break;

        case QUANT_METHOD_NONE:
        default:
            /* 无量化，直接复制 */
            memcpy(weight->data, data, numel * sizeof(float));
            break;
    }

    return weight;
}

Tensor* dequantize_tensor(const QuantizedWeight* weight) {
    if (!weight || !weight->data) {
        return NULL;
    }

    /* 创建 1D 张量 (简化实现) */
    Tensor* dst = (Tensor*)malloc(sizeof(Tensor));
    if (!dst) {
        return NULL;
    }
    memset(dst, 0, sizeof(Tensor));

    /* 设置形状为 1D */
    size_t dims[1] = {weight->numel};
    dst->shape = shape_new(dims, 1);

    /* 计算步幅 */
    dst->strides[0] = 1;

    dst->dtype = DTYPE_F32;
    dst->offset = 0;
    dst->device.type = DEVICE_CPU;
    dst->device.id = 0;
    dst->owns_data = true;
    dst->data = malloc(weight->numel * sizeof(float));

    if (!dst->data) {
        free(dst);
        return NULL;
    }

    float* data = (float*)dst->data;

    switch (weight->config.method) {
        case QUANT_METHOD_FP8_E4M3:
            /* FP8 反量化 */
            if (weight->config.scale_format == SCALE_FORMAT_BLOCK_WISE &&
                weight->config.block_size > 0) {
                /* 块级反量化 */
                size_t num_blocks = weight->num_scales;
                size_t block_size = weight->config.block_size;
                for (size_t b = 0; b < num_blocks; b++) {
                    size_t start = b * block_size;
                    size_t end = start + block_size;
                    if (end > weight->numel) end = weight->numel;

                    for (size_t i = start; i < end; i++) {
                        float val = fp8_e4m3_to_f32(weight->data[i]);
                        data[i] = val * weight->scales[b];
                    }
                }
            } else {
                fp8_e4m3_to_f32_batch(weight->data, data, weight->numel);
                /* 应用全局缩放 */
                if (weight->scales && weight->scales[0] != 1.0f) {
                    for (size_t i = 0; i < weight->numel; i++) {
                        data[i] *= weight->scales[0];
                    }
                }
            }
            break;

        case QUANT_METHOD_INT8:
            /* INT8 反量化 */
            dequantize_int8((const int8_t*)weight->data, data, weight->numel, weight->scales[0]);
            break;

        case QUANT_METHOD_INT4:
            /* INT4 反量化 */
            dequantize_int4_block(weight->data, weight->scales, weight->numel,
                                  weight->config.block_size, data);
            break;

        case QUANT_METHOD_NONE:
        default:
            /* 无量化，直接复制 */
            memcpy(data, weight->data, weight->numel * sizeof(float));
            break;
    }

    return dst;
}
