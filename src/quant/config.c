/**
 * @file config.c
 * @brief 量化配置实现
 */

#include "quant/config.h"
#include <string.h>

/* 量化方法名称 */
static const char* quant_method_names[] = {
    "None",
    "FP8-E4M3",
    "FP8-E5M2",
    "INT8",
    "INT4"
};

/* 缩放格式名称 */
static const char* scale_format_names[] = {
    "BlockWise",
    "PerChannel",
    "PerTensor"
};

QuantConfig quant_config_default(void) {
    QuantConfig config = {
        .method = QUANT_METHOD_NONE,
        .block_size = 0,
        .layout = DATA_LAYOUT_ROW_MAJOR,
        .scale_format = SCALE_FORMAT_PER_TENSOR,
        .scale_dtype = DTYPE_F32,
        .bits = 32
    };
    return config;
}

QuantConfig quant_config_fp8_e4m3(size_t block_size) {
    QuantConfig config = {
        .method = QUANT_METHOD_FP8_E4M3,
        .block_size = block_size,
        .layout = DATA_LAYOUT_ROW_MAJOR,
        .scale_format = block_size > 0 ? SCALE_FORMAT_BLOCK_WISE : SCALE_FORMAT_PER_TENSOR,
        .scale_dtype = DTYPE_F32,
        .bits = 8
    };
    return config;
}

QuantConfig quant_config_int8(ScaleFormat scale_format) {
    QuantConfig config = {
        .method = QUANT_METHOD_INT8,
        .block_size = 0,
        .layout = DATA_LAYOUT_ROW_MAJOR,
        .scale_format = scale_format,
        .scale_dtype = DTYPE_F32,
        .bits = 8
    };
    return config;
}

QuantConfig quant_config_int4(size_t block_size) {
    QuantConfig config = {
        .method = QUANT_METHOD_INT4,
        .block_size = block_size > 0 ? block_size : 32,  /* 默认块大小 32 */
        .layout = DATA_LAYOUT_ROW_MAJOR,
        .scale_format = SCALE_FORMAT_BLOCK_WISE,
        .scale_dtype = DTYPE_F16,
        .bits = 4
    };
    return config;
}

const char* quant_method_name(QuantMethod method) {
    if (method < 0 || method > QUANT_METHOD_INT4) {
        return "Unknown";
    }
    return quant_method_names[method];
}

const char* scale_format_name(ScaleFormat format) {
    if (format < 0 || format > SCALE_FORMAT_PER_TENSOR) {
        return "Unknown";
    }
    return scale_format_names[format];
}

size_t quant_compressed_size(size_t numel, const QuantConfig* config) {
    if (!config || numel == 0) {
        return 0;
    }

    switch (config->method) {
        case QUANT_METHOD_NONE:
            return numel * sizeof(float);

        case QUANT_METHOD_FP8_E4M3:
        case QUANT_METHOD_FP8_E5M2:
            /* FP8: 每个元素 1 字节 + 可选缩放因子 */
            if (config->scale_format == SCALE_FORMAT_BLOCK_WISE && config->block_size > 0) {
                size_t num_blocks = (numel + config->block_size - 1) / config->block_size;
                return numel * sizeof(uint8_t) + num_blocks * sizeof(float);
            }
            return numel * sizeof(uint8_t);

        case QUANT_METHOD_INT8:
            /* INT8: 每个元素 1 字节 + 缩放因子 */
            if (config->scale_format == SCALE_FORMAT_PER_CHANNEL) {
                /* 假设通道数为 numel / 某个维度，这里简化处理 */
                return numel * sizeof(int8_t) + sizeof(float);
            }
            return numel * sizeof(int8_t) + sizeof(float);

        case QUANT_METHOD_INT4:
            /* INT4: 每两个元素打包成 1 字节 + 每块一个缩放因子 */
            if (config->block_size > 0) {
                size_t num_blocks = (numel + config->block_size - 1) / config->block_size;
                return (numel + 1) / 2 + num_blocks * sizeof(float);
            }
            return (numel + 1) / 2 + sizeof(float);

        default:
            return numel * sizeof(float);
    }
}
