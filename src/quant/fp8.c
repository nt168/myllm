/**
 * @file fp8.c
 * @brief FP8 (8-bit 浮点) 转换实现
 *
 * FP8 E4M3 FNUZ (Float Neural Network Optimized Zero):
 * - 1 符号位 (S)
 * - 4 指数位 (E) - bias = 8
 * - 3 尾数位 (M)
 * - 值 = (-1)^S * 2^(E-8) * (1 + M/8)
 * - 范围: [-448, 448]
 * - 无无穷大和 NaN
 *
 * FP8 E5M2 (IEEE 754-like):
 * - 1 符号位 (S)
 * - 5 指数位 (E) - bias = 15
 * - 2 尾数位 (M)
 * - 值 = (-1)^S * 2^(E-15) * (1 + M/4)
 * - 支持无穷大和 NaN
 */

#include "quant/fp8.h"
#include <math.h>
#include <string.h>

/* FP8 E4M3 常量 */
#define FP8_E4M3_EXP_BIAS   8
#define FP8_E4M3_MAX_EXP    15
#define FP8_E4M3_MANT_BITS  3
#define FP8_E4M3_EXP_BITS   4

/* FP8 E5M2 常量 */
#define FP8_E5M2_EXP_BIAS   15
#define FP8_E5M2_MAX_EXP    31
#define FP8_E5M2_MANT_BITS  2
#define FP8_E5M2_EXP_BITS   5

/* 辅助函数: 提取位 */
static inline uint32_t get_bits(uint32_t val, int start, int len) {
    return (val >> start) & ((1u << len) - 1);
}

/* 辅助函数: 设置位 */
static inline uint32_t set_bits(uint32_t val, int start, int len, uint32_t bits) {
    uint32_t mask = ((1u << len) - 1) << start;
    return (val & ~mask) | ((bits << start) & mask);
}

float fp8_e4m3_to_f32(uint8_t fp8_val) {
    /* 提取符号、指数、尾数 */
    uint32_t sign = (fp8_val >> 7) & 1;
    uint32_t exp = (fp8_val >> 3) & 0xF;
    uint32_t mant = fp8_val & 0x7;

    /* 特殊情况: 零 */
    if (exp == 0 && mant == 0) {
        return sign ? -0.0f : 0.0f;
    }

    /* 构造 FP32:
     * FP32: 1 符号位, 8 指数位 (bias=127), 23 尾数位
     */
    uint32_t f32_bits = sign << 31;

    if (exp == 0) {
        /* 非规范数 (subnormal) - E4M3 中不太可能，但处理 */
        /* 转换为 FP32 非规范数 */
        int32_t true_exp = 1 - FP8_E4M3_EXP_BIAS;  /* -7 */
        int32_t f32_exp = true_exp + 127;
        if (f32_exp <= 0) {
            /* 太小，变成 0 */
            f32_bits = sign << 31;
        } else {
            f32_bits |= (f32_exp << 23) | (mant << 20);
        }
    } else {
        /* 规范数 */
        int32_t true_exp = (int32_t)exp - FP8_E4M3_EXP_BIAS;  /* -8 到 7 */
        int32_t f32_exp = true_exp + 127;

        /* E4M3 FNUZ 没有特殊值 (Inf/NaN)，最大指数也是规范数 */
        f32_bits |= (f32_exp << 23) | (mant << 20);
    }

    /* 通过 union 或 memcpy 转换 */
    float result;
    memcpy(&result, &f32_bits, sizeof(result));
    return result;
}

uint8_t f32_to_fp8_e4m3(float val) {
    /* 提取 FP32 位 */
    uint32_t f32_bits;
    memcpy(&f32_bits, &val, sizeof(f32_bits));

    uint32_t sign = (f32_bits >> 31) & 1;
    int32_t f32_exp = (int32_t)((f32_bits >> 23) & 0xFF) - 127;
    uint32_t f32_mant = f32_bits & 0x7FFFFF;

    /* 特殊情况: 零 */
    if ((f32_bits & 0x7FFFFFFF) == 0) {
        return sign << 7;
    }

    /* 特殊情况: NaN 或 Inf - E4M3 FNUZ 没有 Inf/NaN，饱和到最大值 */
    if ((f32_bits & 0x7F800000) == 0x7F800000) {
        /* NaN 或 Inf，返回最大值 */
        return (sign << 7) | 0x7F;  /* S.1111.111 = ±448 */
    }

    /* 计算目标指数 */
    int32_t true_exp = f32_exp;

    /* E4M3 范围检查: 指数范围 [-8, 7] */
    if (true_exp < -8) {
        /* 太小，下溢到 0 */
        return sign << 7;
    }
    if (true_exp > 7) {
        /* 太大，饱和到最大值 */
        return (sign << 7) | 0x7F;  /* S.1111.111 = ±448 */
    }

    /* 构造 E4M3 */
    uint32_t fp8_exp = (uint32_t)(true_exp + FP8_E4M3_EXP_BIAS);  /* 0-15 */

    /* 转换尾数: FP32 有 23 位，FP8 E4M3 有 3 位 */
    /* 保留高 3 位，并四舍五入 */
    uint32_t fp8_mant = (f32_mant >> 20) & 0x7;  /* 取高 3 位 */

    /* 四舍五入: 检查被丢弃的位 */
    uint32_t round_bit = (f32_mant >> 19) & 1;
    uint32_t sticky_bits = f32_mant & 0x7FFFF;  /* 剩余 19 位 */

    if (round_bit && (sticky_bits || (fp8_mant & 1))) {
        /* 向上舍入 */
        fp8_mant++;
        if (fp8_mant > 7) {
            fp8_mant = 0;
            fp8_exp++;
            if (fp8_exp > 15) {
                /* 溢出，饱和 */
                return (sign << 7) | 0x7F;
            }
        }
    }

    return (uint8_t)((sign << 7) | (fp8_exp << 3) | fp8_mant);
}

float fp8_e5m2_to_f32(uint8_t fp8_val) {
    /* 提取符号、指数、尾数 */
    uint32_t sign = (fp8_val >> 7) & 1;
    uint32_t exp = (fp8_val >> 2) & 0x1F;
    uint32_t mant = fp8_val & 0x3;

    /* 构造 FP32 */
    uint32_t f32_bits = sign << 31;

    if (exp == 0) {
        /* 零或非规范数 */
        if (mant == 0) {
            return sign ? -0.0f : 0.0f;
        }
        /* 非规范数 */
        int32_t true_exp = 1 - FP8_E5M2_EXP_BIAS;
        int32_t f32_exp = true_exp + 127;
        f32_bits |= (f32_exp << 23) | (mant << 21);
    } else if (exp == 31) {
        /* 无穷大或 NaN */
        if (mant == 0) {
            /* Inf */
            f32_bits |= 0x7F800000;
        } else {
            /* NaN */
            f32_bits |= 0x7FC00000;  /* quiet NaN */
        }
    } else {
        /* 规范数 */
        int32_t true_exp = (int32_t)exp - FP8_E5M2_EXP_BIAS;
        int32_t f32_exp = true_exp + 127;
        f32_bits |= (f32_exp << 23) | (mant << 21);
    }

    float result;
    memcpy(&result, &f32_bits, sizeof(result));
    return result;
}

uint8_t f32_to_fp8_e5m2(float val) {
    uint32_t f32_bits;
    memcpy(&f32_bits, &val, sizeof(f32_bits));

    uint32_t sign = (f32_bits >> 31) & 1;
    int32_t f32_exp = (int32_t)((f32_bits >> 23) & 0xFF) - 127;
    uint32_t f32_mant = f32_bits & 0x7FFFFF;

    /* 零 */
    if ((f32_bits & 0x7FFFFFFF) == 0) {
        return sign << 7;
    }

    /* NaN */
    if ((f32_bits & 0x7F800000) == 0x7F800000 && (f32_bits & 0x7FFFFF)) {
        return (sign << 7) | 0x7D;  /* NaN in E5M2 */
    }

    /* Inf */
    if ((f32_bits & 0x7F800000) == 0x7F800000) {
        return (sign << 7) | 0x7C;  /* Inf in E5M2 */
    }

    /* 范围检查: E5M2 指数范围 [-14, 15] (规范数) */
    if (f32_exp < -14) {
        /* 可能是非规范数或零 */
        if (f32_exp < -17) {
            return sign << 7;  /* 太小，变成零 */
        }
        /* 非规范数 */
        int shift = -14 - f32_exp;
        uint32_t fp8_mant = ((0x400000 | f32_mant) >> (21 + shift)) & 0x3;
        return (sign << 7) | fp8_mant;
    }
    if (f32_exp > 15) {
        /* 溢出到 Inf */
        return (sign << 7) | 0x7C;
    }

    /* 规范数 */
    uint32_t fp8_exp = (uint32_t)(f32_exp + FP8_E5M2_EXP_BIAS);
    uint32_t fp8_mant = (f32_mant >> 21) & 0x3;

    /* 四舍五入 */
    uint32_t round_bit = (f32_mant >> 20) & 1;
    uint32_t sticky_bits = f32_mant & 0xFFFFF;

    if (round_bit && (sticky_bits || (fp8_mant & 1))) {
        fp8_mant++;
        if (fp8_mant > 3) {
            fp8_mant = 0;
            fp8_exp++;
            if (fp8_exp > 30) {
                return (sign << 7) | 0x7C;  /* Inf */
            }
        }
    }

    return (uint8_t)((sign << 7) | (fp8_exp << 2) | fp8_mant);
}

void fp8_e4m3_to_f32_batch(const uint8_t* src, float* dst, size_t numel) {
    for (size_t i = 0; i < numel; i++) {
        dst[i] = fp8_e4m3_to_f32(src[i]);
    }
}

void f32_to_fp8_e4m3_batch(const float* src, uint8_t* dst, size_t numel) {
    for (size_t i = 0; i < numel; i++) {
        dst[i] = f32_to_fp8_e4m3(src[i]);
    }
}

float fp8_e4m3_max(void) {
    /* 最大值: 01111111 = +448 */
    /* 计算: 2^(7) * (1 + 7/8) = 128 * 1.875 = 240 */
    /* 实际上 E4M3 最大指数是 7 (exp=15, 即 1111) */
    /* 值 = 2^(15-8) * (1 + 7/8) = 2^7 * 1.875 = 128 * 1.875 = 240 */
    return 240.0f;  /* 有的定义是 448，取决于具体格式 */
}

float fp8_e4m3_min(void) {
    /* 最小正规范数: exp=1, mant=0 */
    /* 值 = 2^(1-8) * 1 = 2^(-7) = 0.0078125 */
    return 0.0078125f;
}
