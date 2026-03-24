/**
 * @file fp8.h
 * @brief FP8 (8-bit 浮点) 转换函数
 *
 * FP8 E4M3 FNUZ 格式:
 * - 1 符号位
 * - 4 指数位 (bias = 8)
 * - 3 尾数位
 * - 无无穷大和 NaN
 * - 范围: [-448, 448]
 */

#ifndef MYLLM_QUANT_FP8_H
#define MYLLM_QUANT_FP8_H

#include <stdint.h>
#include <stddef.h>

/**
 * @brief FP8 E4M3 FNUZ 转 FP32
 * @param fp8_val FP8 值 (uint8_t 存储)
 * @return FP32 值
 */
float fp8_e4m3_to_f32(uint8_t fp8_val);

/**
 * @brief FP32 转 FP8 E4M3 FNUZ
 * @param val FP32 值
 * @return FP8 值 (uint8_t 存储)
 */
uint8_t f32_to_fp8_e4m3(float val);

/**
 * @brief FP8 E5M2 转 FP32
 * @param fp8_val FP8 E5M2 值
 * @return FP32 值
 */
float fp8_e5m2_to_f32(uint8_t fp8_val);

/**
 * @brief FP32 转 FP8 E5M2
 * @param val FP32 值
 * @return FP8 E5M2 值
 */
uint8_t f32_to_fp8_e5m2(float val);

/**
 * @brief 批量 FP8 E4M3 转 FP32
 * @param src FP8 源数据
 * @param dst FP32 目标数据
 * @param numel 元素数量
 */
void fp8_e4m3_to_f32_batch(const uint8_t* src, float* dst, size_t numel);

/**
 * @brief 批量 FP32 转 FP8 E4M3
 * @param src FP32 源数据
 * @param dst FP8 目标数据
 * @param numel 元素数量
 */
void f32_to_fp8_e4m3_batch(const float* src, uint8_t* dst, size_t numel);

/**
 * @brief 获取 FP8 E4M3 最大值
 * @return 最大可表示值
 */
float fp8_e4m3_max(void);

/**
 * @brief 获取 FP8 E4M3 最小正值
 * @return 最小正规范数
 */
float fp8_e4m3_min(void);

#endif /* MYLLM_QUANT_FP8_H */
