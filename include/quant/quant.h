/**
 * @file quant.h
 * @brief 量化模块主头文件
 */

#ifndef MYLLM_QUANT_QUANT_H
#define MYLLM_QUANT_QUANT_H

#include "quant/config.h"
#include "quant/fp8.h"
#include "tensor/tensor.h"

/**
 * @brief 量化权重结构
 *
 * 存储量化后的权重和元数据
 */
typedef struct {
    uint8_t* data;              ///< 量化数据
    float* scales;              ///< 缩放因子数组
    int8_t* zero_points;        ///< 零点数组 (可选)
    size_t numel;               ///< 原始元素数量
    size_t data_size;           ///< 量化数据大小 (字节)
    size_t num_scales;          ///< 缩放因子数量
    QuantConfig config;         ///< 量化配置
} QuantizedWeight;

/**
 * @brief 创建量化权重
 * @param numel 原始元素数量
 * @param config 量化配置
 * @return 量化权重指针，失败返回 NULL
 */
QuantizedWeight* quant_weight_new(size_t numel, const QuantConfig* config);

/**
 * @brief 释放量化权重
 * @param weight 量化权重指针
 */
void quant_weight_free(QuantizedWeight* weight);

/**
 * @brief 量化 FP32 张量
 * @param src FP32 源张量
 * @param config 量化配置
 * @return 量化权重，失败返回 NULL
 */
QuantizedWeight* quantize_tensor(const Tensor* src, const QuantConfig* config);

/**
 * @brief 反量化到 FP32 张量
 * @param weight 量化权重
 * @return FP32 张量，失败返回 NULL
 */
Tensor* dequantize_tensor(const QuantizedWeight* weight);

/**
 * @brief 计算 INT8 量化参数 (对称量化)
 * @param data FP32 数据
 * @param numel 元素数量
 * @param scale 输出缩放因子
 * @return 0 成功，-1 失败
 */
int compute_int8_params(const float* data, size_t numel, float* scale);

/**
 * @brief 计算 INT8 量化参数 (非对称量化)
 * @param data FP32 数据
 * @param numel 元素数量
 * @param scale 输出缩放因子
 * @param zero_point 输出零点
 * @return 0 成功，-1 失败
 */
int compute_int8_params_asymmetric(const float* data, size_t numel,
                                   float* scale, int8_t* zero_point);

/**
 * @brief FP32 数据量化为 INT8
 * @param src FP32 源数据
 * @param dst INT8 目标数据
 * @param numel 元素数量
 * @param scale 缩放因子
 * @return 0 成功，-1 失败
 */
int quantize_int8(const float* src, int8_t* dst, size_t numel, float scale);

/**
 * @brief INT8 数据反量化为 FP32
 * @param src INT8 源数据
 * @param dst FP32 目标数据
 * @param numel 元素数量
 * @param scale 缩放因子
 * @return 0 成功，-1 失败
 */
int dequantize_int8(const int8_t* src, float* dst, size_t numel, float scale);

/**
 * @brief 计算块量化参数
 * @param data FP32 数据
 * @param numel 元素数量
 * @param block_size 块大小
 * @param scales 输出缩放因子数组
 * @param num_blocks 块数量
 * @return 0 成功，-1 失败
 */
int compute_block_params(const float* data, size_t numel, size_t block_size,
                         float* scales, size_t num_blocks);

/**
 * @brief INT4 块量化 (Q4_0 风格)
 * @param src FP32 源数据
 * @param numel 元素数量
 * @param block_size 块大小 (通常 32)
 * @param weights 输出量化权重 (每个元素 4 位，两个元素打包一个字节)
 * @param scales 输出缩放因子
 * @return 0 成功，-1 失败
 */
int quantize_int4_block(const float* src, size_t numel, size_t block_size,
                        uint8_t* weights, float* scales);

/**
 * @brief INT4 块反量化
 * @param weights 量化权重 (每个元素 4 位)
 * @param scales 缩放因子数组
 * @param numel 元素数量
 * @param block_size 块大小
 * @param dst FP32 目标数据
 * @return 0 成功，-1 失败
 */
int dequantize_int4_block(const uint8_t* weights, const float* scales,
                          size_t numel, size_t block_size, float* dst);

/**
 * @brief 获取量化模块版本
 * @return 版本字符串
 */
const char* quant_version(void);

#endif /* MYLLM_QUANT_QUANT_H */
