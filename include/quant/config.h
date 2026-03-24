/**
 * @file config.h
 * @brief 量化配置定义
 */

#ifndef MYLLM_QUANT_CONFIG_H
#define MYLLM_QUANT_CONFIG_H

#include "tensor/dtype.h"

/**
 * @brief 量化方法类型
 */
typedef enum {
    QUANT_METHOD_NONE = 0,      ///< 无量化 (FP32/FP16)
    QUANT_METHOD_FP8_E4M3,      ///< FP8 E4M3 FNUZ 格式
    QUANT_METHOD_FP8_E5M2,      ///< FP8 E5M2 格式
    QUANT_METHOD_INT8,          ///< INT8 量化
    QUANT_METHOD_INT4           ///< INT4 量化 (Q4_0, Q4_K 等)
} QuantMethod;

/**
 * @brief 数据布局方式
 */
typedef enum {
    DATA_LAYOUT_ROW_MAJOR = 0,  ///< 行主序
    DATA_LAYOUT_COLUMN_MAJOR,   ///< 列主序
    DATA_LAYOUT_TILE            ///< 分块布局
} DataLayout;

/**
 * @brief 缩放因子格式
 */
typedef enum {
    SCALE_FORMAT_BLOCK_WISE = 0, ///< 按块缩放
    SCALE_FORMAT_PER_CHANNEL,    ///< 按通道缩放
    SCALE_FORMAT_PER_TENSOR      ///< 整个张量单一缩放因子
} ScaleFormat;

/**
 * @brief 量化配置结构
 */
typedef struct {
    QuantMethod method;          ///< 量化方法
    size_t block_size;           ///< 块大小 (用于 block-wise 量化)
    DataLayout layout;           ///< 数据布局
    ScaleFormat scale_format;    ///< 缩放因子格式
    DType scale_dtype;           ///< 缩放因子数据类型
    int bits;                    ///< 量化位数 (4, 8 等)
} QuantConfig;

/**
 * @brief 创建默认量化配置 (无量化)
 * @return 默认配置
 */
QuantConfig quant_config_default(void);

/**
 * @brief 创建 FP8 E4M3 量化配置
 * @param block_size 块大小 (0 表示 per-tensor)
 * @return FP8 E4M3 配置
 */
QuantConfig quant_config_fp8_e4m3(size_t block_size);

/**
 * @brief 创建 INT8 量化配置
 * @param scale_format 缩放因子格式
 * @return INT8 配置
 */
QuantConfig quant_config_int8(ScaleFormat scale_format);

/**
 * @brief 创建 INT4 量化配置
 * @param block_size 块大小
 * @return INT4 配置
 */
QuantConfig quant_config_int4(size_t block_size);

/**
 * @brief 获取量化方法名称
 * @param method 量化方法
 * @return 方法名称字符串
 */
const char* quant_method_name(QuantMethod method);

/**
 * @brief 获取缩放格式名称
 * @param format 缩放格式
 * @return 格式名称字符串
 */
const char* scale_format_name(ScaleFormat format);

/**
 * @brief 计算量化后的字节数
 * @param numel 元素数量
 * @param config 量化配置
 * @return 量化后需要的字节数
 */
size_t quant_compressed_size(size_t numel, const QuantConfig* config);

#endif /* MYLLM_QUANT_CONFIG_H */
