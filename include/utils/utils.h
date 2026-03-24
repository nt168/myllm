/**
 * @file utils.h
 * @brief 工具模块主头文件
 */

#ifndef MYLLM_UTILS_UTILS_H
#define MYLLM_UTILS_UTILS_H

#include "utils/log.h"
#include "utils/timer.h"
#include "utils/error.h"
#include "utils/memory.h"

/**
 * @brief 获取工具模块版本
 * @return 版本字符串
 */
const char* utils_version(void);

/**
 * @brief 初始化工具模块
 * @return 0 成功，-1 失败
 */
int utils_init(void);

/**
 * @brief 关闭工具模块
 */
void utils_shutdown(void);

#endif /* MYLLM_UTILS_UTILS_H */
