/**
 * @file log.h
 * @brief 日志工具
 *
 * 提供分级日志功能，支持不同日志级别和输出目标
 */

#ifndef MYLLM_UTILS_LOG_H
#define MYLLM_UTILS_LOG_H

#include <stdarg.h>
#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

/**
 * @brief 日志级别
 */
typedef enum {
    LOG_LEVEL_TRACE = 0,    ///< 最详细日志
    LOG_LEVEL_DEBUG = 1,    ///< 调试日志
    LOG_LEVEL_INFO  = 2,    ///< 信息日志
    LOG_LEVEL_WARN  = 3,    ///< 警告日志
    LOG_LEVEL_ERROR = 4,    ///< 错误日志
    LOG_LEVEL_FATAL = 5,    ///< 致命错误日志
    LOG_LEVEL_OFF   = 6     ///< 关闭日志
} LogLevel;

/**
 * @brief 日志输出目标类型
 */
typedef enum {
    LOG_TARGET_NONE     = 0,    ///< 无输出
    LOG_TARGET_STDOUT   = 1,    ///< 标准输出
    LOG_TARGET_STDERR   = 2,    ///< 标准错误
    LOG_TARGET_FILE     = 4,    ///< 文件
    LOG_TARGET_CALLBACK = 8     ///< 回调函数
} LogTarget;

/**
 * @brief 日志回调函数类型
 * @param level 日志级别
 * @param file 源文件名
 * @param line 行号
 * @param func 函数名
 * @param msg 日志消息
 * @param user_data 用户数据
 */
typedef void (*LogCallback)(LogLevel level, const char* file, int line,
                           const char* func, const char* msg, void* user_data);

/**
 * @brief 日志配置结构
 */
typedef struct {
    LogLevel min_level;         ///< 最小日志级别
    int targets;                ///< 输出目标 (LogTarget 位掩码)
    const char* file_path;      ///< 日志文件路径
    LogCallback callback;       ///< 日志回调
    void* callback_data;        ///< 回调用户数据
    int show_timestamp;         ///< 显示时间戳
    int show_file_line;         ///< 显示文件和行号
    int show_function;          ///< 显示函数名
    int use_colors;             ///< 使用颜色 (仅终端)
} LogConfig;

/* ============================================================================
 * 配置函数
 * ============================================================================ */

/**
 * @brief 初始化日志系统
 * @param config 日志配置，NULL 使用默认配置
 * @return 0 成功，-1 失败
 */
int log_init(const LogConfig* config);

/**
 * @brief 关闭日志系统
 */
void log_shutdown(void);

/**
 * @brief 设置最小日志级别
 * @param level 日志级别
 */
void log_set_level(LogLevel level);

/**
 * @brief 获取当前日志级别
 * @return 当前日志级别
 */
LogLevel log_get_level(void);

/**
 * @brief 设置日志输出目标
 * @param targets 输出目标位掩码
 */
void log_set_targets(int targets);

/**
 * @brief 设置日志文件
 * @param path 文件路径
 * @return 0 成功，-1 失败
 */
int log_set_file(const char* path);

/**
 * @brief 设置日志回调
 * @param callback 回调函数
 * @param user_data 用户数据
 */
void log_set_callback(LogCallback callback, void* user_data);

/* ============================================================================
 * 日志函数
 * ============================================================================ */

/**
 * @brief 内部日志函数 (不直接调用)
 */
void log_write(LogLevel level, const char* file, int line,
               const char* func, const char* fmt, ...);

/**
 * @brief 内部日志函数 (va_list 版本)
 */
void log_write_v(LogLevel level, const char* file, int line,
                 const char* func, const char* fmt, va_list args);

/* ============================================================================
 * 便捷宏
 * ============================================================================ */

#define LOG_TRACE(...) \
    log_write(LOG_LEVEL_TRACE, __FILE__, __LINE__, __func__, __VA_ARGS__)

#define LOG_DEBUG(...) \
    log_write(LOG_LEVEL_DEBUG, __FILE__, __LINE__, __func__, __VA_ARGS__)

#define LOG_INFO(...) \
    log_write(LOG_LEVEL_INFO, __FILE__, __LINE__, __func__, __VA_ARGS__)

#define LOG_WARN(...) \
    log_write(LOG_LEVEL_WARN, __FILE__, __LINE__, __func__, __VA_ARGS__)

#define LOG_ERROR(...) \
    log_write(LOG_LEVEL_ERROR, __FILE__, __LINE__, __func__, __VA_ARGS__)

#define LOG_FATAL(...) \
    log_write(LOG_LEVEL_FATAL, __FILE__, __LINE__, __func__, __VA_ARGS__)

/* 条件日志 */
#define LOG_DEBUG_IF(cond, ...) \
    do { if (cond) LOG_DEBUG(__VA_ARGS__); } while(0)

#define LOG_INFO_IF(cond, ...) \
    do { if (cond) LOG_INFO(__VA_ARGS__); } while(0)

/* 进入/退出函数日志 */
#define LOG_ENTER() LOG_TRACE(">>> %s", __func__)
#define LOG_EXIT()  LOG_TRACE("<<< %s", __func__)
#define LOG_EXIT_VAL(val) LOG_TRACE("<<< %s = %d", __func__, (int)(val))

/* ============================================================================
 * 工具函数
 * ============================================================================ */

/**
 * @brief 获取日志级别名称
 * @param level 日志级别
 * @return 级别名称字符串
 */
const char* log_level_name(LogLevel level);

/**
 * @brief 从字符串解析日志级别
 * @param name 级别名称
 * @return 日志级别，无效返回 LOG_LEVEL_INFO
 */
LogLevel log_level_from_name(const char* name);

#ifdef __cplusplus
}
#endif

#endif /* MYLLM_UTILS_LOG_H */
