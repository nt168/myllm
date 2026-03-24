/**
 * @file utils.c
 * @brief 工具模块主实现
 */

#include "utils/utils.h"

#define UTILS_VERSION "0.1.0"

const char* utils_version(void) {
    return UTILS_VERSION;
}

int utils_init(void) {
    /* 初始化日志系统 */
    LogConfig log_config = {
        .min_level = LOG_LEVEL_INFO,
        .targets = LOG_TARGET_STDERR,
        .file_path = NULL,
        .callback = NULL,
        .callback_data = NULL,
        .show_timestamp = 1,
        .show_file_line = 0,
        .show_function = 0,
        .use_colors = 1
    };

    if (log_init(&log_config) != 0) {
        return -1;
    }

    /* 启用内存跟踪 */
    memory_tracking_enable();

    return 0;
}

void utils_shutdown(void) {
    memory_tracking_disable();
    log_shutdown();
}
