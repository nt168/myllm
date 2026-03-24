/**
 * @file log.c
 * @brief 日志工具实现
 */

#include "utils/log.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <unistd.h>

/* 日志级别名称 */
static const char* level_names[] = {
    "TRACE", "DEBUG", "INFO", "WARN", "ERROR", "FATAL", "OFF"
};

/* 日志级别颜色 (ANSI) */
static const char* level_colors[] = {
    "\033[90m",      /* TRACE: 灰色 */
    "\033[36m",      /* DEBUG: 青色 */
    "\033[32m",      /* INFO: 绿色 */
    "\033[33m",      /* WARN: 黄色 */
    "\033[31m",      /* ERROR: 红色 */
    "\033[35m",      /* FATAL: 紫色 */
    "\033[0m"        /* OFF: 重置 */
};

/* 全局日志配置 */
static struct {
    LogConfig config;
    FILE* log_file;
    int initialized;
} g_log = {
    .config = {
        .min_level = LOG_LEVEL_INFO,
        .targets = LOG_TARGET_STDERR,
        .file_path = NULL,
        .callback = NULL,
        .callback_data = NULL,
        .show_timestamp = 1,
        .show_file_line = 0,
        .show_function = 0,
        .use_colors = 1
    },
    .log_file = NULL,
    .initialized = 0
};

int log_init(const LogConfig* config) {
    if (g_log.initialized) {
        log_shutdown();
    }

    if (config) {
        g_log.config = *config;
    }

    /* 如果指定了文件，打开它 */
    if (g_log.config.file_path) {
        g_log.log_file = fopen(g_log.config.file_path, "a");
        if (!g_log.log_file) {
            return -1;
        }
    }

    /* 检测是否是终端 */
    if (g_log.config.targets & (LOG_TARGET_STDOUT | LOG_TARGET_STDERR)) {
        int is_tty = 0;
        if (g_log.config.targets & LOG_TARGET_STDOUT) {
            is_tty = isatty(fileno(stdout));
        }
        if (g_log.config.targets & LOG_TARGET_STDERR) {
            is_tty = is_tty || isatty(fileno(stderr));
        }
        g_log.config.use_colors = is_tty && g_log.config.use_colors;
    }

    g_log.initialized = 1;
    return 0;
}

void log_shutdown(void) {
    if (g_log.log_file) {
        fclose(g_log.log_file);
        g_log.log_file = NULL;
    }
    g_log.initialized = 0;
}

void log_set_level(LogLevel level) {
    if (level >= LOG_LEVEL_TRACE && level <= LOG_LEVEL_OFF) {
        g_log.config.min_level = level;
    }
}

LogLevel log_get_level(void) {
    return g_log.config.min_level;
}

void log_set_targets(int targets) {
    g_log.config.targets = targets;
}

int log_set_file(const char* path) {
    if (g_log.log_file) {
        fclose(g_log.log_file);
        g_log.log_file = NULL;
    }

    if (path) {
        g_log.log_file = fopen(path, "a");
        if (!g_log.log_file) {
            return -1;
        }
        g_log.config.file_path = path;
        g_log.config.targets |= LOG_TARGET_FILE;
    }

    return 0;
}

void log_set_callback(LogCallback callback, void* user_data) {
    g_log.config.callback = callback;
    g_log.config.callback_data = user_data;
    if (callback) {
        g_log.config.targets |= LOG_TARGET_CALLBACK;
    }
}

void log_write_v(LogLevel level, const char* file, int line,
                 const char* func, const char* fmt, va_list args) {
    /* 检查级别 */
    if (level < g_log.config.min_level) {
        return;
    }

    /* 格式化消息 */
    char msg[4096];
    vsnprintf(msg, sizeof(msg), fmt, args);

    /* 构建完整日志行 */
    char line_buf[8192];
    int pos = 0;

    /* 时间戳 */
    if (g_log.config.show_timestamp) {
        time_t now = time(NULL);
        struct tm* tm_info = localtime(&now);
        pos += strftime(line_buf + pos, sizeof(line_buf) - pos,
                       "%Y-%m-%d %H:%M:%S", tm_info);
        pos += snprintf(line_buf + pos, sizeof(line_buf) - pos, " ");
    }

    /* 日志级别 */
    if (g_log.config.use_colors) {
        pos += snprintf(line_buf + pos, sizeof(line_buf) - pos,
                       "%s%-5s\033[0m ",
                       level_colors[level], level_names[level]);
    } else {
        pos += snprintf(line_buf + pos, sizeof(line_buf) - pos,
                       "%-5s ", level_names[level]);
    }

    /* 文件和行号 */
    if (g_log.config.show_file_line) {
        const char* basename = strrchr(file, '/');
        basename = basename ? basename + 1 : file;
        pos += snprintf(line_buf + pos, sizeof(line_buf) - pos,
                       "[%s:%d] ", basename, line);
    }

    /* 函数名 */
    if (g_log.config.show_function && func) {
        pos += snprintf(line_buf + pos, sizeof(line_buf) - pos,
                       "%s(): ", func);
    }

    /* 消息 */
    pos += snprintf(line_buf + pos, sizeof(line_buf) - pos, "%s\n", msg);

    /* 输出到目标 */
    if (g_log.config.targets & LOG_TARGET_STDOUT) {
        fputs(line_buf, stdout);
        fflush(stdout);
    }

    if (g_log.config.targets & LOG_TARGET_STDERR) {
        fputs(line_buf, stderr);
        fflush(stderr);
    }

    if ((g_log.config.targets & LOG_TARGET_FILE) && g_log.log_file) {
        fputs(line_buf, g_log.log_file);
        fflush(g_log.log_file);
    }

    if ((g_log.config.targets & LOG_TARGET_CALLBACK) && g_log.config.callback) {
        g_log.config.callback(level, file, line, func, msg, g_log.config.callback_data);
    }
}

void log_write(LogLevel level, const char* file, int line,
               const char* func, const char* fmt, ...) {
    va_list args;
    va_start(args, fmt);
    log_write_v(level, file, line, func, fmt, args);
    va_end(args);
}

const char* log_level_name(LogLevel level) {
    if (level < 0 || level > LOG_LEVEL_OFF) {
        return "UNKNOWN";
    }
    return level_names[level];
}

LogLevel log_level_from_name(const char* name) {
    if (!name) return LOG_LEVEL_INFO;

    for (int i = 0; i <= LOG_LEVEL_OFF; i++) {
        if (strcasecmp(name, level_names[i]) == 0) {
            return (LogLevel)i;
        }
    }

    return LOG_LEVEL_INFO;
}
