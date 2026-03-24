/**
 * @file timer.h
 * @brief 计时工具
 *
 * 提供高精度计时功能，用于性能分析和基准测试
 */

#ifndef MYLLM_UTILS_TIMER_H
#define MYLLM_UTILS_TIMER_H

#include <stdint.h>
#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

/**
 * @brief 计时器结构
 */
typedef struct {
    uint64_t start_ns;      ///< 开始时间 (纳秒)
    uint64_t end_ns;        ///< 结束时间 (纳秒)
    uint64_t total_ns;      ///< 累计时间 (纳秒)
    size_t count;           ///< 计数
    const char* name;       ///< 计时器名称
} Timer;

/**
 * @brief 时间统计结构
 */
typedef struct {
    double total_ms;        ///< 总时间 (毫秒)
    double avg_ms;          ///< 平均时间 (毫秒)
    double min_ms;          ///< 最小时间 (毫秒)
    double max_ms;          ///< 最大时间 (毫秒)
    size_t count;           ///< 调用次数
} TimerStats;

/* ============================================================================
 * 基本计时函数
 * ============================================================================ */

/**
 * @brief 获取当前时间 (纳秒)
 * @return 当前时间戳 (纳秒)
 */
uint64_t timer_now_ns(void);

/**
 * @brief 获取当前时间 (微秒)
 * @return 当前时间戳 (微秒)
 */
uint64_t timer_now_us(void);

/**
 * @brief 获取当前时间 (毫秒)
 * @return 当前时间戳 (毫秒)
 */
uint64_t timer_now_ms(void);

/**
 * @brief 获取当前时间 (秒)
 * @return 当前时间戳 (秒)
 */
double timer_now_sec(void);

/* ============================================================================
 * 计时器操作
 * ============================================================================ */

/**
 * @brief 创建计时器
 * @param name 计时器名称 (可选)
 * @return 计时器指针，失败返回 NULL
 */
Timer* timer_new(const char* name);

/**
 * @brief 释放计时器
 * @param timer 计时器指针
 */
void timer_free(Timer* timer);

/**
 * @brief 初始化计时器 (栈上分配)
 * @param timer 计时器指针
 * @param name 名称 (可选)
 */
void timer_init(Timer* timer, const char* name);

/**
 * @brief 开始计时
 * @param timer 计时器指针
 */
void timer_start(Timer* timer);

/**
 * @brief 停止计时
 * @param timer 计时器指针
 * @return 经过的时间 (纳秒)
 */
uint64_t timer_stop(Timer* timer);

/**
 * @brief 重置计时器
 * @param timer 计时器指针
 */
void timer_reset(Timer* timer);

/**
 * @brief 获取经过的时间 (纳秒)
 * @param timer 计时器指针
 * @return 经过的时间
 */
uint64_t timer_elapsed_ns(const Timer* timer);

/**
 * @brief 获取经过的时间 (微秒)
 * @param timer 计时器指针
 * @return 经过的时间
 */
double timer_elapsed_us(const Timer* timer);

/**
 * @brief 获取经过的时间 (毫秒)
 * @param timer 计时器指针
 * @return 经过的时间
 */
double timer_elapsed_ms(const Timer* timer);

/**
 * @brief 获取经过的时间 (秒)
 * @param timer 计时器指针
 * @return 经过的时间
 */
double timer_elapsed_sec(const Timer* timer);

/* ============================================================================
 * 作用域计时
 * ============================================================================ */

/**
 * @brief 作用域计时器结构
 */
typedef struct {
    Timer timer;
    const char* scope_name;
} ScopeTimer;

/**
 * @brief 初始化作用域计时器
 * @param st 作用域计时器指针
 * @param name 作用域名称
 */
void scope_timer_init(ScopeTimer* st, const char* name);

/**
 * @brief 结束作用域计时 (打印日志)
 * @param st 作用域计时器指针
 */
void scope_timer_end(ScopeTimer* st);

/* 便捷宏：自动作用域计时 */
#define SCOPE_TIMER(name) \
    ScopeTimer _scope_timer_##__LINE__; \
    scope_timer_init(&_scope_timer_##__LINE__, name); \
    __attribute__((cleanup(scope_timer_end))) ScopeTimer* _scope_timer_ptr_##__LINE__ = &_scope_timer_##__LINE__

/* ============================================================================
 * 多计时器管理
 * ============================================================================ */

#define MAX_TIMERS 32

/**
 * @brief 计时器管理器
 */
typedef struct {
    Timer timers[MAX_TIMERS];
    int count;
} TimerManager;

/**
 * @brief 初始化计时器管理器
 * @param mgr 管理器指针
 */
void timer_manager_init(TimerManager* mgr);

/**
 * @brief 获取或创建命名计时器
 * @param mgr 管理器指针
 * @param name 计时器名称
 * @return 计时器指针，失败返回 NULL
 */
Timer* timer_manager_get(TimerManager* mgr, const char* name);

/**
 * @brief 打印所有计时器统计
 * @param mgr 管理器指针
 */
void timer_manager_print(const TimerManager* mgr);

/**
 * @brief 获取计时器统计
 * @param timer 计时器指针
 * @param stats 统计输出
 */
void timer_get_stats(const Timer* timer, TimerStats* stats);

/* ============================================================================
 * 帧率计算
 * ============================================================================ */

/**
 * @brief 帧率计数器
 */
typedef struct {
    uint64_t start_time;
    size_t frame_count;
    double fps;
} FPSCounter;

/**
 * @brief 初始化帧率计数器
 * @param counter 计数器指针
 */
void fps_counter_init(FPSCounter* counter);

/**
 * @brief 记录一帧
 * @param counter 计数器指针
 */
void fps_counter_tick(FPSCounter* counter);

/**
 * @brief 获取当前帧率
 * @param counter 计数器指针
 * @return 帧率 (FPS)
 */
double fps_counter_get(const FPSCounter* counter);

#ifdef __cplusplus
}
#endif

#endif /* MYLLM_UTILS_TIMER_H */
