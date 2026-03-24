/**
 * @file timer.c
 * @brief 计时工具实现
 */

#include "utils/timer.h"
#include "utils/log.h"
#include <stdlib.h>
#include <string.h>
#include <stdio.h>

/* 平台特定头文件 */
#ifdef _WIN32
    #include <windows.h>
#else
    #include <time.h>
    #include <unistd.h>
#endif

/* ============================================================================
 * 基本计时函数
 * ============================================================================ */

uint64_t timer_now_ns(void) {
#ifdef _WIN32
    LARGE_INTEGER freq, counter;
    QueryPerformanceFrequency(&freq);
    QueryPerformanceCounter(&counter);
    return (uint64_t)((double)counter.QuadPart / (double)freq.QuadPart * 1e9);
#else
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (uint64_t)ts.tv_sec * 1000000000ULL + (uint64_t)ts.tv_nsec;
#endif
}

uint64_t timer_now_us(void) {
    return timer_now_ns() / 1000;
}

uint64_t timer_now_ms(void) {
    return timer_now_ns() / 1000000;
}

double timer_now_sec(void) {
    return (double)timer_now_ns() / 1e9;
}

/* ============================================================================
 * 计时器操作
 * ============================================================================ */

Timer* timer_new(const char* name) {
    Timer* timer = (Timer*)malloc(sizeof(Timer));
    if (!timer) return NULL;
    timer_init(timer, name);
    return timer;
}

void timer_free(Timer* timer) {
    free(timer);
}

void timer_init(Timer* timer, const char* name) {
    if (!timer) return;
    memset(timer, 0, sizeof(Timer));
    timer->name = name;
}

void timer_start(Timer* timer) {
    if (!timer) return;
    timer->start_ns = timer_now_ns();
}

uint64_t timer_stop(Timer* timer) {
    if (!timer) return 0;
    timer->end_ns = timer_now_ns();
    uint64_t elapsed = timer->end_ns - timer->start_ns;
    timer->total_ns += elapsed;
    timer->count++;
    return elapsed;
}

void timer_reset(Timer* timer) {
    if (!timer) return;
    timer->start_ns = 0;
    timer->end_ns = 0;
    timer->total_ns = 0;
    timer->count = 0;
}

uint64_t timer_elapsed_ns(const Timer* timer) {
    if (!timer) return 0;
    return timer->end_ns - timer->start_ns;
}

double timer_elapsed_us(const Timer* timer) {
    return (double)timer_elapsed_ns(timer) / 1000.0;
}

double timer_elapsed_ms(const Timer* timer) {
    return (double)timer_elapsed_ns(timer) / 1000000.0;
}

double timer_elapsed_sec(const Timer* timer) {
    return (double)timer_elapsed_ns(timer) / 1e9;
}

/* ============================================================================
 * 作用域计时
 * ============================================================================ */

void scope_timer_init(ScopeTimer* st, const char* name) {
    if (!st) return;
    st->scope_name = name;
    timer_init(&st->timer, name);
    timer_start(&st->timer);
}

void scope_timer_end(ScopeTimer* st) {
    if (!st) return;
    uint64_t elapsed = timer_stop(&st->timer);
    double ms = (double)elapsed / 1000000.0;
    LOG_INFO("[TIMER] %s: %.3f ms", st->scope_name, ms);
}

/* ============================================================================
 * 多计时器管理
 * ============================================================================ */

void timer_manager_init(TimerManager* mgr) {
    if (!mgr) return;
    memset(mgr, 0, sizeof(TimerManager));
}

Timer* timer_manager_get(TimerManager* mgr, const char* name) {
    if (!mgr || !name) return NULL;

    /* 查找现有计时器 */
    for (int i = 0; i < mgr->count; i++) {
        if (mgr->timers[i].name && strcmp(mgr->timers[i].name, name) == 0) {
            return &mgr->timers[i];
        }
    }

    /* 创建新计时器 */
    if (mgr->count < MAX_TIMERS) {
        Timer* timer = &mgr->timers[mgr->count++];
        timer_init(timer, name);
        return timer;
    }

    return NULL;
}

void timer_get_stats(const Timer* timer, TimerStats* stats) {
    if (!timer || !stats) return;

    stats->count = timer->count;
    stats->total_ms = (double)timer->total_ns / 1000000.0;
    stats->avg_ms = timer->count > 0 ? stats->total_ms / timer->count : 0.0;
    stats->min_ms = 0.0;  /* 需要单独跟踪 */
    stats->max_ms = 0.0;
}

void timer_manager_print(const TimerManager* mgr) {
    if (!mgr) return;

    printf("\n=== Timer Statistics ===\n");
    printf("%-20s %10s %10s %10s\n", "Name", "Total(ms)", "Avg(ms)", "Count");
    printf("%-20s %10s %10s %10s\n", "----", "---------", "-------", "-----");

    for (int i = 0; i < mgr->count; i++) {
        const Timer* t = &mgr->timers[i];
        double total_ms = (double)t->total_ns / 1000000.0;
        double avg_ms = t->count > 0 ? total_ms / t->count : 0.0;

        printf("%-20s %10.3f %10.3f %10zu\n",
               t->name ? t->name : "unnamed",
               total_ms, avg_ms, t->count);
    }
    printf("========================\n\n");
}

/* ============================================================================
 * 帧率计算
 * ============================================================================ */

void fps_counter_init(FPSCounter* counter) {
    if (!counter) return;
    memset(counter, 0, sizeof(FPSCounter));
    counter->start_time = timer_now_ns();
}

void fps_counter_tick(FPSCounter* counter) {
    if (!counter) return;
    counter->frame_count++;

    /* 每秒更新一次 FPS */
    uint64_t now = timer_now_ns();
    uint64_t elapsed = now - counter->start_time;

    if (elapsed >= 1000000000ULL) {  /* 1 秒 */
        counter->fps = (double)counter->frame_count * 1e9 / elapsed;
        counter->start_time = now;
        counter->frame_count = 0;
    }
}

double fps_counter_get(const FPSCounter* counter) {
    if (!counter) return 0.0;
    return counter->fps;
}
