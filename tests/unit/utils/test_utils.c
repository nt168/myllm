/**
 * @file test_utils.c
 * @brief 工具模块单元测试
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <unistd.h>
#include "utils/utils.h"
#include "common/test_common.h"

/* ============================================================================
 * 日志测试
 * ============================================================================ */

static void test_log_init_shutdown(void) {
    LogConfig config = {
        .min_level = LOG_LEVEL_DEBUG,
        .targets = LOG_TARGET_STDERR,
        .show_timestamp = 1,
        .use_colors = 0
    };

    int ret = log_init(&config);
    ASSERT_EQ(ret, 0, "log_init should succeed");

    log_shutdown();
    TEST_PASS();
}

static void test_log_level(void) {
    log_init(NULL);

    log_set_level(LOG_LEVEL_WARN);
    ASSERT_EQ(log_get_level(), LOG_LEVEL_WARN, "Level should be WARN");

    log_set_level(LOG_LEVEL_DEBUG);
    ASSERT_EQ(log_get_level(), LOG_LEVEL_DEBUG, "Level should be DEBUG");

    log_shutdown();
    TEST_PASS();
}

static void test_log_level_name(void) {
    ASSERT_EQ(strcmp(log_level_name(LOG_LEVEL_TRACE), "TRACE"), 0, "TRACE name");
    ASSERT_EQ(strcmp(log_level_name(LOG_LEVEL_DEBUG), "DEBUG"), 0, "DEBUG name");
    ASSERT_EQ(strcmp(log_level_name(LOG_LEVEL_INFO), "INFO"), 0, "INFO name");
    ASSERT_EQ(strcmp(log_level_name(LOG_LEVEL_WARN), "WARN"), 0, "WARN name");
    ASSERT_EQ(strcmp(log_level_name(LOG_LEVEL_ERROR), "ERROR"), 0, "ERROR name");
    ASSERT_EQ(strcmp(log_level_name(LOG_LEVEL_FATAL), "FATAL"), 0, "FATAL name");

    TEST_PASS();
}

static void test_log_level_from_name(void) {
    ASSERT_EQ(log_level_from_name("TRACE"), LOG_LEVEL_TRACE, "Parse TRACE");
    ASSERT_EQ(log_level_from_name("debug"), LOG_LEVEL_DEBUG, "Parse debug");
    ASSERT_EQ(log_level_from_name("INFO"), LOG_LEVEL_INFO, "Parse INFO");
    ASSERT_EQ(log_level_from_name("Warn"), LOG_LEVEL_WARN, "Parse Warn");
    ASSERT_EQ(log_level_from_name("invalid"), LOG_LEVEL_INFO, "Invalid defaults to INFO");
    ASSERT_EQ(log_level_from_name(NULL), LOG_LEVEL_INFO, "NULL defaults to INFO");

    TEST_PASS();
}

static void test_log_write(void) {
    LogConfig config = {
        .min_level = LOG_LEVEL_DEBUG,
        .targets = LOG_TARGET_STDOUT,
        .show_timestamp = 0,
        .show_file_line = 0,
        .use_colors = 0
    };
    log_init(&config);

    /* 这些不应该崩溃 */
    LOG_DEBUG("Test debug message: %d", 42);
    LOG_INFO("Test info message");
    LOG_WARN("Test warning message");

    log_shutdown();
    TEST_PASS();
}

/* ============================================================================
 * 计时器测试
 * ============================================================================ */

static void test_timer_now(void) {
    uint64_t ns = timer_now_ns();
    uint64_t us = timer_now_us();
    uint64_t ms = timer_now_ms();
    double sec = timer_now_sec();

    ASSERT_TRUE(ns > 0, "ns should be positive");
    ASSERT_TRUE(us > 0, "us should be positive");
    ASSERT_TRUE(ms > 0, "ms should be positive");
    ASSERT_TRUE(sec > 0.0, "sec should be positive");

    /* 验证单位转换 */
    ASSERT_TRUE(ns / 1000 >= us - 1 && ns / 1000 <= us + 1, "ns to us conversion");
    ASSERT_TRUE(us / 1000 >= ms - 1 && us / 1000 <= ms + 1, "us to ms conversion");

    TEST_PASS();
}

static void test_timer_basic(void) {
    Timer timer;
    timer_init(&timer, "test");

    timer_start(&timer);
    usleep(10000);  /* 10 ms */
    uint64_t elapsed = timer_stop(&timer);

    /* 应该至少 5ms，不超过 100ms */
    ASSERT_TRUE(elapsed > 5000000ULL, "Elapsed should be > 5ms");
    ASSERT_TRUE(elapsed < 100000000ULL, "Elapsed should be < 100ms");

    TEST_PASS();
}

static void test_timer_elapsed(void) {
    Timer timer;
    timer_init(&timer, NULL);

    timer_start(&timer);
    usleep(5000);  /* 5 ms */
    timer_stop(&timer);

    double ms = timer_elapsed_ms(&timer);
    ASSERT_TRUE(ms > 4.0 && ms < 50.0, "Elapsed ms should be around 5");

    TEST_PASS();
}

static void test_timer_reset(void) {
    Timer timer;
    timer_init(&timer, NULL);

    timer_start(&timer);
    usleep(5000);
    timer_stop(&timer);

    ASSERT_TRUE(timer.count == 1, "Count should be 1");

    timer_reset(&timer);
    ASSERT_EQ(timer.count, 0, "Count should be 0 after reset");
    ASSERT_EQ(timer.total_ns, 0, "Total should be 0 after reset");

    TEST_PASS();
}

static void test_timer_new_free(void) {
    Timer* timer = timer_new("dynamic");
    ASSERT_NOT_NULL(timer, "timer_new should return non-NULL");

    timer_start(timer);
    usleep(1000);
    timer_stop(timer);

    timer_free(timer);
    TEST_PASS();
}

static void test_timer_manager(void) {
    TimerManager mgr;
    timer_manager_init(&mgr);

    Timer* t1 = timer_manager_get(&mgr, "timer1");
    ASSERT_NOT_NULL(t1, "Should get timer1");

    Timer* t2 = timer_manager_get(&mgr, "timer2");
    ASSERT_NOT_NULL(t2, "Should get timer2");

    Timer* t1_again = timer_manager_get(&mgr, "timer1");
    ASSERT_EQ(t1, t1_again, "Should return same timer for same name");

    TEST_PASS();
}

static void test_fps_counter(void) {
    FPSCounter counter;
    fps_counter_init(&counter);

    ASSERT_EQ(counter.frame_count, 0, "Initial frame count should be 0");

    fps_counter_tick(&counter);
    fps_counter_tick(&counter);
    fps_counter_tick(&counter);

    ASSERT_EQ(counter.frame_count, 3, "Frame count should be 3");

    TEST_PASS();
}

/* ============================================================================
 * 错误处理测试
 * ============================================================================ */

static void test_error_message(void) {
    ASSERT_EQ(strcmp(error_message(MYLLM_OK), "Success"), 0, "OK message");
    ASSERT_EQ(strcmp(error_message(MYLLM_ERROR_INVALID_ARG), "Invalid argument"), 0, "Invalid arg message");
    ASSERT_EQ(strcmp(error_message(MYLLM_ERROR_OUT_OF_MEMORY), "Out of memory"), 0, "OOM message");
    ASSERT_EQ(strcmp(error_message(MYLLM_ERROR_CACHE_OVERFLOW), "Cache overflow"), 0, "Cache overflow message");

    TEST_PASS();
}

static void test_error_name(void) {
    ASSERT_EQ(strcmp(error_name(MYLLM_OK), "OK"), 0, "OK name");
    ASSERT_EQ(strcmp(error_name(MYLLM_ERROR_NULL_POINTER), "ERROR_NULL_POINTER"), 0, "Null pointer name");
    ASSERT_EQ(strcmp(error_name(MYLLM_ERROR_TENSOR_SHAPE), "ERROR_TENSOR_SHAPE"), 0, "Tensor shape name");

    TEST_PASS();
}

static void test_error_context(void) {
    error_set_context(MYLLM_ERROR_INVALID_ARG, "test.c", 42, "test_func", "Test error: %d", 123);

    const ErrorContext* ctx = error_get_context();
    ASSERT_NOT_NULL(ctx, "Context should not be NULL");
    ASSERT_EQ(ctx->code, MYLLM_ERROR_INVALID_ARG, "Error code should match");
    ASSERT_EQ(ctx->line, 42, "Line should match");
    ASSERT_TRUE(strstr(ctx->message, "123") != NULL, "Message should contain formatted value");

    error_clear_context();
    ctx = error_get_context();
    ASSERT_EQ(ctx->code, 0, "Code should be cleared");

    TEST_PASS();
}

/* ============================================================================
 * 内存工具测试
 * ============================================================================ */

static void test_memory_tracking(void) {
    memory_tracking_enable();
    ASSERT_TRUE(memory_tracking_is_enabled(), "Tracking should be enabled");

    memory_reset_stats();
    MemoryStats stats;
    memory_get_stats(&stats);

    ASSERT_EQ(stats.current_bytes, 0, "Initial bytes should be 0");
    ASSERT_EQ(stats.total_allocs, 0, "Initial allocs should be 0");

    TEST_PASS();
}

static void test_memory_tracked_malloc_free(void) {
    memory_tracking_enable();
    memory_reset_stats();

    void* ptr = MYLLM_MALLOC(1024);
    ASSERT_NOT_NULL(ptr, "malloc should succeed");

    MemoryStats stats;
    memory_get_stats(&stats);
    ASSERT_EQ(stats.active_allocs, 1, "Should have 1 active alloc");
    ASSERT_TRUE(stats.current_bytes >= 1024, "Should have at least 1024 bytes");

    MYLLM_FREE(ptr);

    memory_get_stats(&stats);
    ASSERT_EQ(stats.active_allocs, 0, "Should have 0 active allocs");
    ASSERT_EQ(stats.current_bytes, 0, "Current bytes should be 0");

    TEST_PASS();
}

static void test_memory_tracked_calloc(void) {
    memory_tracking_enable();
    memory_reset_stats();

    int* arr = (int*)MYLLM_CALLOC(10, sizeof(int));
    ASSERT_NOT_NULL(arr, "calloc should succeed");

    /* 验证清零 */
    for (int i = 0; i < 10; i++) {
        ASSERT_EQ(arr[i], 0, "Memory should be zeroed");
    }

    MYLLM_FREE(arr);
    TEST_PASS();
}

static void test_memory_tracked_realloc(void) {
    memory_tracking_enable();
    memory_reset_stats();

    char* ptr = (char*)MYLLM_MALLOC(10);
    ASSERT_NOT_NULL(ptr, "malloc should succeed");
    strcpy(ptr, "hello");

    ptr = (char*)MYLLM_REALLOC(ptr, 100);
    ASSERT_NOT_NULL(ptr, "realloc should succeed");
    ASSERT_EQ(strcmp(ptr, "hello"), 0, "Data should be preserved");

    MYLLM_FREE(ptr);
    TEST_PASS();
}

static void test_memory_aligned_alloc(void) {
    void* ptr = memory_aligned_alloc(64, 256);
    ASSERT_NOT_NULL(ptr, "aligned_alloc should succeed");
    ASSERT_TRUE(memory_is_aligned(ptr, 64), "Should be 64-byte aligned");

    memory_aligned_free(ptr);
    TEST_PASS();
}

static void test_memory_align_size(void) {
    ASSERT_EQ(memory_align_size(1, 16), 16, "1 aligned to 16");
    ASSERT_EQ(memory_align_size(16, 16), 16, "16 aligned to 16");
    ASSERT_EQ(memory_align_size(17, 16), 32, "17 aligned to 16");
    ASSERT_EQ(memory_align_size(100, 64), 128, "100 aligned to 64");

    TEST_PASS();
}

static void test_memory_format_size(void) {
    char buf[32];

    memory_format_size(512, buf, sizeof(buf));
    ASSERT_TRUE(strstr(buf, "512") != NULL, "Should contain 512");

    memory_format_size(1024, buf, sizeof(buf));
    ASSERT_TRUE(strstr(buf, "KB") != NULL, "Should use KB");

    memory_format_size(1048576, buf, sizeof(buf));
    ASSERT_TRUE(strstr(buf, "MB") != NULL, "Should use MB");

    TEST_PASS();
}

static void test_memory_safe_copy(void) {
    char src[] = "hello world";
    char dst[20];

    int ret = memory_safe_copy(dst, src, strlen(src) + 1);
    ASSERT_EQ(ret, 0, "copy should succeed");
    ASSERT_EQ(strcmp(dst, src), 0, "Data should match");

    TEST_PASS();
}

/* ============================================================================
 * 工具模块整体测试
 * ============================================================================ */

static void test_utils_version(void) {
    const char* version = utils_version();
    ASSERT_NOT_NULL(version, "Version should not be NULL");
    ASSERT_TRUE(strlen(version) > 0, "Version should not be empty");

    TEST_PASS();
}

static void test_utils_init_shutdown(void) {
    int ret = utils_init();
    ASSERT_EQ(ret, 0, "utils_init should succeed");

    utils_shutdown();
    TEST_PASS();
}

/* ============================================================================
 * 主函数
 * ============================================================================ */

int main(void) {
    printf("========================================\n");
    printf("Utils Module Unit Tests\n");
    printf("========================================\n\n");

    /* 日志测试 */
    printf("[Log Tests]\n");
    RUN_TEST(test_log_init_shutdown);
    RUN_TEST(test_log_level);
    RUN_TEST(test_log_level_name);
    RUN_TEST(test_log_level_from_name);
    RUN_TEST(test_log_write);
    printf("\n");

    /* 计时器测试 */
    printf("[Timer Tests]\n");
    RUN_TEST(test_timer_now);
    RUN_TEST(test_timer_basic);
    RUN_TEST(test_timer_elapsed);
    RUN_TEST(test_timer_reset);
    RUN_TEST(test_timer_new_free);
    RUN_TEST(test_timer_manager);
    RUN_TEST(test_fps_counter);
    printf("\n");

    /* 错误处理测试 */
    printf("[Error Tests]\n");
    RUN_TEST(test_error_message);
    RUN_TEST(test_error_name);
    RUN_TEST(test_error_context);
    printf("\n");

    /* 内存工具测试 */
    printf("[Memory Tests]\n");
    RUN_TEST(test_memory_tracking);
    RUN_TEST(test_memory_tracked_malloc_free);
    RUN_TEST(test_memory_tracked_calloc);
    RUN_TEST(test_memory_tracked_realloc);
    RUN_TEST(test_memory_aligned_alloc);
    RUN_TEST(test_memory_align_size);
    RUN_TEST(test_memory_format_size);
    RUN_TEST(test_memory_safe_copy);
    printf("\n");

    /* 整体测试 */
    printf("[Utils Tests]\n");
    RUN_TEST(test_utils_version);
    RUN_TEST(test_utils_init_shutdown);
    printf("\n");

    TEST_SUMMARY();
    return TEST_RETURN();
}
