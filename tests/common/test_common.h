/**
 * @file test_common.h
 * @brief 测试通用工具和宏
 */

#ifndef MYLLM_TEST_COMMON_H
#define MYLLM_TEST_COMMON_H

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

/* 测试计数器 */
static int tests_run = 0;
static int tests_passed = 0;
static int tests_failed = 0;

/* 测试宏 */
#define TEST_START(name) \
    do { \
        printf("  [TEST] %s... ", name); \
        fflush(stdout); \
    } while(0)

#define TEST_PASS() \
    do { \
        printf("PASS\n"); \
        tests_passed++; \
    } while(0)

#define TEST_FAIL(msg) \
    do { \
        printf("FAIL: %s\n", msg); \
        tests_failed++; \
    } while(0)

#define ASSERT_TRUE(cond, msg) \
    do { \
        if (!(cond)) { \
            TEST_FAIL(msg); \
            return; \
        } \
    } while(0)

#define ASSERT_FALSE(cond, msg) ASSERT_TRUE(!(cond), msg)

#define ASSERT_EQ(a, b, msg) \
    do { \
        if ((a) != (b)) { \
            char _buf[256]; \
            snprintf(_buf, sizeof(_buf), "%s (expected %zd, got %zd)", msg, (size_t)(b), (size_t)(a)); \
            TEST_FAIL(_buf); \
            return; \
        } \
    } while(0)

#define ASSERT_NE(a, b, msg) \
    do { \
        if ((a) == (b)) { \
            char _buf[256]; \
            snprintf(_buf, sizeof(_buf), "%s (values are equal: %zd)", msg, (size_t)(a)); \
            TEST_FAIL(_buf); \
            return; \
        } \
    } while(0)

#define ASSERT_NOT_NULL(ptr, msg) \
    do { \
        if ((ptr) == NULL) { \
            TEST_FAIL(msg); \
            return; \
        } \
    } while(0)

#define ASSERT_NULL(ptr, msg) \
    do { \
        if ((ptr) != NULL) { \
            TEST_FAIL(msg); \
            return; \
        } \
    } while(0)

#define ASSERT_FLOAT_EQ(a, b, eps, msg) \
    do { \
        if (fabs((a) - (b)) > (eps)) { \
            char _buf[256]; \
            snprintf(_buf, sizeof(_buf), "%s (expected %f, got %f)", msg, (double)(b), (double)(a)); \
            TEST_FAIL(_buf); \
            return; \
        } \
    } while(0)

/* 测试套件宏 */
#define RUN_TEST(test_func) \
    do { \
        tests_run++; \
        TEST_START(#test_func); \
        test_func(); \
    } while(0)

#define TEST_SUMMARY() \
    do { \
        printf("\n========================================\n"); \
        printf("Tests run: %d\n", tests_run); \
        printf("Passed: %d\n", tests_passed); \
        printf("Failed: %d\n", tests_failed); \
        printf("========================================\n"); \
    } while(0)

#define TEST_RETURN() (tests_failed > 0 ? 1 : 0)

#endif /* MYLLM_TEST_COMMON_H */
