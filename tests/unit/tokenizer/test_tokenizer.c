/**
 * @file test_tokenizer.c
 * @brief 分词器模块单元测试
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <sys/stat.h>
#include <fcntl.h>
#include <unistd.h>
#include "tokenizer/tokenizer.h"
#include "common/test_common.h"

/* ============================================================================
 * 测试数据目录
 * ============================================================================ */

#define TEST_DATA_DIR "/tmp/myllm_test_tokenizer"

/* ============================================================================
 * 测试辅助函数
 * ============================================================================ */

/**
 * @brief 创建简单的测试用 tokenizer.json
 */
static bool create_test_tokenizer_json(const char* path) {
    const char* json = "{"
        "\"version\":\"1.0\","
        "\"model\":{"
            "\"type\":\"Vocab\","
            "\"vocab\":{"
                "\"<unk>\":0,"
                "\"<pad>\":1,"
                "\"<s>\":2,"
                "\"</s>\":3,"
                "\"hello\":4,"
                "\"world\":5,"
                "\"the\":6,"
                "\"a\":7,"
                "\"is\":8,"
                "\"test\":9"
            "}"
        "},"
        "\"added_tokens\":["
            "{\"content\":\"<s>\",\"id\":2,\"special\":true},"
            "{\"content\":\"</s>\",\"id\":3,\"special\":true}"
        "],"
        "\"bos_token\":\"<s>\","
        "\"eos_token\":\"</s>\","
        "\"unk_token\":\"<unk>\","
        "\"pad_token\":\"<pad>\""
    "}";

    FILE* f = fopen(path, "w");
    if (!f) return false;

    bool success = (fprintf(f, "%s", json) > 0);
    fclose(f);
    return success;
}

/**
 * @brief 创建测试用的 tokenizer_config.json
 */
static bool create_test_tokenizer_config(const char* path) {
    const char* config = "{"
        "\"bos_token\":\"<s>\","
        "\"eos_token\":\"</s>\","
        "\"unk_token\":\"<unk>\","
        "\"pad_token\":\"<pad>\""
    "}";

    FILE* f = fopen(path, "w");
    if (!f) return false;

    bool success = (fprintf(f, "%s", config) > 0);
    fclose(f);
    return success;
}

/**
 * @brief 创建测试目录
 */
static bool setup_test_dir(void) {
    char cmd[512];
    snprintf(cmd, sizeof(cmd), "mkdir -p %s", TEST_DATA_DIR);
    system(cmd);

    char path[512];

    snprintf(path, sizeof(path), "%s/tokenizer.json", TEST_DATA_DIR);
    if (!create_test_tokenizer_json(path)) return false;

    snprintf(path, sizeof(path), "%s/tokenizer_config.json", TEST_DATA_DIR);
    if (!create_test_tokenizer_config(path)) return false;

    return true;
}

/**
 * @brief 清理测试目录
 */
static void cleanup_test_dir(void) {
    char cmd[512];
    snprintf(cmd, sizeof(cmd), "rm -rf %s", TEST_DATA_DIR);
    system(cmd);
}

/* ============================================================================
 * UTF-8 工具测试
 * ============================================================================ */

static void test_utf8_strlen_ascii(void) {
    ASSERT_EQ(utf8_strlen("hello"), 5, "ASCII string length");
    ASSERT_EQ(utf8_strlen(""), 0, "Empty string length");
    ASSERT_EQ(utf8_strlen("a"), 1, "Single char length");
    TEST_PASS();
}

static void test_utf8_strlen_unicode(void) {
    ASSERT_EQ(utf8_strlen("你好"), 2, "Chinese chars");
    ASSERT_EQ(utf8_strlen("世界"), 2, "More Chinese chars");
    ASSERT_EQ(utf8_strlen("hi你好"), 4, "Mixed ASCII and Chinese");
    TEST_PASS();
}

static void test_utf8_char_len(void) {
    ASSERT_EQ(utf8_char_len("a"), 1, "ASCII char");
    ASSERT_EQ(utf8_char_len("\xc4"), 2, "2-byte UTF-8");
    ASSERT_EQ(utf8_char_len("\xe4"), 3, "3-byte UTF-8");
    ASSERT_EQ(utf8_char_len("\xf0"), 4, "4-byte UTF-8");
    TEST_PASS();
}

/* ============================================================================
 * 生命周期测试
 * ============================================================================ */

static void test_tokenizer_new(void) {
    BPETokenizer* tokenizer = bpe_tokenizer_new();
    ASSERT_NOT_NULL(tokenizer, "Tokenizer should not be NULL");
    ASSERT_EQ(bpe_tokenizer_vocab_size(tokenizer), 0, "Vocab size should be 0");

    bpe_tokenizer_free(tokenizer);
    TEST_PASS();
}

static void test_tokenizer_free_null(void) {
    bpe_tokenizer_free(NULL);
    TEST_PASS();
}

static void test_tokenizer_from_file(void) {
    char path[512];
    snprintf(path, sizeof(path), "%s/tokenizer.json", TEST_DATA_DIR);

    BPETokenizer* tokenizer = bpe_tokenizer_from_file(path);
    ASSERT_NOT_NULL(tokenizer, "Tokenizer should not be NULL");
    ASSERT_TRUE(bpe_tokenizer_vocab_size(tokenizer) > 0, "Vocab size should be > 0");

    bpe_tokenizer_free(tokenizer);
    TEST_PASS();
}

static void test_tokenizer_from_file_invalid(void) {
    BPETokenizer* tokenizer = bpe_tokenizer_from_file("/nonexistent/tokenizer.json");
    ASSERT_NULL(tokenizer, "Should return NULL for invalid path");
    TEST_PASS();
}

static void test_tokenizer_from_dir(void) {
    BPETokenizer* tokenizer = bpe_tokenizer_from_dir(TEST_DATA_DIR);
    ASSERT_NOT_NULL(tokenizer, "Tokenizer should not be NULL");
    ASSERT_TRUE(bpe_tokenizer_vocab_size(tokenizer) > 0, "Vocab size should be > 0");

    bpe_tokenizer_free(tokenizer);
    TEST_PASS();
}

/* ============================================================================
 * 词表操作测试
 * ============================================================================ */

static void test_tokenizer_vocab_size(void) {
    char path[512];
    snprintf(path, sizeof(path), "%s/tokenizer.json", TEST_DATA_DIR);

    BPETokenizer* tokenizer = bpe_tokenizer_from_file(path);
    ASSERT_NOT_NULL(tokenizer, "Tokenizer should not be NULL");

    size_t size = bpe_tokenizer_vocab_size(tokenizer);
    ASSERT_TRUE(size > 0, "Vocab size should be > 0");

    bpe_tokenizer_free(tokenizer);
    TEST_PASS();
}

static void test_tokenizer_token_to_id(void) {
    char path[512];
    snprintf(path, sizeof(path), "%s/tokenizer.json", TEST_DATA_DIR);

    BPETokenizer* tokenizer = bpe_tokenizer_from_file(path);
    ASSERT_NOT_NULL(tokenizer, "Tokenizer should not be NULL");

    int32_t id = bpe_tokenizer_token_to_id(tokenizer, "hello");
    ASSERT_EQ(id, 4, "Token 'hello' should have id 4");

    id = bpe_tokenizer_token_to_id(tokenizer, "world");
    ASSERT_EQ(id, 5, "Token 'world' should have id 5");

    id = bpe_tokenizer_token_to_id(tokenizer, "nonexistent");
    ASSERT_EQ(id, bpe_tokenizer_unk_id(tokenizer), "Unknown token should return UNK id");

    bpe_tokenizer_free(tokenizer);
    TEST_PASS();
}

static void test_tokenizer_id_to_token(void) {
    char path[512];
    snprintf(path, sizeof(path), "%s/tokenizer.json", TEST_DATA_DIR);

    BPETokenizer* tokenizer = bpe_tokenizer_from_file(path);
    ASSERT_NOT_NULL(tokenizer, "Tokenizer should not be NULL");

    const char* token = bpe_tokenizer_id_to_token(tokenizer, 4);
    ASSERT_NOT_NULL(token, "Token should not be NULL");
    ASSERT_TRUE(strcmp(token, "hello") == 0, "ID 4 should be 'hello'");

    token = bpe_tokenizer_id_to_token(tokenizer, 9999);
    ASSERT_NULL(token, "Invalid ID should return NULL");

    bpe_tokenizer_free(tokenizer);
    TEST_PASS();
}

/* ============================================================================
 * 特殊 Token 测试
 * ============================================================================ */

static void test_tokenizer_special_tokens(void) {
    char path[512];
    snprintf(path, sizeof(path), "%s/tokenizer.json", TEST_DATA_DIR);

    BPETokenizer* tokenizer = bpe_tokenizer_from_file(path);
    ASSERT_NOT_NULL(tokenizer, "Tokenizer should not be NULL");

    ASSERT_EQ(bpe_tokenizer_bos_id(tokenizer), 2, "BOS id should be 2");
    ASSERT_EQ(bpe_tokenizer_eos_id(tokenizer), 3, "EOS id should be 3");
    ASSERT_EQ(bpe_tokenizer_pad_id(tokenizer), 1, "PAD id should be 1");
    ASSERT_EQ(bpe_tokenizer_unk_id(tokenizer), 0, "UNK id should be 0");

    const char* bos = bpe_tokenizer_bos_token(tokenizer);
    ASSERT_NOT_NULL(bos, "BOS token should not be NULL");
    ASSERT_TRUE(strcmp(bos, "<s>") == 0, "BOS token should be '<s>'");

    const char* eos = bpe_tokenizer_eos_token(tokenizer);
    ASSERT_NOT_NULL(eos, "EOS token should not be NULL");
    ASSERT_TRUE(strcmp(eos, "</s>") == 0, "EOS token should be '</s>'");

    bpe_tokenizer_free(tokenizer);
    TEST_PASS();
}

static void test_tokenizer_is_special_token(void) {
    char path[512];
    snprintf(path, sizeof(path), "%s/tokenizer.json", TEST_DATA_DIR);

    BPETokenizer* tokenizer = bpe_tokenizer_from_file(path);
    ASSERT_NOT_NULL(tokenizer, "Tokenizer should not be NULL");

    ASSERT_TRUE(bpe_tokenizer_is_special_token(tokenizer, 2), "BOS should be special");
    ASSERT_TRUE(bpe_tokenizer_is_special_token(tokenizer, 3), "EOS should be special");
    ASSERT_TRUE(bpe_tokenizer_is_special_token(tokenizer, 0), "UNK should be special");
    ASSERT_FALSE(bpe_tokenizer_is_special_token(tokenizer, 4), "hello should not be special");

    bpe_tokenizer_free(tokenizer);
    TEST_PASS();
}

/* ============================================================================
 * 编码测试
 * ============================================================================ */

static void test_tokenizer_encode_simple(void) {
    char path[512];
    snprintf(path, sizeof(path), "%s/tokenizer.json", TEST_DATA_DIR);

    BPETokenizer* tokenizer = bpe_tokenizer_from_file(path);
    ASSERT_NOT_NULL(tokenizer, "Tokenizer should not be NULL");

    int32_t ids[64];
    int count = bpe_tokenizer_encode(tokenizer, "hello", false, ids, 64);
    ASSERT_TRUE(count > 0, "Should encode 'hello'");

    bpe_tokenizer_free(tokenizer);
    TEST_PASS();
}

static void test_tokenizer_encode_with_special(void) {
    char path[512];
    snprintf(path, sizeof(path), "%s/tokenizer.json", TEST_DATA_DIR);

    BPETokenizer* tokenizer = bpe_tokenizer_from_file(path);
    ASSERT_NOT_NULL(tokenizer, "Tokenizer should not be NULL");

    int32_t ids[64];
    int count = bpe_tokenizer_encode(tokenizer, "hello", true, ids, 64);
    ASSERT_TRUE(count >= 2, "Should include BOS and EOS");

    /* 第一个应该是 BOS */
    ASSERT_EQ(ids[0], bpe_tokenizer_bos_id(tokenizer), "First should be BOS");

    bpe_tokenizer_free(tokenizer);
    TEST_PASS();
}

/* ============================================================================
 * 解码测试
 * ============================================================================ */

static void test_tokenizer_decode_simple(void) {
    char path[512];
    snprintf(path, sizeof(path), "%s/tokenizer.json", TEST_DATA_DIR);

    BPETokenizer* tokenizer = bpe_tokenizer_from_file(path);
    ASSERT_NOT_NULL(tokenizer, "Tokenizer should not be NULL");

    int32_t ids[] = {4};  /* "hello" */
    char out[256];
    int count = bpe_tokenizer_decode(tokenizer, ids, 1, false, out, sizeof(out));
    ASSERT_TRUE(count > 0, "Should decode");

    bpe_tokenizer_free(tokenizer);
    TEST_PASS();
}

static void test_tokenizer_decode_skip_special(void) {
    char path[512];
    snprintf(path, sizeof(path), "%s/tokenizer.json", TEST_DATA_DIR);

    BPETokenizer* tokenizer = bpe_tokenizer_from_file(path);
    ASSERT_NOT_NULL(tokenizer, "Tokenizer should not be NULL");

    int32_t ids[] = {2, 4, 3};  /* <s>, hello, </s> */
    char out[256];
    int count = bpe_tokenizer_decode(tokenizer, ids, 3, true, out, sizeof(out));
    ASSERT_TRUE(count > 0, "Should decode");

    /* 不应包含特殊 token */
    ASSERT_TRUE(strstr(out, "<s>") == NULL, "Should not contain <s>");
    ASSERT_TRUE(strstr(out, "</s>") == NULL, "Should not contain </s>");

    bpe_tokenizer_free(tokenizer);
    TEST_PASS();
}

/* ============================================================================
 * 边界情况测试
 * ============================================================================ */

static void test_tokenizer_null_handling(void) {
    char path[512];
    snprintf(path, sizeof(path), "%s/tokenizer.json", TEST_DATA_DIR);

    BPETokenizer* tokenizer = bpe_tokenizer_from_file(path);
    ASSERT_NOT_NULL(tokenizer, "Tokenizer should not be NULL");

    int32_t ids[64];
    int count = bpe_tokenizer_encode(tokenizer, NULL, false, ids, 64);
    ASSERT_EQ(count, -1, "NULL text should return -1");

    count = bpe_tokenizer_encode(tokenizer, "hello", false, NULL, 0);
    ASSERT_EQ(count, -1, "NULL ids should return -1");

    char out[256];
    count = bpe_tokenizer_decode(tokenizer, NULL, 0, false, out, sizeof(out));
    ASSERT_EQ(count, -1, "NULL ids should return -1");

    bpe_tokenizer_free(tokenizer);
    TEST_PASS();
}

static void test_tokenizer_small_buffer(void) {
    char path[512];
    snprintf(path, sizeof(path), "%s/tokenizer.json", TEST_DATA_DIR);

    BPETokenizer* tokenizer = bpe_tokenizer_from_file(path);
    ASSERT_NOT_NULL(tokenizer, "Tokenizer should not be NULL");

    int32_t ids[1];
    int count = bpe_tokenizer_encode(tokenizer, "hello world test", true, ids, 1);
    ASSERT_EQ(count, 1, "Should only fit 1 token");

    bpe_tokenizer_free(tokenizer);
    TEST_PASS();
}

/* ============================================================================
 * 主函数
 * ============================================================================ */

int main(void) {
    printf("========================================\n");
    printf("Tokenizer Module Unit Tests\n");
    printf("========================================\n\n");

    /* 设置测试数据 */
    if (!setup_test_dir()) {
        fprintf(stderr, "Failed to setup test data directory\n");
        return 1;
    }

    /* UTF-8 工具测试 */
    printf("[UTF-8 Utility Tests]\n");
    RUN_TEST(test_utf8_strlen_ascii);
    RUN_TEST(test_utf8_strlen_unicode);
    RUN_TEST(test_utf8_char_len);
    printf("\n");

    /* 生命周期测试 */
    printf("[Lifecycle Tests]\n");
    RUN_TEST(test_tokenizer_new);
    RUN_TEST(test_tokenizer_free_null);
    RUN_TEST(test_tokenizer_from_file);
    RUN_TEST(test_tokenizer_from_file_invalid);
    RUN_TEST(test_tokenizer_from_dir);
    printf("\n");

    /* 词表操作测试 */
    printf("[Vocabulary Tests]\n");
    RUN_TEST(test_tokenizer_vocab_size);
    RUN_TEST(test_tokenizer_token_to_id);
    RUN_TEST(test_tokenizer_id_to_token);
    printf("\n");

    /* 特殊 Token 测试 */
    printf("[Special Token Tests]\n");
    RUN_TEST(test_tokenizer_special_tokens);
    RUN_TEST(test_tokenizer_is_special_token);
    printf("\n");

    /* 编码测试 */
    printf("[Encoding Tests]\n");
    RUN_TEST(test_tokenizer_encode_simple);
    RUN_TEST(test_tokenizer_encode_with_special);
    printf("\n");

    /* 解码测试 */
    printf("[Decoding Tests]\n");
    RUN_TEST(test_tokenizer_decode_simple);
    RUN_TEST(test_tokenizer_decode_skip_special);
    printf("\n");

    /* 边界情况测试 */
    printf("[Edge Case Tests]\n");
    RUN_TEST(test_tokenizer_null_handling);
    RUN_TEST(test_tokenizer_small_buffer);
    printf("\n");

    /* 清理测试数据 */
    cleanup_test_dir();

    TEST_SUMMARY();
    return TEST_RETURN();
}
