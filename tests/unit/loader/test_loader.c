/**
 * @file test_loader.c
 * @brief 加载器模块单元测试
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <sys/stat.h>
#include <fcntl.h>
#include <unistd.h>
#include "loader/safetensors.h"
#include "loader/loader.h"
#include "common/test_common.h"

/* ============================================================================
 * 测试数据目录
 * ============================================================================ */

#define TEST_DATA_DIR "/tmp/myllm_test_loader"

/* ============================================================================
 * 测试辅助函数
 * ============================================================================ */

/**
 * @brief 创建测试用的 SafeTensors 文件
 */
static bool create_test_safetensors(const char* path) {
    /* 简化的 SafeTensors 格式:
     * - 8 字节: JSON 元数据长度
     * - JSON 元数据
     * - 张量数据
     */

    /* JSON 元数据 */
    const char* json = "{"
        "\"test_tensor\":{"
        "\"dtype\":\"F32\","
        "\"shape\":[2,3],"
        "\"data_offsets\":[0,24]"
        "},"
        "\"test_vector\":{"
        "\"dtype\":\"F32\","
        "\"shape\":[4],"
        "\"data_offsets\":[24,40]"
        "}"
        "}";

    size_t json_len = strlen(json);
    uint64_t header = (uint64_t)json_len;

    /* 计算总大小: 8 + JSON + 数据 */
    size_t data_size = 40; /* 10 个 float32 */
    size_t total_size = 8 + json_len + data_size;

    uint8_t* buffer = (uint8_t*)malloc(total_size);
    if (!buffer) return false;

    /* 写入头部 */
    memcpy(buffer, &header, 8);

    /* 写入 JSON */
    memcpy(buffer + 8, json, json_len);

    /* 写入数据 */
    float* data = (float*)(buffer + 8 + json_len);
    /* test_tensor: [[0,1,2],[3,4,5]] */
    data[0] = 0.0f; data[1] = 1.0f; data[2] = 2.0f;
    data[3] = 3.0f; data[4] = 4.0f; data[5] = 5.0f;
    /* test_vector: [10,20,30,40] */
    data[6] = 10.0f; data[7] = 20.0f; data[8] = 30.0f; data[9] = 40.0f;

    /* 写入文件 */
    FILE* f = fopen(path, "wb");
    if (!f) {
        free(buffer);
        return false;
    }

    bool success = (fwrite(buffer, 1, total_size, f) == total_size);
    fclose(f);
    free(buffer);

    return success;
}

/**
 * @brief 创建测试用的 config.json
 */
static bool create_test_config(const char* path) {
    const char* config = "{"
        "\"architectures\":[\"LlamaForCausalLM\"],"
        "\"model_type\":\"llama\","
        "\"hidden_size\":256,"
        "\"intermediate_size\":512,"
        "\"num_attention_heads\":8,"
        "\"num_hidden_layers\":4,"
        "\"vocab_size\":1000,"
        "\"num_key_value_heads\":8,"
        "\"max_position_embeddings\":512,"
        "\"rope_theta\":10000.0,"
        "\"rms_norm_eps\":1e-6,"
        "\"torch_dtype\":\"float32\","
        "\"tie_word_embeddings\":false"
        "}";

    FILE* f = fopen(path, "w");
    if (!f) return false;

    bool success = (fprintf(f, "%s", config) > 0);
    fclose(f);
    return success;
}

/**
 * @brief 创建测试用的模型目录
 */
static bool setup_test_model_dir(void) {
    char cmd[512];
    snprintf(cmd, sizeof(cmd), "mkdir -p %s", TEST_DATA_DIR);
    system(cmd);

    char path[512];

    /* 创建 config.json */
    snprintf(path, sizeof(path), "%s/config.json", TEST_DATA_DIR);
    if (!create_test_config(path)) return false;

    /* 创建 model.safetensors */
    snprintf(path, sizeof(path), "%s/model.safetensors", TEST_DATA_DIR);
    if (!create_test_safetensors(path)) return false;

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
 * SafeTensors 数据类型测试
 * ============================================================================ */

static void test_safetensors_dtype_size(void) {
    ASSERT_EQ(safetensors_dtype_size(ST_F32), 4, "F32 size should be 4");
    ASSERT_EQ(safetensors_dtype_size(ST_F16), 2, "F16 size should be 2");
    ASSERT_EQ(safetensors_dtype_size(ST_BF16), 2, "BF16 size should be 2");
    ASSERT_EQ(safetensors_dtype_size(ST_I32), 4, "I32 size should be 4");
    ASSERT_EQ(safetensors_dtype_size(ST_I64), 8, "I64 size should be 8");
    ASSERT_EQ(safetensors_dtype_size(ST_I8), 1, "I8 size should be 1");
    ASSERT_EQ(safetensors_dtype_size(ST_U8), 1, "U8 size should be 1");

    TEST_PASS();
}

static void test_safetensors_dtype_from_string(void) {
    ASSERT_EQ(safetensors_dtype_from_string("F32"), ST_F32, "F32 string");
    ASSERT_EQ(safetensors_dtype_from_string("F16"), ST_F16, "F16 string");
    ASSERT_EQ(safetensors_dtype_from_string("BF16"), ST_BF16, "BF16 string");
    ASSERT_EQ(safetensors_dtype_from_string("I32"), ST_I32, "I32 string");
    ASSERT_EQ(safetensors_dtype_from_string("I64"), ST_I64, "I64 string");
    ASSERT_EQ(safetensors_dtype_from_string("UNKNOWN"), ST_UNKNOWN, "Unknown string");

    TEST_PASS();
}

static void test_safetensors_dtype_to_model(void) {
    ASSERT_EQ(safetensors_dtype_to_model(ST_F32), DTYPE_F32, "F32 to model");
    ASSERT_EQ(safetensors_dtype_to_model(ST_F16), DTYPE_F16, "F16 to model");
    ASSERT_EQ(safetensors_dtype_to_model(ST_BF16), DTYPE_BF16, "BF16 to model");
    ASSERT_EQ(safetensors_dtype_to_model(ST_UNKNOWN), DTYPE_F32, "Unknown defaults to F32");

    TEST_PASS();
}

/* ============================================================================
 * SafeTensors 加载测试
 * ============================================================================ */

static void test_safetensors_new(void) {
    char path[512];
    snprintf(path, sizeof(path), "%s/model.safetensors", TEST_DATA_DIR);

    SafeTensorsLoader* loader = safetensors_new(path);
    ASSERT_NOT_NULL(loader, "Loader should not be NULL");

    safetensors_free(loader);
    TEST_PASS();
}

static void test_safetensors_new_invalid(void) {
    SafeTensorsLoader* loader = safetensors_new("/nonexistent/path.safetensors");
    ASSERT_NULL(loader, "Should return NULL for invalid path");

    TEST_PASS();
}

static void test_safetensors_num_tensors(void) {
    char path[512];
    snprintf(path, sizeof(path), "%s/model.safetensors", TEST_DATA_DIR);

    SafeTensorsLoader* loader = safetensors_new(path);
    ASSERT_NOT_NULL(loader, "Loader should not be NULL");

    ASSERT_EQ(safetensors_num_tensors(loader), 2, "Should have 2 tensors");

    safetensors_free(loader);
    TEST_PASS();
}

static void test_safetensors_has_tensor(void) {
    char path[512];
    snprintf(path, sizeof(path), "%s/model.safetensors", TEST_DATA_DIR);

    SafeTensorsLoader* loader = safetensors_new(path);
    ASSERT_NOT_NULL(loader, "Loader should not be NULL");

    ASSERT_TRUE(safetensors_has_tensor(loader, "test_tensor"), "Should have test_tensor");
    ASSERT_TRUE(safetensors_has_tensor(loader, "test_vector"), "Should have test_vector");
    ASSERT_FALSE(safetensors_has_tensor(loader, "nonexistent"), "Should not have nonexistent");

    safetensors_free(loader);
    TEST_PASS();
}

static void test_safetensors_get_info(void) {
    char path[512];
    snprintf(path, sizeof(path), "%s/model.safetensors", TEST_DATA_DIR);

    SafeTensorsLoader* loader = safetensors_new(path);
    ASSERT_NOT_NULL(loader, "Loader should not be NULL");

    TensorInfo info;
    bool ok = safetensors_get_info(loader, "test_tensor", &info);
    ASSERT_TRUE(ok, "Should get tensor info");
    ASSERT_EQ(info.ndim, 2, "Should be 2D tensor");
    ASSERT_EQ(info.dims[0], 2, "First dim should be 2");
    ASSERT_EQ(info.dims[1], 3, "Second dim should be 3");
    ASSERT_EQ(info.dtype, ST_F32, "Should be F32");

    safetensors_free(loader);
    TEST_PASS();
}

static void test_safetensors_load_tensor(void) {
    char path[512];
    snprintf(path, sizeof(path), "%s/model.safetensors", TEST_DATA_DIR);

    SafeTensorsLoader* loader = safetensors_new(path);
    ASSERT_NOT_NULL(loader, "Loader should not be NULL");

    Tensor* t = safetensors_load_tensor(loader, "test_tensor");
    ASSERT_NOT_NULL(t, "Should load tensor");
    ASSERT_EQ(t->shape.ndim, 2, "Should be 2D");
    ASSERT_EQ(t->shape.dims[0], 2, "First dim should be 2");
    ASSERT_EQ(t->shape.dims[1], 3, "Second dim should be 3");

    /* 验证数据 */
    float* data = (float*)t->data;
    ASSERT_FLOAT_EQ(data[0], 0.0f, 1e-5f, "Data[0]");
    ASSERT_FLOAT_EQ(data[1], 1.0f, 1e-5f, "Data[1]");
    ASSERT_FLOAT_EQ(data[5], 5.0f, 1e-5f, "Data[5]");

    /* 手动释放张量 */
    free(t->shape.dims);
    free(t->strides);
    free(t->data);
    free(t);
    safetensors_free(loader);
    TEST_PASS();
}

static void test_safetensors_get_raw_data(void) {
    char path[512];
    snprintf(path, sizeof(path), "%s/model.safetensors", TEST_DATA_DIR);

    SafeTensorsLoader* loader = safetensors_new(path);
    ASSERT_NOT_NULL(loader, "Loader should not be NULL");

    TensorInfo info;
    const float* data = (const float*)safetensors_get_raw_data(loader, "test_tensor", &info);
    ASSERT_NOT_NULL(data, "Should get raw data");
    ASSERT_FLOAT_EQ(data[0], 0.0f, 1e-5f, "Data[0]");
    ASSERT_FLOAT_EQ(data[5], 5.0f, 1e-5f, "Data[5]");

    safetensors_free(loader);
    TEST_PASS();
}

/* ============================================================================
 * ModelConfig 测试
 * ============================================================================ */

static void test_model_config_init(void) {
    ModelConfig config;
    model_config_init(&config);

    ASSERT_EQ(config.hidden_size, 4096, "Default hidden_size");
    ASSERT_EQ(config.intermediate_size, 11008, "Default intermediate_size");
    ASSERT_EQ(config.num_attention_heads, 32, "Default num_attention_heads");
    ASSERT_EQ(config.num_hidden_layers, 32, "Default num_hidden_layers");
    ASSERT_EQ(config.vocab_size, 32000, "Default vocab_size");
    ASSERT_EQ(config.torch_dtype, DTYPE_F32, "Default dtype");

    TEST_PASS();
}

static void test_model_config_load(void) {
    char path[512];
    snprintf(path, sizeof(path), "%s/config.json", TEST_DATA_DIR);

    ModelConfig config;
    bool ok = model_config_load(path, &config);
    ASSERT_TRUE(ok, "Should load config");

    ASSERT_TRUE(strcmp(config.architecture, "LlamaForCausalLM") == 0, "Architecture");
    ASSERT_TRUE(strcmp(config.model_type, "llama") == 0, "Model type");
    ASSERT_EQ(config.hidden_size, 256, "hidden_size");
    ASSERT_EQ(config.intermediate_size, 512, "intermediate_size");
    ASSERT_EQ(config.num_attention_heads, 8, "num_attention_heads");
    ASSERT_EQ(config.num_hidden_layers, 4, "num_hidden_layers");
    ASSERT_EQ(config.vocab_size, 1000, "vocab_size");
    ASSERT_EQ(config.num_key_value_heads, 8, "num_key_value_heads");
    ASSERT_EQ(config.max_position_embeddings, 512, "max_position_embeddings");
    ASSERT_FLOAT_EQ((float)config.rope_theta, 10000.0f, 1e-5f, "rope_theta");
    ASSERT_FLOAT_EQ(config.rms_norm_eps, 1e-6f, 1e-10f, "rms_norm_eps");
    ASSERT_FALSE(config.tie_word_embeddings, "tie_word_embeddings");

    TEST_PASS();
}

/* ============================================================================
 * WeightLoader 测试
 * ============================================================================ */

static void test_weight_loader_new(void) {
    WeightLoader* loader = weight_loader_new(TEST_DATA_DIR);
    ASSERT_NOT_NULL(loader, "Loader should not be NULL");

    weight_loader_free(loader);
    TEST_PASS();
}

static void test_weight_loader_config(void) {
    WeightLoader* loader = weight_loader_new(TEST_DATA_DIR);
    ASSERT_NOT_NULL(loader, "Loader should not be NULL");

    const ModelConfig* config = weight_loader_config(loader);
    ASSERT_NOT_NULL(config, "Config should not be NULL");
    ASSERT_EQ(config->hidden_size, 256, "hidden_size from config");

    weight_loader_free(loader);
    TEST_PASS();
}

static void test_weight_loader_has_tensor(void) {
    WeightLoader* loader = weight_loader_new(TEST_DATA_DIR);
    ASSERT_NOT_NULL(loader, "Loader should not be NULL");

    ASSERT_TRUE(weight_loader_has_tensor(loader, "test_tensor"), "Should have test_tensor");
    ASSERT_FALSE(weight_loader_has_tensor(loader, "nonexistent"), "Should not have nonexistent");

    weight_loader_free(loader);
    TEST_PASS();
}

static void test_weight_loader_load(void) {
    WeightLoader* loader = weight_loader_new(TEST_DATA_DIR);
    ASSERT_NOT_NULL(loader, "Loader should not be NULL");

    Tensor* t = weight_loader_load(loader, "test_tensor");
    ASSERT_NOT_NULL(t, "Should load tensor");
    ASSERT_EQ(t->shape.ndim, 2, "Should be 2D");

    /* 手动释放张量 */
    free(t->shape.dims);
    free(t->strides);
    free(t->data);
    free(t);
    weight_loader_free(loader);
    TEST_PASS();
}

static void test_weight_loader_get_raw(void) {
    WeightLoader* loader = weight_loader_new(TEST_DATA_DIR);
    ASSERT_NOT_NULL(loader, "Loader should not be NULL");

    const float* data = (const float*)weight_loader_get_raw(loader, "test_tensor");
    ASSERT_NOT_NULL(data, "Should get raw data");
    ASSERT_FLOAT_EQ(data[0], 0.0f, 1e-5f, "Data[0]");

    weight_loader_free(loader);
    TEST_PASS();
}

/* ============================================================================
 * 内存测试
 * ============================================================================ */

static void test_safetensors_from_memory(void) {
    /* 读取文件到内存 */
    char path[512];
    snprintf(path, sizeof(path), "%s/model.safetensors", TEST_DATA_DIR);

    FILE* f = fopen(path, "rb");
    ASSERT_NOT_NULL(f, "Should open file");

    fseek(f, 0, SEEK_END);
    long size = ftell(f);
    fseek(f, 0, SEEK_SET);

    uint8_t* data = (uint8_t*)malloc(size);
    ASSERT_NOT_NULL(data, "Should allocate memory");

    ASSERT_EQ(fread(data, 1, size, f), (size_t)size, "Should read file");
    fclose(f);

    /* 从内存创建加载器 */
    SafeTensorsLoader* loader = safetensors_from_memory(data, size);
    ASSERT_NOT_NULL(loader, "Should create loader from memory");
    ASSERT_EQ(safetensors_num_tensors(loader), 2, "Should have 2 tensors");

    safetensors_free(loader);
    free(data);
    TEST_PASS();
}

/* ============================================================================
 * 主函数
 * ============================================================================ */

int main(void) {
    printf("========================================\n");
    printf("Loader Module Unit Tests\n");
    printf("========================================\n\n");

    /* 设置测试数据 */
    if (!setup_test_model_dir()) {
        fprintf(stderr, "Failed to setup test data directory\n");
        return 1;
    }

    /* SafeTensors 数据类型测试 */
    printf("[SafeTensors DType Tests]\n");
    RUN_TEST(test_safetensors_dtype_size);
    RUN_TEST(test_safetensors_dtype_from_string);
    RUN_TEST(test_safetensors_dtype_to_model);
    printf("\n");

    /* SafeTensors 加载测试 */
    printf("[SafeTensors Loading Tests]\n");
    RUN_TEST(test_safetensors_new);
    RUN_TEST(test_safetensors_new_invalid);
    RUN_TEST(test_safetensors_num_tensors);
    RUN_TEST(test_safetensors_has_tensor);
    RUN_TEST(test_safetensors_get_info);
    RUN_TEST(test_safetensors_load_tensor);
    RUN_TEST(test_safetensors_get_raw_data);
    printf("\n");

    /* ModelConfig 测试 */
    printf("[ModelConfig Tests]\n");
    RUN_TEST(test_model_config_init);
    RUN_TEST(test_model_config_load);
    printf("\n");

    /* WeightLoader 测试 */
    printf("[WeightLoader Tests]\n");
    RUN_TEST(test_weight_loader_new);
    RUN_TEST(test_weight_loader_config);
    RUN_TEST(test_weight_loader_has_tensor);
    RUN_TEST(test_weight_loader_load);
    RUN_TEST(test_weight_loader_get_raw);
    printf("\n");

    /* 内存测试 */
    printf("[Memory Tests]\n");
    RUN_TEST(test_safetensors_from_memory);
    printf("\n");

    /* 清理测试数据 */
    cleanup_test_dir();

    TEST_SUMMARY();
    return TEST_RETURN();
}
