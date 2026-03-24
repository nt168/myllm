/**
 * @file loader.c
 * @brief 模型加载器实现
 */

#include "loader/loader.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <ctype.h>

/* ============================================================================
 * 内部 JSON 解析辅助函数
 * ============================================================================ */

static const char* skip_ws(const char* p) {
    while (*p == ' ' || *p == '\t' || *p == '\n' || *p == '\r') p++;
    return p;
}

static const char* find_key(const char* json, const char* key) {
    char search[256];
    snprintf(search, sizeof(search), "\"%s\"", key);
    return strstr(json, search);
}

static bool extract_string_val(const char* json, const char* key, char* out, size_t max_len) {
    const char* pos = find_key(json, key);
    if (!pos) return false;

    pos = strchr(pos + strlen(key) + 2, ':');
    if (!pos) return false;

    pos = skip_ws(pos + 1);
    if (*pos != '"') return false;

    pos++;
    const char* end = strchr(pos, '"');
    if (!end) return false;

    size_t len = end - pos;
    if (len >= max_len) len = max_len - 1;
    strncpy(out, pos, len);
    out[len] = '\0';
    return true;
}

static bool extract_int_val(const char* json, const char* key, int64_t* out) {
    const char* pos = find_key(json, key);
    if (!pos) return false;

    pos = strchr(pos + strlen(key) + 2, ':');
    if (!pos) return false;

    pos = skip_ws(pos + 1);
    *out = strtoll(pos, NULL, 10);
    return true;
}

static bool extract_float_val(const char* json, const char* key, double* out) {
    const char* pos = find_key(json, key);
    if (!pos) return false;

    pos = strchr(pos + strlen(key) + 2, ':');
    if (!pos) return false;

    pos = skip_ws(pos + 1);
    *out = strtod(pos, NULL);
    return true;
}

static bool extract_bool_val(const char* json, const char* key, bool* out) {
    const char* pos = find_key(json, key);
    if (!pos) return false;

    pos = strchr(pos + strlen(key) + 2, ':');
    if (!pos) return false;

    pos = skip_ws(pos + 1);
    if (strncmp(pos, "true", 4) == 0) {
        *out = true;
        return true;
    } else if (strncmp(pos, "false", 5) == 0) {
        *out = false;
        return true;
    }
    return false;
}

/* ============================================================================
 * 模型配置
 * ============================================================================ */

void model_config_init(ModelConfig* config) {
    if (!config) return;
    memset(config, 0, sizeof(ModelConfig));
    config->hidden_size = 4096;
    config->intermediate_size = 11008;
    config->num_attention_heads = 32;
    config->num_hidden_layers = 32;
    config->vocab_size = 32000;
    config->num_key_value_heads = 32;
    config->head_dim = 128;
    config->max_position_embeddings = 2048;
    config->rope_theta = 10000.0;
    config->rms_norm_eps = 1e-6f;
    config->torch_dtype = DTYPE_F32;
    config->tie_word_embeddings = false;
}

bool model_config_load(const char* path, ModelConfig* config) {
    if (!path || !config) return false;

    FILE* f = fopen(path, "r");
    if (!f) {
        fprintf(stderr, "ModelConfig: cannot open '%s'\n", path);
        return false;
    }

    /* 读取整个文件 */
    fseek(f, 0, SEEK_END);
    long size = ftell(f);
    fseek(f, 0, SEEK_SET);

    char* json = (char*)malloc(size + 1);
    if (!json) {
        fclose(f);
        return false;
    }

    if (fread(json, 1, size, f) != (size_t)size) {
        free(json);
        fclose(f);
        return false;
    }
    json[size] = '\0';
    fclose(f);

    /* 初始化默认值 */
    model_config_init(config);

    /* 解析 architectures 数组 */
    const char* arch_pos = find_key(json, "architectures");
    if (arch_pos) {
        arch_pos = strchr(arch_pos, '[');
        if (arch_pos) {
            arch_pos = strchr(arch_pos, '"');
            if (arch_pos) {
                arch_pos++;
                const char* end = strchr(arch_pos, '"');
                if (end) {
                    size_t len = end - arch_pos;
                    if (len >= sizeof(config->architecture)) len = sizeof(config->architecture) - 1;
                    strncpy(config->architecture, arch_pos, len);
                    config->architecture[len] = '\0';
                }
            }
        }
    }

    extract_string_val(json, "model_type", config->model_type, sizeof(config->model_type));

    int64_t val;
    if (extract_int_val(json, "hidden_size", &val)) config->hidden_size = (size_t)val;
    if (extract_int_val(json, "intermediate_size", &val)) config->intermediate_size = (size_t)val;
    if (extract_int_val(json, "num_attention_heads", &val)) config->num_attention_heads = (size_t)val;
    if (extract_int_val(json, "num_hidden_layers", &val)) config->num_hidden_layers = (size_t)val;
    if (extract_int_val(json, "vocab_size", &val)) config->vocab_size = (size_t)val;
    if (extract_int_val(json, "num_key_value_heads", &val)) config->num_key_value_heads = (size_t)val;
    if (extract_int_val(json, "max_position_embeddings", &val)) config->max_position_embeddings = (size_t)val;

    double fval;
    if (extract_float_val(json, "rope_theta", &fval)) config->rope_theta = fval;
    if (extract_float_val(json, "rms_norm_eps", &fval)) config->rms_norm_eps = (float)fval;

    extract_bool_val(json, "tie_word_embeddings", &config->tie_word_embeddings);

    /* 解析 torch_dtype */
    char dtype_str[32] = {0};
    if (extract_string_val(json, "torch_dtype", dtype_str, sizeof(dtype_str))) {
        if (strcmp(dtype_str, "float32") == 0 || strcmp(dtype_str, "F32") == 0) {
            config->torch_dtype = DTYPE_F32;
        } else if (strcmp(dtype_str, "float16") == 0 || strcmp(dtype_str, "F16") == 0) {
            config->torch_dtype = DTYPE_F16;
        } else if (strcmp(dtype_str, "bfloat16") == 0 || strcmp(dtype_str, "BF16") == 0) {
            config->torch_dtype = DTYPE_BF16;
        }
    }

    /* 计算 head_dim */
    if (config->num_attention_heads > 0) {
        config->head_dim = config->hidden_size / config->num_attention_heads;
    }

    /* 如果没有 num_key_value_heads，默认等于 num_attention_heads */
    if (config->num_key_value_heads == 0) {
        config->num_key_value_heads = config->num_attention_heads;
    }

    free(json);
    return true;
}

/* ============================================================================
 * 权重加载器
 * ============================================================================ */

WeightLoader* weight_loader_new(const char* model_dir) {
    if (!model_dir) return NULL;

    WeightLoader* loader = (WeightLoader*)calloc(1, sizeof(WeightLoader));
    if (!loader) return NULL;

    strncpy(loader->model_dir, model_dir, sizeof(loader->model_dir) - 1);

    /* 加载 config.json */
    char config_path[1024];
    snprintf(config_path, sizeof(config_path), "%s/config.json", model_dir);

    if (!model_config_load(config_path, &loader->config)) {
        fprintf(stderr, "WeightLoader: failed to load config from '%s'\n", config_path);
        free(loader);
        return NULL;
    }

    /* 查找 safetensors 文件 */
    char safetensors_path[1024];

    /* 尝试 model.safetensors */
    snprintf(safetensors_path, sizeof(safetensors_path), "%s/model.safetensors", model_dir);

    FILE* f = fopen(safetensors_path, "rb");
    if (!f) {
        /* 尝试 model-00001-of-00001.safetensors */
        snprintf(safetensors_path, sizeof(safetensors_path),
                 "%s/model-00001-of-00001.safetensors", model_dir);
        f = fopen(safetensors_path, "rb");
    }

    if (!f) {
        fprintf(stderr, "WeightLoader: no safetensors file found in '%s'\n", model_dir);
        free(loader);
        return NULL;
    }
    fclose(f);

    /* 加载 SafeTensors */
    loader->safetensors = safetensors_new(safetensors_path);
    if (!loader->safetensors) {
        fprintf(stderr, "WeightLoader: failed to load safetensors from '%s'\n", safetensors_path);
        free(loader);
        return NULL;
    }

    return loader;
}

void weight_loader_free(WeightLoader* loader) {
    if (!loader) return;

    if (loader->safetensors) {
        safetensors_free(loader->safetensors);
    }
    free(loader);
}

const ModelConfig* weight_loader_config(const WeightLoader* loader) {
    return loader ? &loader->config : NULL;
}

Tensor* weight_loader_load(const WeightLoader* loader, const char* name) {
    if (!loader || !name) return NULL;
    return safetensors_load_tensor(loader->safetensors, name);
}

Tensor* weight_loader_load_f32(const WeightLoader* loader, const char* name) {
    if (!loader || !name) return NULL;
    return safetensors_load_tensor_f32(loader->safetensors, name);
}

bool weight_loader_has_tensor(const WeightLoader* loader, const char* name) {
    if (!loader || !name) return false;
    return safetensors_has_tensor(loader->safetensors, name);
}

const void* weight_loader_get_raw(const WeightLoader* loader, const char* name) {
    if (!loader || !name) return NULL;
    return safetensors_get_raw_data(loader->safetensors, name, NULL);
}
