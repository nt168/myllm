/**
 * @file safetensors.c
 * @brief SafeTensors 格式加载器实现
 *
 * SafeTensors 格式规范:
 * - 8字节小端 uint64: JSON 元数据长度 N
 * - N 字节: UTF-8 JSON 元数据
 * - 剩余: 张量数据
 *
 * JSON 元数据格式:
 * {
 *   "tensor_name": {
 *     "dtype": "F32",
 *     "shape": [dim1, dim2, ...],
 *     "data_offsets": [start, end]
 *   },
 *   "__metadata__": { ... }
 * }
 */

#include "loader/safetensors.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>

/* ============================================================================
 * 内部 JSON 解析 (简化版，无依赖)
 * ============================================================================ */

/* 跳过空白字符 */
static const char* skip_ws(const char* p) {
    while (*p == ' ' || *p == '\t' || *p == '\n' || *p == '\r') p++;
    return p;
}

/* 查找键 */
static const char* find_key(const char* json, const char* key) {
    char search[256];
    snprintf(search, sizeof(search), "\"%s\"", key);
    return strstr(json, search);
}

/* 提取字符串值 */
static bool extract_string(const char* json, const char* key, char* out, size_t max_len) {
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

/* 提取数组值 */
static size_t extract_array(const char* json, const char* key, size_t* out, size_t max_len) {
    const char* pos = find_key(json, key);
    if (!pos) return 0;

    pos = strchr(pos + strlen(key) + 2, ':');
    if (!pos) return 0;

    pos = skip_ws(pos + 1);
    if (*pos != '[') return 0;

    pos++;
    size_t count = 0;

    while (*pos && *pos != ']' && count < max_len) {
        pos = skip_ws(pos);
        if (*pos == ']') break;

        /* 解析数字 */
        out[count++] = strtoull(pos, (char**)&pos, 10);

        pos = skip_ws(pos);
        if (*pos == ',') pos++;
    }

    return count;
}

/* 提取数据偏移 */
static bool extract_offsets(const char* json, size_t* start, size_t* end) {
    const char* pos = find_key(json, "data_offsets");
    if (!pos) return false;

    pos = strchr(pos, ':');
    if (!pos) return false;

    pos = skip_ws(pos + 1);
    if (*pos != '[') return false;

    pos++;
    pos = skip_ws(pos);
    *start = strtoull(pos, (char**)&pos, 10);

    pos = skip_ws(pos);
    if (*pos == ',') pos++;
    pos = skip_ws(pos);
    *end = strtoull(pos, (char**)&pos, 10);

    return true;
}

/* ============================================================================
 * DType 转换
 * ============================================================================ */

SafeTensorsDType safetensors_dtype_from_string(const char* str) {
    if (!str) return ST_UNKNOWN;
    if (strcmp(str, "F32") == 0) return ST_F32;
    if (strcmp(str, "F16") == 0) return ST_F16;
    if (strcmp(str, "BF16") == 0) return ST_BF16;
    if (strcmp(str, "I32") == 0) return ST_I32;
    if (strcmp(str, "I64") == 0) return ST_I64;
    if (strcmp(str, "I16") == 0) return ST_I16;
    if (strcmp(str, "I8") == 0) return ST_I8;
    if (strcmp(str, "U8") == 0) return ST_U8;
    if (strcmp(str, "U32") == 0) return ST_U32;
    if (strcmp(str, "U64") == 0) return ST_U64;
    if (strcmp(str, "F8_E4M3") == 0) return ST_F8_E4M3;
    if (strcmp(str, "F8_E5M2") == 0) return ST_F8_E5M2;
    return ST_UNKNOWN;
}

DType safetensors_dtype_to_model(SafeTensorsDType st_dtype) {
    switch (st_dtype) {
        case ST_F32: return DTYPE_F32;
        case ST_F16: return DTYPE_F16;
        case ST_BF16: return DTYPE_BF16;
        case ST_I32: return DTYPE_I32;
        case ST_I64: return DTYPE_I64;
        default: return DTYPE_F32;  /* 默认 */
    }
}

size_t safetensors_dtype_size(SafeTensorsDType dtype) {
    switch (dtype) {
        case ST_F32: return 4;
        case ST_F16: return 2;
        case ST_BF16: return 2;
        case ST_I32: return 4;
        case ST_I64: return 8;
        case ST_I16: return 2;
        case ST_I8: return 1;
        case ST_U8: return 1;
        case ST_U32: return 4;
        case ST_U64: return 8;
        case ST_F8_E4M3: return 1;
        case ST_F8_E5M2: return 1;
        default: return 4;
    }
}

/* ============================================================================
 * 内部: 解析张量信息
 * ============================================================================ */

static bool parse_tensor_info(const char* json_start, const char* name_start, size_t name_len, TensorInfo* info) {
    /* 复制名称 */
    if (name_len >= sizeof(info->name) - 1) name_len = sizeof(info->name) - 1;
    memcpy(info->name, name_start, name_len);
    info->name[name_len] = '\0';

    /* 提取 dtype */
    char dtype_str[32] = {0};
    if (!extract_string(json_start, "dtype", dtype_str, sizeof(dtype_str))) {
        return false;
    }
    info->dtype = safetensors_dtype_from_string(dtype_str);

    /* 提取 shape */
    info->ndim = extract_array(json_start, "shape", info->dims, 8);
    if (info->ndim == 0) return false;

    /* 提取偏移 */
    size_t start, end;
    if (!extract_offsets(json_start, &start, &end)) {
        return false;
    }
    info->offset = start;
    info->num_bytes = end - start;

    return true;
}

/* ============================================================================
 * 生命周期管理
 * ============================================================================ */

SafeTensorsLoader* safetensors_new(const char* path) {
    if (!path) return NULL;

    FILE* f = fopen(path, "rb");
    if (!f) {
        fprintf(stderr, "SafeTensors: cannot open file '%s'\n", path);
        return NULL;
    }

    /* 获取文件大小 */
    fseek(f, 0, SEEK_END);
    long file_size = ftell(f);
    fseek(f, 0, SEEK_SET);

    if (file_size < 8) {
        fprintf(stderr, "SafeTensors: file too small\n");
        fclose(f);
        return NULL;
    }

    /* 读取整个文件 */
    uint8_t* data = (uint8_t*)malloc(file_size);
    if (!data) {
        fclose(f);
        return NULL;
    }

    if (fread(data, 1, file_size, f) != (size_t)file_size) {
        fprintf(stderr, "SafeTensors: failed to read file\n");
        free(data);
        fclose(f);
        return NULL;
    }
    fclose(f);

    SafeTensorsLoader* loader = safetensors_from_memory(data, file_size);
    if (loader) {
        loader->owns_data = true;
    } else {
        free(data);
    }

    return loader;
}

SafeTensorsLoader* safetensors_from_memory(const void* data, size_t size) {
    if (!data || size < 8) {
        fprintf(stderr, "SafeTensors: invalid parameters\n");
        return NULL;
    }

    const uint8_t* ptr = (const uint8_t*)data;

    /* 读取 JSON 大小 (小端 uint64) */
    uint64_t json_size = 0;
    for (int i = 0; i < 8; i++) {
        json_size |= ((uint64_t)ptr[i]) << (i * 8);
    }

    if (json_size > size - 8) {
        fprintf(stderr, "SafeTensors: invalid json size %llu\n", (unsigned long long)json_size);
        return NULL;
    }

    /* 分配加载器 */
    SafeTensorsLoader* loader = (SafeTensorsLoader*)calloc(1, sizeof(SafeTensorsLoader));
    if (!loader) return NULL;

    loader->data = (uint8_t*)data;
    loader->data_size = size;
    loader->json_size = json_size;
    loader->owns_data = false;

    /* 复制 JSON 元数据 */
    loader->json_metadata = (char*)malloc(json_size + 1);
    if (!loader->json_metadata) {
        free(loader);
        return NULL;
    }
    memcpy(loader->json_metadata, ptr + 8, json_size);
    loader->json_metadata[json_size] = '\0';

    /* 第一遍: 计算张量数量 */
    const char* p = loader->json_metadata;
    size_t num_tensors = 0;
    while ((p = strstr(p, "\"dtype\"")) != NULL) {
        num_tensors++;
        p++;
    }

    if (num_tensors == 0) {
        fprintf(stderr, "SafeTensors: no tensors found\n");
        free(loader->json_metadata);
        free(loader);
        return NULL;
    }

    /* 分配张量信息数组 */
    loader->tensors = (TensorInfo*)calloc(num_tensors, sizeof(TensorInfo));
    if (!loader->tensors) {
        free(loader->json_metadata);
        free(loader);
        return NULL;
    }
    loader->num_tensors = num_tensors;

    /* 第二遍: 解析张量信息 */
    p = loader->json_metadata;
    size_t tensor_idx = 0;

    while (tensor_idx < num_tensors && p) {
        /* 查找张量名 (格式: "name": { ... }) */
        const char* quote1 = strchr(p, '"');
        if (!quote1) break;
        const char* name_start = quote1 + 1;

        const char* quote2 = strchr(name_start, '"');
        if (!quote2) break;
        size_t name_len = quote2 - name_start;

        /* 查找冒号和对象开始 */
        const char* colon = strchr(quote2, ':');
        const char* next_pos = quote2 + 1;  /* 默认跳过当前引号对 */

        if (colon) {
            colon = skip_ws(colon + 1);
            if (*colon == '{') {
                const char* obj_start = colon;
                const char* obj_end = strchr(obj_start, '}');
                if (obj_end) {
                    /* 更新下次搜索位置到对象结束后 */
                    next_pos = obj_end + 1;

                    /* 跳过特殊键和元数据 */
                    if (name_len > 0 && name_start[0] != '_' &&
                        strncmp(name_start, "dtype", 5) != 0 &&
                        strncmp(name_start, "shape", 5) != 0 &&
                        strncmp(name_start, "data_offsets", 12) != 0 &&
                        strncmp(name_start, "format", 6) != 0) {

                        /* 检查是否有 dtype */
                        if (find_key(obj_start, "dtype") && find_key(obj_start, "data_offsets")) {
                            /* 提取张量信息 */
                            size_t obj_len = obj_end - obj_start + 1;
                            char* obj_copy = (char*)malloc(obj_len + 1);
                            if (obj_copy) {
                                memcpy(obj_copy, obj_start, obj_len);
                                obj_copy[obj_len] = '\0';

                                if (parse_tensor_info(obj_copy, name_start, name_len,
                                                     &loader->tensors[tensor_idx])) {
                                    tensor_idx++;
                                }
                                free(obj_copy);
                            }
                        }
                    }
                }
            }
        }

        p = next_pos;
    }

    loader->num_tensors = tensor_idx;

    return loader;
}

void safetensors_free(SafeTensorsLoader* loader) {
    if (!loader) return;

    if (loader->owns_data && loader->data) {
        free(loader->data);
    }
    free(loader->json_metadata);
    free(loader->tensors);
    free(loader);
}

/* ============================================================================
 * 状态查询
 * ============================================================================ */

size_t safetensors_num_tensors(const SafeTensorsLoader* loader) {
    return loader ? loader->num_tensors : 0;
}

size_t safetensors_get_names(const SafeTensorsLoader* loader, char** names, size_t max_names) {
    if (!loader || !names) return 0;

    size_t count = (loader->num_tensors < max_names) ? loader->num_tensors : max_names;
    for (size_t i = 0; i < count; i++) {
        names[i] = loader->tensors[i].name;
    }
    return count;
}

bool safetensors_has_tensor(const SafeTensorsLoader* loader, const char* name) {
    if (!loader || !name) return false;

    for (size_t i = 0; i < loader->num_tensors; i++) {
        if (strcmp(loader->tensors[i].name, name) == 0) {
            return true;
        }
    }
    return false;
}

bool safetensors_get_info(const SafeTensorsLoader* loader, const char* name, TensorInfo* info) {
    if (!loader || !name || !info) return false;

    for (size_t i = 0; i < loader->num_tensors; i++) {
        if (strcmp(loader->tensors[i].name, name) == 0) {
            *info = loader->tensors[i];
            return true;
        }
    }
    return false;
}

/* ============================================================================
 * 内部: 查找张量信息
 * ============================================================================ */

static TensorInfo* find_tensor(SafeTensorsLoader* loader, const char* name) {
    for (size_t i = 0; i < loader->num_tensors; i++) {
        if (strcmp(loader->tensors[i].name, name) == 0) {
            return &loader->tensors[i];
        }
    }
    return NULL;
}

/* ============================================================================
 * 张量加载
 * ============================================================================ */

Tensor* safetensors_load_tensor(const SafeTensorsLoader* loader, const char* name) {
    if (!loader || !name) return NULL;

    SafeTensorsLoader* l = (SafeTensorsLoader*)loader;
    TensorInfo* info = find_tensor(l, name);
    if (!info) {
        fprintf(stderr, "SafeTensors: tensor '%s' not found\n", name);
        return NULL;
    }

    /* 创建张量 */
    Tensor* t = (Tensor*)malloc(sizeof(Tensor));
    if (!t) return NULL;
    memset(t, 0, sizeof(Tensor));

    /* 使用 shape_new 创建形状 */
    t->shape = shape_new(info->dims, info->ndim);

    /* 计算步幅 */
    if (info->ndim > 0) {
        t->strides[info->ndim - 1] = 1;
        for (int i = (int)info->ndim - 2; i >= 0; i--) {
            t->strides[i] = t->strides[i + 1] * t->shape.dims[i + 1];
        }
    }

    t->dtype = safetensors_dtype_to_model(info->dtype);
    t->offset = 0;
    t->device.type = DEVICE_CPU;
    t->device.id = 0;
    t->owns_data = true;

    /* 复制数据 */
    t->data = malloc(info->num_bytes);
    if (!t->data) {
        free(t);
        return NULL;
    }

    /* 数据偏移: 8 (header) + json_size + tensor_offset */
    size_t base_offset = 8 + loader->json_size;
    memcpy(t->data, loader->data + base_offset + info->offset, info->num_bytes);

    return t;
}

Tensor* safetensors_load_tensor_f32(const SafeTensorsLoader* loader, const char* name) {
    if (!loader || !name) return NULL;

    SafeTensorsLoader* l = (SafeTensorsLoader*)loader;
    TensorInfo* info = find_tensor(l, name);
    if (!info) {
        fprintf(stderr, "SafeTensors: tensor '%s' not found\n", name);
        return NULL;
    }

    /* 先加载原始张量 */
    Tensor* src = safetensors_load_tensor(loader, name);
    if (!src) return NULL;

    /* 如果已经是 F32，直接返回 */
    if (info->dtype == ST_F32) {
        return src;
    }

    /* 创建 F32 张量 */
    Tensor* dst = (Tensor*)malloc(sizeof(Tensor));
    if (!dst) {
        free(src->data);
        free(src);
        return NULL;
    }
    memset(dst, 0, sizeof(Tensor));

    /* 复制形状信息 */
    dst->shape = src->shape;  /* shape is embedded, copy directly */
    memcpy(dst->strides, src->strides, MYLLM_MAX_NDIM * sizeof(size_t));

    dst->dtype = DTYPE_F32;
    dst->offset = 0;
    dst->device.type = DEVICE_CPU;
    dst->device.id = 0;
    dst->owns_data = true;

    size_t numel = shape_numel(&src->shape);
    dst->data = malloc(numel * sizeof(float));

    if (!dst->data) {
        free(dst);
        free(src->data);
        free(src);
        return NULL;
    }

    float* dst_data = (float*)dst->data;

    /* 类型转换 */
    switch (info->dtype) {
        case ST_F16: {
            uint16_t* src_data = (uint16_t*)src->data;
            for (size_t i = 0; i < numel; i++) {
                /* FP16 转 FP32 */
                uint16_t h = src_data[i];
                uint32_t sign = (h & 0x8000) << 16;
                uint32_t exponent = (h >> 10) & 0x1F;
                uint32_t mantissa = h & 0x3FF;

                if (exponent == 0) {
                    if (mantissa == 0) {
                        dst_data[i] = *(float*)&sign;
                    } else {
                        /* 非规格化数 */
                        exponent = 127 - 15 + 1;
                        while (!(mantissa & 0x400)) {
                            mantissa <<= 1;
                            exponent--;
                        }
                        mantissa &= 0x3FF;
                        uint32_t f = sign | ((exponent - 1) << 23) | (mantissa << 13);
                        dst_data[i] = *(float*)&f;
                    }
                } else if (exponent == 31) {
                    /* 无穷大或 NaN */
                    uint32_t f = sign | 0x7F800000 | (mantissa << 13);
                    dst_data[i] = *(float*)&f;
                } else {
                    /* 规格化数 */
                    uint32_t f = sign | ((exponent + 112) << 23) | (mantissa << 13);
                    dst_data[i] = *(float*)&f;
                }
            }
            break;
        }
        case ST_BF16: {
            uint16_t* src_data = (uint16_t*)src->data;
            for (size_t i = 0; i < numel; i++) {
                /* BF16 转 FP32: 零扩展 */
                uint32_t f = (uint32_t)src_data[i] << 16;
                dst_data[i] = *(float*)&f;
            }
            break;
        }
        case ST_I32: {
            int32_t* src_data = (int32_t*)src->data;
            for (size_t i = 0; i < numel; i++) {
                dst_data[i] = (float)src_data[i];
            }
            break;
        }
        case ST_I64: {
            int64_t* src_data = (int64_t*)src->data;
            for (size_t i = 0; i < numel; i++) {
                dst_data[i] = (float)src_data[i];
            }
            break;
        }
        default:
            /* 其他类型，尝试直接复制 */
            memcpy(dst->data, src->data, numel * sizeof(float));
            break;
    }

    /* 释放源张量 */
    free(src->data);
    free(src);

    return dst;
}

const void* safetensors_get_raw_data(
    const SafeTensorsLoader* loader,
    const char* name,
    TensorInfo* info
) {
    if (!loader || !name) return NULL;

    SafeTensorsLoader* l = (SafeTensorsLoader*)loader;
    TensorInfo* tinfo = find_tensor(l, name);
    if (!tinfo) return NULL;

    if (info) {
        *info = *tinfo;
    }

    size_t base_offset = 8 + loader->json_size;
    return loader->data + base_offset + tinfo->offset;
}
