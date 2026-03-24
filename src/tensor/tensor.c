/**
 * @file tensor.c
 * @brief 张量实现
 */

#include "tensor/tensor.h"
#include <stdlib.h>
#include <string.h>
#include <stdio.h>
#include <math.h>

/* ============================================================================
 * 内部辅助函数
 * ============================================================================ */

/**
 * @brief 计算线性偏移的字节位置
 */
static size_t compute_linear_offset(const Tensor* t, size_t linear_index) {
    if (t->shape.ndim == 0) {
        return t->offset;
    }

    size_t offset = 0;
    size_t remaining = linear_index;

    /* 从最后一维开始计算 */
    for (ssize_t i = (ssize_t)t->shape.ndim - 1; i >= 0; i--) {
        size_t dim_size = t->shape.dims[i];
        size_t idx = remaining % dim_size;
        remaining /= dim_size;
        offset += idx * t->strides[i];
    }

    return t->offset + offset * dtype_size(t->dtype);
}

/**
 * @brief f16 到 f32 转换 (简化版)
 */
static float f16_to_f32(uint16_t h) {
    uint32_t sign = ((uint32_t)h & 0x8000) << 16;
    uint32_t exponent = ((uint32_t)h & 0x7C00) >> 10;
    uint32_t mantissa = ((uint32_t)h & 0x03FF);

    if (exponent == 0) {
        if (mantissa == 0) {
            return *(float*)&sign;
        }
        /* 非规格化数 */
        exponent = 1;
        while ((mantissa & 0x0400) == 0) {
            mantissa <<= 1;
            exponent--;
        }
        mantissa &= 0x03FF;
        exponent = 127 - 15 - exponent + 1;
    } else if (exponent == 0x1F) {
        /* 无穷或NaN */
        uint32_t f32 = sign | 0x7F800000 | (mantissa << 13);
        return *(float*)&f32;
    } else {
        exponent = exponent - 15 + 127;
    }

    uint32_t f32 = sign | (exponent << 23) | (mantissa << 13);
    return *(float*)&f32;
}

/**
 * @brief f32 到 f16 转换 (简化版)
 */
static uint16_t f32_to_f16(float f) {
    uint32_t f32 = *(uint32_t*)&f;
    uint32_t sign = (f32 & 0x80000000) >> 16;
    int32_t exponent = ((f32 & 0x7F800000) >> 23) - 127 + 15;
    uint32_t mantissa = (f32 & 0x007FFFFF) >> 13;

    if (exponent <= 0) {
        if (exponent < -10) return (uint16_t)sign;
        mantissa |= 0x0400;
        mantissa >>= (1 - exponent);
        return (uint16_t)(sign | mantissa);
    } else if (exponent >= 0x1F) {
        return (uint16_t)(sign | 0x7C00 | (mantissa ? 1 : 0));
    }

    return (uint16_t)(sign | (exponent << 10) | mantissa);
}

/**
 * @brief bf16 到 f32 转换
 */
static float bf16_to_f32(uint16_t b) {
    uint32_t f32 = ((uint32_t)b) << 16;
    return *(float*)&f32;
}

/**
 * @brief f32 到 bf16 转换
 */
static uint16_t f32_to_bf16(float f) {
    uint32_t f32 = *(uint32_t*)&f;
    return (uint16_t)(f32 >> 16);
}

/* ============================================================================
 * 创建与释放
 * ============================================================================ */

Tensor* tensor_new(const Shape* shape, DType dtype) {
    if (!shape) return NULL;

    Tensor* t = (Tensor*)malloc(sizeof(Tensor));
    if (!t) return NULL;

    memset(t, 0, sizeof(Tensor));
    t->shape = *shape;
    t->dtype = dtype;
    t->device = device_cpu();
    t->offset = 0;
    t->owns_data = true;

    /* 计算连续存储的步幅 */
    shape_strides(shape, t->strides);

    size_t numel = shape_numel(shape);
    size_t total_bytes = numel * dtype_size(dtype);

    if (total_bytes > 0) {
        t->data = malloc(total_bytes);
        if (!t->data) {
            free(t);
            return NULL;
        }
    }

    return t;
}

Tensor* tensor_zeros(const size_t* dims, size_t ndim, DType dtype) {
    Shape shape = shape_new(dims, ndim);
    Tensor* t = tensor_new(&shape, dtype);
    if (t && t->data) {
        size_t total_bytes = shape_numel(&shape) * dtype_size(dtype);
        memset(t->data, 0, total_bytes);
    }
    return t;
}

Tensor* tensor_ones(const size_t* dims, size_t ndim, DType dtype) {
    Shape shape = shape_new(dims, ndim);
    Tensor* t = tensor_new(&shape, dtype);
    if (!t) return NULL;

    size_t numel = shape_numel(&shape);

    switch (dtype) {
        case DTYPE_F32: {
            float* p = (float*)t->data;
            for (size_t i = 0; i < numel; i++) p[i] = 1.0f;
            break;
        }
        case DTYPE_F16: {
            uint16_t* p = (uint16_t*)t->data;
            for (size_t i = 0; i < numel; i++) p[i] = f32_to_f16(1.0f);
            break;
        }
        case DTYPE_BF16: {
            uint16_t* p = (uint16_t*)t->data;
            for (size_t i = 0; i < numel; i++) p[i] = f32_to_bf16(1.0f);
            break;
        }
        case DTYPE_I32: {
            int32_t* p = (int32_t*)t->data;
            for (size_t i = 0; i < numel; i++) p[i] = 1;
            break;
        }
        case DTYPE_I64: {
            int64_t* p = (int64_t*)t->data;
            for (size_t i = 0; i < numel; i++) p[i] = 1;
            break;
        }
        default:
            break;
    }

    return t;
}

Tensor* tensor_from_data(const void* data, const size_t* dims, size_t ndim,
                          DType dtype, bool copy) {
    if (!data || !dims) return NULL;

    Shape shape = shape_new(dims, ndim);
    size_t total_bytes = shape_numel(&shape) * dtype_size(dtype);

    Tensor* t = (Tensor*)malloc(sizeof(Tensor));
    if (!t) return NULL;

    memset(t, 0, sizeof(Tensor));
    t->shape = shape;
    shape_strides(&shape, t->strides);
    t->dtype = dtype;
    t->device = device_cpu();
    t->offset = 0;

    if (copy) {
        t->data = malloc(total_bytes);
        if (!t->data) {
            free(t);
            return NULL;
        }
        memcpy(t->data, data, total_bytes);
        t->owns_data = true;
    } else {
        t->data = (void*)data;
        t->owns_data = false;
    }

    return t;
}

Tensor* tensor_from_f32(const float* data, const size_t* dims, size_t ndim, DType dtype) {
    if (!data || !dims) return NULL;

    Shape shape = shape_new(dims, ndim);
    size_t numel = shape_numel(&shape);

    Tensor* t = tensor_new(&shape, dtype);
    if (!t) return NULL;

    uint8_t* dst = (uint8_t*)t->data;

    switch (dtype) {
        case DTYPE_F32:
            memcpy(dst, data, numel * sizeof(float));
            break;
        case DTYPE_F16: {
            uint16_t* p = (uint16_t*)dst;
            for (size_t i = 0; i < numel; i++) {
                p[i] = f32_to_f16(data[i]);
            }
            break;
        }
        case DTYPE_BF16: {
            uint16_t* p = (uint16_t*)dst;
            for (size_t i = 0; i < numel; i++) {
                p[i] = f32_to_bf16(data[i]);
            }
            break;
        }
        case DTYPE_I32: {
            int32_t* p = (int32_t*)dst;
            for (size_t i = 0; i < numel; i++) {
                p[i] = (int32_t)data[i];
            }
            break;
        }
        case DTYPE_I64: {
            int64_t* p = (int64_t*)dst;
            for (size_t i = 0; i < numel; i++) {
                p[i] = (int64_t)data[i];
            }
            break;
        }
        default:
            break;
    }

    return t;
}

Tensor* tensor_from_i32(const int32_t* data, const size_t* dims, size_t ndim, DType dtype) {
    if (!data || !dims) return NULL;

    Shape shape = shape_new(dims, ndim);
    size_t numel = shape_numel(&shape);

    Tensor* t = tensor_new(&shape, dtype);
    if (!t) return NULL;

    if (dtype == DTYPE_I32) {
        memcpy(t->data, data, numel * sizeof(int32_t));
    } else {
        /* 类型转换 */
        float* fdata = (float*)malloc(numel * sizeof(float));
        if (!fdata) {
            tensor_free(t);
            return NULL;
        }
        for (size_t i = 0; i < numel; i++) {
            fdata[i] = (float)data[i];
        }
        Tensor* result = tensor_from_f32(fdata, dims, ndim, dtype);
        free(fdata);
        tensor_free(t);
        return result;
    }

    return t;
}

void tensor_free(Tensor* tensor) {
    if (!tensor) return;

    if (tensor->owns_data && tensor->data) {
        free(tensor->data);
    }

    free(tensor);
}

Tensor* tensor_clone(const Tensor* tensor) {
    if (!tensor) return NULL;

    Tensor* t = (Tensor*)malloc(sizeof(Tensor));
    if (!t) return NULL;

    *t = *tensor;
    t->offset = 0;
    t->owns_data = true;

    /* 计算连续存储大小 */
    size_t numel = shape_numel(&tensor->shape);
    size_t total_bytes = numel * dtype_size(tensor->dtype);

    if (total_bytes > 0) {
        t->data = malloc(total_bytes);
        if (!t->data) {
            free(t);
            return NULL;
        }

        /* 复制数据，处理非连续情况 */
        if (tensor_is_contiguous(tensor)) {
            const uint8_t* src = (const uint8_t*)tensor->data + tensor->offset;
            memcpy(t->data, src, total_bytes);
        } else {
            /* 非连续存储，逐元素复制 */
            for (size_t i = 0; i < numel; i++) {
                float val = tensor_get_f32(tensor, i);
                tensor_set_f32(t, i, val);
            }
        }
    }

    /* 更新步幅为连续 */
    shape_strides(&tensor->shape, t->strides);

    return t;
}

Tensor* tensor_view(const Tensor* tensor) {
    if (!tensor) return NULL;

    Tensor* t = (Tensor*)malloc(sizeof(Tensor));
    if (!t) return NULL;

    *t = *tensor;
    t->owns_data = false;

    return t;
}

/* ============================================================================
 * 属性访问
 * ============================================================================ */

void* tensor_data_ptr(Tensor* t) {
    if (!t || !t->data) return NULL;
    return (uint8_t*)t->data + t->offset;
}

const void* tensor_data_ptr_const(const Tensor* t) {
    if (!t || !t->data) return NULL;
    return (const uint8_t*)t->data + t->offset;
}

bool tensor_is_contiguous(const Tensor* t) {
    if (!t || t->shape.ndim == 0) return true;

    size_t expected_strides[MYLLM_MAX_NDIM];
    shape_strides(&t->shape, expected_strides);

    for (size_t i = 0; i < t->shape.ndim; i++) {
        if (t->strides[i] != expected_strides[i]) {
            return false;
        }
    }
    return true;
}

/* ============================================================================
 * 数据访问
 * ============================================================================ */

float tensor_get_f32(const Tensor* t, size_t index) {
    if (!t || !t->data) return 0.0f;
    if (index >= shape_numel(&t->shape)) return 0.0f;

    size_t byte_offset = compute_linear_offset(t, index);
    const uint8_t* ptr = (const uint8_t*)t->data + byte_offset;

    switch (t->dtype) {
        case DTYPE_F32:
            return *(const float*)ptr;
        case DTYPE_F16:
            return f16_to_f32(*(const uint16_t*)ptr);
        case DTYPE_BF16:
            return bf16_to_f32(*(const uint16_t*)ptr);
        case DTYPE_I32:
            return (float)(*(const int32_t*)ptr);
        case DTYPE_I64:
            return (float)(*(const int64_t*)ptr);
        default:
            return 0.0f;
    }
}

void tensor_set_f32(Tensor* t, size_t index, float value) {
    if (!t || !t->data) return;
    if (index >= shape_numel(&t->shape)) return;

    size_t byte_offset = compute_linear_offset(t, index);
    uint8_t* ptr = (uint8_t*)t->data + byte_offset;

    switch (t->dtype) {
        case DTYPE_F32:
            *(float*)ptr = value;
            break;
        case DTYPE_F16:
            *(uint16_t*)ptr = f32_to_f16(value);
            break;
        case DTYPE_BF16:
            *(uint16_t*)ptr = f32_to_bf16(value);
            break;
        case DTYPE_I32:
            *(int32_t*)ptr = (int32_t)value;
            break;
        case DTYPE_I64:
            *(int64_t*)ptr = (int64_t)value;
            break;
        default:
            break;
    }
}

float tensor_get_at(const Tensor* t, const size_t* indices) {
    if (!t || !indices) return 0.0f;

    size_t linear_index = shape_offset(&t->shape, indices);
    return tensor_get_f32(t, linear_index);
}

void tensor_set_at(Tensor* t, const size_t* indices, float value) {
    if (!t || !indices) return;

    size_t linear_index = shape_offset(&t->shape, indices);
    tensor_set_f32(t, linear_index, value);
}

int tensor_to_f32(const Tensor* t, float* out, size_t out_size) {
    if (!t || !out) return -1;

    size_t numel = shape_numel(&t->shape);
    if (out_size < numel) return -1;

    for (size_t i = 0; i < numel; i++) {
        out[i] = tensor_get_f32(t, i);
    }

    return 0;
}

int tensor_set_data_f32(Tensor* t, const float* data, size_t data_size) {
    if (!t || !data) return -1;

    size_t numel = shape_numel(&t->shape);
    if (data_size < numel) return -1;

    for (size_t i = 0; i < numel; i++) {
        tensor_set_f32(t, i, data[i]);
    }

    return 0;
}

/* ============================================================================
 * 形状操作
 * ============================================================================ */

Tensor* tensor_reshape(const Tensor* t, const ssize_t* new_dims, size_t new_ndim) {
    if (!t) return NULL;

    Shape new_shape;
    if (shape_reshape(&t->shape, new_dims, new_ndim, &new_shape) != 0) {
        return NULL;
    }

    if (!tensor_is_contiguous(t)) {
        /* 非连续存储需要先复制 */
        Tensor* contiguous = tensor_contiguous(t);
        if (!contiguous) return NULL;

        Tensor* result = (Tensor*)malloc(sizeof(Tensor));
        if (!result) {
            tensor_free(contiguous);
            return NULL;
        }

        result->data = contiguous->data;
        result->shape = new_shape;
        shape_strides(&new_shape, result->strides);
        result->dtype = t->dtype;
        result->device = t->device;
        result->offset = 0;
        result->owns_data = true;

        /* 释放 contiguous 结构但保留数据 */
        free(contiguous);

        return result;
    }

    /* 连续存储：创建视图 */
    Tensor* result = (Tensor*)malloc(sizeof(Tensor));
    if (!result) return NULL;

    result->data = t->data;
    result->shape = new_shape;
    shape_strides(&new_shape, result->strides);
    result->dtype = t->dtype;
    result->device = t->device;
    result->offset = t->offset;
    result->owns_data = false;

    return result;
}

Tensor* tensor_transpose(const Tensor* t, size_t dim1, size_t dim2) {
    if (!t) return NULL;
    if (dim1 >= t->shape.ndim || dim2 >= t->shape.ndim) return NULL;

    Tensor* result = (Tensor*)malloc(sizeof(Tensor));
    if (!result) return NULL;

    *result = *t;
    result->shape = shape_transpose(&t->shape, dim1, dim2);
    result->owns_data = false;

    /* 交换步幅 */
    size_t tmp = result->strides[dim1];
    result->strides[dim1] = result->strides[dim2];
    result->strides[dim2] = tmp;

    return result;
}

Tensor* tensor_permute(const Tensor* t, const size_t* dims, size_t ndim) {
    if (!t || !dims) return NULL;
    if (ndim != t->shape.ndim) return NULL;

    /* 验证 dims 是否为有效排列 */
    bool seen[MYLLM_MAX_NDIM] = {false};
    for (size_t i = 0; i < ndim; i++) {
        if (dims[i] >= ndim || seen[dims[i]]) {
            return NULL;
        }
        seen[dims[i]] = true;
    }

    Tensor* result = (Tensor*)malloc(sizeof(Tensor));
    if (!result) return NULL;

    *result = *t;
    result->owns_data = false;

    /* 重排形状和步幅 */
    for (size_t i = 0; i < ndim; i++) {
        result->shape.dims[i] = t->shape.dims[dims[i]];
        result->strides[i] = t->strides[dims[i]];
    }

    return result;
}

Tensor* tensor_squeeze(const Tensor* t, int dim) {
    if (!t) return NULL;

    Tensor* result = (Tensor*)malloc(sizeof(Tensor));
    if (!result) return NULL;

    *result = *t;
    result->shape = shape_squeeze(&t->shape, dim);
    result->owns_data = false;

    /* 更新步幅 */
    if (dim < 0) {
        /* 移除所有大小为1的维度 */
        size_t j = 0;
        for (size_t i = 0; i < t->shape.ndim; i++) {
            if (t->shape.dims[i] != 1) {
                result->strides[j++] = t->strides[i];
            }
        }
    } else if ((size_t)dim < t->shape.ndim && t->shape.dims[dim] == 1) {
        /* 移除指定维度 */
        size_t j = 0;
        for (size_t i = 0; i < t->shape.ndim; i++) {
            if ((int)i != dim) {
                result->strides[j++] = t->strides[i];
            }
        }
    }

    return result;
}

Tensor* tensor_unsqueeze(const Tensor* t, size_t dim) {
    if (!t) return NULL;

    Tensor* result = (Tensor*)malloc(sizeof(Tensor));
    if (!result) return NULL;

    *result = *t;
    result->shape = shape_unsqueeze(&t->shape, dim);
    result->owns_data = false;

    /* 插入步幅 */
    size_t insert_pos = (dim > t->shape.ndim) ? t->shape.ndim : dim;
    size_t new_stride = (insert_pos >= t->shape.ndim) ? 1 :
        t->strides[insert_pos] * t->shape.dims[insert_pos];

    size_t j = 0;
    for (size_t i = 0; i <= t->shape.ndim; i++) {
        if (i == insert_pos) {
            result->strides[i] = new_stride;
        } else {
            result->strides[i] = t->strides[j++];
        }
    }

    return result;
}

Tensor* tensor_contiguous(const Tensor* t) {
    if (!t) return NULL;

    if (tensor_is_contiguous(t)) {
        return tensor_view(t);
    }

    return tensor_clone(t);
}

Tensor* tensor_slice(const Tensor* t, const size_t* starts, const size_t* ends, size_t n_ranges) {
    if (!t || !starts || !ends) return NULL;
    if (n_ranges > t->shape.ndim) return NULL;

    Tensor* result = (Tensor*)malloc(sizeof(Tensor));
    if (!result) return NULL;

    *result = *t;
    result->owns_data = false;

    size_t new_offset = t->offset;

    for (size_t i = 0; i < n_ranges; i++) {
        if (starts[i] > ends[i] || ends[i] > t->shape.dims[i]) {
            free(result);
            return NULL;
        }

        new_offset += starts[i] * t->strides[i] * dtype_size(t->dtype);
        result->shape.dims[i] = ends[i] - starts[i];
    }

    result->offset = new_offset;

    return result;
}

Tensor* tensor_index(const Tensor* t, const size_t* indices, size_t n_indices) {
    if (!t || !indices) return NULL;
    if (n_indices > t->shape.ndim) return NULL;

    Tensor* result = (Tensor*)malloc(sizeof(Tensor));
    if (!result) return NULL;

    *result = *t;
    result->owns_data = false;

    /* 计算偏移 */
    size_t offset = t->offset;
    for (size_t i = 0; i < n_indices; i++) {
        if (indices[i] >= t->shape.dims[i]) {
            free(result);
            return NULL;
        }
        offset += indices[i] * t->strides[i] * dtype_size(t->dtype);
    }

    result->offset = offset;
    result->shape.ndim = t->shape.ndim - n_indices;

    /* 更新形状和步幅 */
    for (size_t i = n_indices; i < t->shape.ndim; i++) {
        result->shape.dims[i - n_indices] = t->shape.dims[i];
        result->strides[i - n_indices] = t->strides[i];
    }

    return result;
}

/* ============================================================================
 * 工具函数
 * ============================================================================ */

void tensor_print_info(const Tensor* t) {
    if (!t) {
        printf("Tensor(NULL)\n");
        return;
    }

    printf("Tensor(");
    shape_print(&t->shape);
    printf(", dtype=%s", dtype_name(t->dtype));
    printf(", device=%s", t->device.type == DEVICE_CPU ? "cpu" : "cuda");
    printf(", contiguous=%s", tensor_is_contiguous(t) ? "true" : "false");
    printf(")\n");
}

void tensor_print(const Tensor* t, size_t max_elements) {
    if (!t) {
        printf("Tensor(NULL)\n");
        return;
    }

    tensor_print_info(t);

    size_t numel = shape_numel(&t->shape);
    size_t print_count = (max_elements > 0 && max_elements < numel) ? max_elements : numel;

    printf("[");
    for (size_t i = 0; i < print_count; i++) {
        if (i > 0) printf(", ");
        printf("%.6f", tensor_get_f32(t, i));
    }
    if (print_count < numel) {
        printf(", ...");
    }
    printf("]\n");
}

bool tensor_equals(const Tensor* a, const Tensor* b, float tolerance) {
    if (!a || !b) return false;
    if (!shape_equals(&a->shape, &b->shape)) return false;
    if (a->dtype != b->dtype) return false;

    size_t numel = shape_numel(&a->shape);
    for (size_t i = 0; i < numel; i++) {
        float va = tensor_get_f32(a, i);
        float vb = tensor_get_f32(b, i);
        if (fabsf(va - vb) > tolerance) {
            return false;
        }
    }

    return true;
}
