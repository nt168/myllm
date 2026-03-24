/**
 * @file shape.c
 * @brief 形状操作实现
 */

#include "tensor/shape.h"
#include <stdlib.h>
#include <string.h>
#include <stdarg.h>
#include <stdio.h>

Shape shape_scalar(void) {
    Shape s;
    s.ndim = 0;
    memset(s.dims, 0, sizeof(s.dims));
    return s;
}

Shape shape_new(const size_t* dims, size_t ndim) {
    Shape s;
    s.ndim = (ndim > MYLLM_MAX_NDIM) ? MYLLM_MAX_NDIM : ndim;
    memset(s.dims, 0, sizeof(s.dims));
    if (dims && ndim > 0) {
        memcpy(s.dims, dims, s.ndim * sizeof(size_t));
    }
    return s;
}

Shape shape_from_dims(size_t ndim, ...) {
    Shape s;
    s.ndim = (ndim > MYLLM_MAX_NDIM) ? MYLLM_MAX_NDIM : ndim;
    memset(s.dims, 0, sizeof(s.dims));

    va_list args;
    va_start(args, ndim);
    for (size_t i = 0; i < s.ndim; i++) {
        s.dims[i] = va_arg(args, size_t);
    }
    va_end(args);

    return s;
}

size_t shape_numel(const Shape* s) {
    if (!s || s->ndim == 0) {
        return 1;  /* 标量有1个元素 */
    }

    size_t numel = 1;
    for (size_t i = 0; i < s->ndim; i++) {
        numel *= s->dims[i];
    }
    return numel;
}

bool shape_is_empty(const Shape* s) {
    if (!s) return true;
    for (size_t i = 0; i < s->ndim; i++) {
        if (s->dims[i] == 0) {
            return true;
        }
    }
    return false;
}

void shape_strides(const Shape* s, size_t* strides) {
    if (!s || !strides) return;

    if (s->ndim == 0) {
        return;  /* 标量无步幅 */
    }

    /* 从后向前计算步幅 */
    size_t stride = 1;
    for (ssize_t i = (ssize_t)s->ndim - 1; i >= 0; i--) {
        strides[i] = stride;
        stride *= s->dims[i];
    }
}

size_t shape_offset(const Shape* s, const size_t* indices) {
    if (!s || !indices) return 0;

    size_t strides[MYLLM_MAX_NDIM];
    shape_strides(s, strides);

    size_t offset = 0;
    for (size_t i = 0; i < s->ndim; i++) {
        offset += indices[i] * strides[i];
    }
    return offset;
}

int shape_broadcast(const Shape* a, const Shape* b, Shape* result) {
    if (!a || !b || !result) return -1;

    size_t max_ndim = (a->ndim > b->ndim) ? a->ndim : b->ndim;
    result->ndim = max_ndim;
    memset(result->dims, 0, sizeof(result->dims));

    /* 从右向左对齐维度 */
    for (size_t i = 0; i < max_ndim; i++) {
        size_t a_dim = 1;
        size_t b_dim = 1;

        if (i < a->ndim) {
            a_dim = a->dims[a->ndim - 1 - i];
        }
        if (i < b->ndim) {
            b_dim = b->dims[b->ndim - 1 - i];
        }

        /* 检查兼容性 */
        if (a_dim != 1 && b_dim != 1 && a_dim != b_dim) {
            return -1;  /* 不兼容 */
        }

        result->dims[max_ndim - 1 - i] = (a_dim > b_dim) ? a_dim : b_dim;
    }

    return 0;
}

int shape_reshape(const Shape* s, const ssize_t* new_dims, size_t new_ndim, Shape* result) {
    if (!s || !new_dims || !result) return -1;
    if (new_ndim > MYLLM_MAX_NDIM) return -1;

    size_t total = shape_numel(s);

    /* 统计 -1 出现次数 */
    int minus_one_count = 0;
    ssize_t minus_one_pos = -1;

    for (size_t i = 0; i < new_ndim; i++) {
        if (new_dims[i] == -1) {
            minus_one_count++;
            minus_one_pos = (ssize_t)i;
        }
    }

    if (minus_one_count > 1) {
        return -1;  /* 只能有一个 -1 */
    }

    /* 计算已知维度的乘积 */
    size_t known_product = 1;
    for (size_t i = 0; i < new_ndim; i++) {
        if (new_dims[i] > 0) {
            known_product *= (size_t)new_dims[i];
        } else if (new_dims[i] < -1) {
            return -1;  /* 无效维度 */
        }
    }

    /* 构建结果形状 */
    result->ndim = new_ndim;
    memset(result->dims, 0, sizeof(result->dims));

    for (size_t i = 0; i < new_ndim; i++) {
        if (new_dims[i] == -1) {
            /* 推断维度 */
            if (known_product == 0 || total % known_product != 0) {
                return -1;
            }
            result->dims[i] = total / known_product;
        } else {
            result->dims[i] = (size_t)new_dims[i];
        }
    }

    /* 验证元素总数 */
    if (shape_numel(result) != total) {
        return -1;
    }

    return 0;
}

Shape shape_squeeze(const Shape* s, int dim) {
    if (!s) return shape_scalar();

    Shape result;
    memset(&result, 0, sizeof(result));

    if (dim < 0) {
        /* 移除所有大小为1的维度 */
        size_t j = 0;
        for (size_t i = 0; i < s->ndim; i++) {
            if (s->dims[i] != 1) {
                result.dims[j++] = s->dims[i];
            }
        }
        result.ndim = j;
    } else if ((size_t)dim < s->ndim && s->dims[dim] == 1) {
        /* 移除指定维度 */
        size_t j = 0;
        for (size_t i = 0; i < s->ndim; i++) {
            if ((int)i != dim) {
                result.dims[j++] = s->dims[i];
            }
        }
        result.ndim = j;
    } else {
        /* 不做改变 */
        result = *s;
    }

    return result;
}

Shape shape_unsqueeze(const Shape* s, size_t dim) {
    if (!s) return shape_scalar();

    Shape result;
    memset(&result, 0, sizeof(result));

    size_t insert_pos = (dim > s->ndim) ? s->ndim : dim;

    result.ndim = s->ndim + 1;
    if (result.ndim > MYLLM_MAX_NDIM) {
        result.ndim = MYLLM_MAX_NDIM;
        return result;
    }

    /* 复制维度并在指定位置插入1 */
    size_t j = 0;
    for (size_t i = 0; i < result.ndim; i++) {
        if (i == insert_pos) {
            result.dims[i] = 1;
        } else {
            result.dims[i] = s->dims[j++];
        }
    }

    return result;
}

Shape shape_transpose(const Shape* s, size_t dim1, size_t dim2) {
    if (!s) return shape_scalar();

    Shape result = *s;

    if (dim1 < s->ndim && dim2 < s->ndim) {
        size_t tmp = result.dims[dim1];
        result.dims[dim1] = result.dims[dim2];
        result.dims[dim2] = tmp;
    }

    return result;
}

Shape shape_clone(const Shape* s) {
    if (!s) return shape_scalar();
    return *s;
}

bool shape_equals(const Shape* a, const Shape* b) {
    if (!a || !b) return false;
    if (a->ndim != b->ndim) return false;

    for (size_t i = 0; i < a->ndim; i++) {
        if (a->dims[i] != b->dims[i]) {
            return false;
        }
    }
    return true;
}

void shape_print(const Shape* s) {
    if (!s) {
        printf("Shape(NULL)");
        return;
    }

    if (s->ndim == 0) {
        printf("Shape(scalar)");
        return;
    }

    printf("Shape([");
    for (size_t i = 0; i < s->ndim; i++) {
        if (i > 0) printf(", ");
        printf("%zu", s->dims[i]);
    }
    printf("])");
}
