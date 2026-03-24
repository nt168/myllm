/**
 * @file tokenizer.h
 * @brief 分词器模块
 *
 * 支持 BPE (Byte-Pair Encoding) 分词器，兼容 HuggingFace tokenizers 格式。
 */

#ifndef MYLLM_TOKENIZER_H
#define MYLLM_TOKENIZER_H

#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

/* ============================================================================
 * 常量定义
 * ============================================================================ */

/** 最大 token 长度 */
#define TOKENIZER_MAX_TOKEN_LEN 256

/** 默认词表大小 */
#define TOKENIZER_DEFAULT_VOCAB_SIZE 32000

/** 最大编码长度 */
#define TOKENIZER_MAX_ENCODE_LENGTH 4096

/* ============================================================================
 * 数据结构
 * ============================================================================ */

/**
 * @brief BPE 分词器结构 (不透明指针)
 */
typedef struct BPETokenizer BPETokenizer;

/* ============================================================================
 * 生命周期管理
 * ============================================================================ */

/**
 * @brief 创建空的 BPE 分词器
 */
BPETokenizer* bpe_tokenizer_new(void);

/**
 * @brief 从 tokenizer.json 文件加载分词器
 */
BPETokenizer* bpe_tokenizer_from_file(const char* path);

/**
 * @brief 从模型目录加载分词器
 */
BPETokenizer* bpe_tokenizer_from_dir(const char* model_dir);

/**
 * @brief 释放分词器
 */
void bpe_tokenizer_free(BPETokenizer* tokenizer);

/* ============================================================================
 * 词表操作
 * ============================================================================ */

/**
 * @brief 获取词表大小
 */
size_t bpe_tokenizer_vocab_size(const BPETokenizer* tokenizer);

/**
 * @brief 将 token 字符串转换为 ID
 */
int32_t bpe_tokenizer_token_to_id(const BPETokenizer* tokenizer, const char* token);

/**
 * @brief 将 token ID 转换为字符串
 */
const char* bpe_tokenizer_id_to_token(const BPETokenizer* tokenizer, int32_t id);

/* ============================================================================
 * 特殊 Token
 * ============================================================================ */

/**
 * @brief 获取 BOS token ID
 */
int32_t bpe_tokenizer_bos_id(const BPETokenizer* tokenizer);

/**
 * @brief 获取 EOS token ID
 */
int32_t bpe_tokenizer_eos_id(const BPETokenizer* tokenizer);

/**
 * @brief 获取 PAD token ID
 */
int32_t bpe_tokenizer_pad_id(const BPETokenizer* tokenizer);

/**
 * @brief 获取 UNK token ID
 */
int32_t bpe_tokenizer_unk_id(const BPETokenizer* tokenizer);

/**
 * @brief 获取 BOS token 字符串
 */
const char* bpe_tokenizer_bos_token(const BPETokenizer* tokenizer);

/**
 * @brief 获取 EOS token 字符串
 */
const char* bpe_tokenizer_eos_token(const BPETokenizer* tokenizer);

/**
 * @brief 检查是否为特殊 token
 */
bool bpe_tokenizer_is_special_token(const BPETokenizer* tokenizer, int32_t id);

/* ============================================================================
 * 编码/解码
 * ============================================================================ */

/**
 * @brief 将文本编码为 token IDs
 *
 * @param tokenizer 分词器指针
 * @param text 输入文本
 * @param add_special_tokens 是否添加特殊 token (BOS/EOS)
 * @param ids 输出 token ID 数组
 * @param max_ids 数组最大容量
 * @return 实际编码的 token 数量，失败返回 -1
 */
int bpe_tokenizer_encode(
    const BPETokenizer* tokenizer,
    const char* text,
    bool add_special_tokens,
    int32_t* ids,
    size_t max_ids
);

/**
 * @brief 将 token IDs 解码为文本
 *
 * @param tokenizer 分词器指针
 * @param ids token ID 数组
 * @param num_ids token 数量
 * @param skip_special_tokens 是否跳过特殊 token
 * @param out 输出文本缓冲区
 * @param max_out 缓冲区最大容量
 * @return 实际解码的字符数，失败返回 -1
 */
int bpe_tokenizer_decode(
    const BPETokenizer* tokenizer,
    const int32_t* ids,
    size_t num_ids,
    bool skip_special_tokens,
    char* out,
    size_t max_out
);

/* ============================================================================
 * UTF-8 工具函数
 * ============================================================================ */

/**
 * @brief 计算 UTF-8 字符串的字符数
 */
size_t utf8_strlen(const char* str);

/**
 * @brief 获取 UTF-8 字符的字节长度
 */
int utf8_char_len(const char* str);

#ifdef __cplusplus
}
#endif

#endif /* MYLLM_TOKENIZER_H */
