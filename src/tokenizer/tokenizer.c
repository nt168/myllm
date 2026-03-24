/**
 * @file tokenizer.c
 * @brief BPE 分词器实现
 */

#include "tokenizer/tokenizer.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <ctype.h>

/* ============================================================================
 * 内部常量
 * ============================================================================ */

#define HASH_SIZE 65537
#define MAX_VOCAB_SIZE 200000
#define MAX_MERGES 500000

/* ============================================================================
 * 内部数据结构
 * ============================================================================ */

/**
 * @brief 哈希表条目
 */
typedef struct TokenHashEntry {
    char* key;
    int32_t value;
    struct TokenHashEntry* next;
} TokenHashEntry;

/**
 * @brief BPE 分词器结构
 */
struct BPETokenizer {
    /* 词表 */
    char** vocab;
    int32_t vocab_size;

    /* Token -> ID 哈希表 */
    TokenHashEntry** token_to_id;
    int hash_size;

    /* BPE 合并规则 */
    char** merge_tokens;
    int32_t* merge_ranks;
    int32_t num_merges;

    /* 特殊 token IDs */
    int32_t bos_id;
    int32_t eos_id;
    int32_t pad_id;
    int32_t unk_id;

    /* 特殊 token 字符串 */
    char* bos_token;
    char* eos_token;
    char* pad_token;
    char* unk_token;

    /* 特殊 token ID 集合 */
    int32_t* special_ids;
    int32_t num_special_ids;
};

/* ============================================================================
 * UTF-8 工具函数
 * ============================================================================ */

int utf8_char_len(const char* str) {
    if (!str || !*str) return 0;

    unsigned char c = (unsigned char)str[0];
    if ((c & 0x80) == 0) return 1;
    if ((c & 0xE0) == 0xC0) return 2;
    if ((c & 0xF0) == 0xE0) return 3;
    if ((c & 0xF8) == 0xF0) return 4;
    return 1;
}

size_t utf8_strlen(const char* str) {
    if (!str) return 0;

    size_t len = 0;
    const char* p = str;
    while (*p) {
        int char_len = utf8_char_len(p);
        p += (char_len > 0) ? char_len : 1;
        len++;
    }
    return len;
}

/* ============================================================================
 * 哈希表实现
 * ============================================================================ */

static uint32_t hash_string(const char* str) {
    uint32_t hash = 5381;
    int c;
    while ((c = *str++)) {
        hash = ((hash << 5) + hash) + c;
    }
    return hash;
}

static TokenHashEntry** hash_table_new(int size) {
    return (TokenHashEntry**)calloc(size, sizeof(TokenHashEntry*));
}

static void hash_table_free(TokenHashEntry** table, int size) {
    if (!table) return;
    for (int i = 0; i < size; i++) {
        TokenHashEntry* entry = table[i];
        while (entry) {
            TokenHashEntry* next = entry->next;
            free(entry->key);
            free(entry);
            entry = next;
        }
    }
    free(table);
}

static void hash_table_insert(TokenHashEntry** table, int size, const char* key, int32_t value) {
    if (!table || !key) return;

    uint32_t idx = hash_string(key) % size;
    TokenHashEntry* entry = (TokenHashEntry*)malloc(sizeof(TokenHashEntry));
    if (!entry) return;

    entry->key = strdup(key);
    entry->value = value;
    entry->next = table[idx];
    table[idx] = entry;
}

static int32_t hash_table_lookup(TokenHashEntry** table, int size, const char* key, int32_t default_val) {
    if (!table || !key) return default_val;

    uint32_t idx = hash_string(key) % size;
    TokenHashEntry* entry = table[idx];
    while (entry) {
        if (strcmp(entry->key, key) == 0) {
            return entry->value;
        }
        entry = entry->next;
    }
    return default_val;
}

/* ============================================================================
 * 内部 JSON 解析
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

static char* extract_json_string(const char* json, const char* key) {
    const char* pos = find_key(json, key);
    if (!pos) return NULL;

    pos = strchr(pos + strlen(key) + 2, ':');
    if (!pos) return NULL;

    pos = skip_ws(pos + 1);
    if (*pos != '"') return NULL;

    pos++;
    const char* end = pos;
    while (*end && *end != '"') {
        if (*end == '\\') end++;
        if (*end) end++;
    }
    if (!*end) return NULL;

    size_t len = end - pos;
    char* result = (char*)malloc(len + 1);
    if (!result) return NULL;

    size_t j = 0;
    for (size_t i = 0; i < len; i++) {
        if (pos[i] == '\\' && i + 1 < len) {
            i++;
            switch (pos[i]) {
                case 'n': result[j++] = '\n'; break;
                case 'r': result[j++] = '\r'; break;
                case 't': result[j++] = '\t'; break;
                case '\\': result[j++] = '\\'; break;
                case '"': result[j++] = '"'; break;
                default: result[j++] = pos[i]; break;
            }
        } else {
            result[j++] = pos[i];
        }
    }
    result[j] = '\0';
    return result;
}

static int extract_json_int(const char* json, const char* key, int default_val) {
    const char* pos = find_key(json, key);
    if (!pos) return default_val;

    pos = strchr(pos + strlen(key) + 2, ':');
    if (!pos) return default_val;

    pos = skip_ws(pos + 1);
    return (int)strtol(pos, NULL, 10);
}

static bool extract_json_bool(const char* json, const char* key, bool default_val) {
    const char* pos = find_key(json, key);
    if (!pos) return default_val;

    pos = strchr(pos + strlen(key) + 2, ':');
    if (!pos) return default_val;

    pos = skip_ws(pos + 1);
    if (strncmp(pos, "true", 4) == 0) return true;
    if (strncmp(pos, "false", 5) == 0) return false;
    return default_val;
}

/* ============================================================================
 * 词表解析
 * ============================================================================ */

static int parse_vocab(const char* json, BPETokenizer* tokenizer) {
    /* 首先找到 "model" 对象 */
    const char* model_key = find_key(json, "model");
    if (!model_key) {
        fprintf(stderr, "Tokenizer: 'model' key not found\n");
        return -1;
    }

    /* 找到 model 对象的起始大括号 */
    const char* model_brace = strchr(model_key, '{');
    if (!model_brace) {
        fprintf(stderr, "Tokenizer: 'model' object brace not found\n");
        return -1;
    }

    /* 在 model 对象内查找 "vocab" */
    const char* vocab_key = strstr(model_brace, "\"vocab\"");
    if (!vocab_key) {
        fprintf(stderr, "Tokenizer: 'vocab' not found in 'model' object\n");
        return -1;
    }

    /* 找到 vocab 对象的起始大括号 */
    const char* vocab_start = strchr(vocab_key, '{');
    if (!vocab_start) {
        fprintf(stderr, "Tokenizer: vocab object brace not found\n");
        return -1;
    }

    /* 计算词表大小 - 需要正确处理字符串中的转义引号 */
    int vocab_size = 0;
    const char* p = vocab_start + 1;
    int depth = 1;  /* 已经在 vocab 对象内 */
    bool in_string = false;

    while (*p) {
        if (in_string) {
            if (*p == '\\' && *(p+1)) {
                p += 2;  /* 跳过转义字符 */
                continue;
            }
            if (*p == '"') {
                in_string = false;
            }
        } else {
            if (*p == '"') {
                in_string = true;
            } else if (*p == '{') {
                depth++;
            } else if (*p == '}') {
                depth--;
                if (depth == 0) break;
            } else if (*p == ':' && depth == 1) {
                vocab_size++;
            }
        }
        p++;
    }

    if (vocab_size == 0 || vocab_size > MAX_VOCAB_SIZE) {
        fprintf(stderr, "Tokenizer: invalid vocab size %d\n", vocab_size);
        return -1;
    }

    fprintf(stderr, "Tokenizer: loading vocab with %d entries\n", vocab_size);

    tokenizer->vocab_size = vocab_size;
    tokenizer->vocab = (char**)calloc(vocab_size, sizeof(char*));
    if (!tokenizer->vocab) return -1;

    tokenizer->hash_size = HASH_SIZE;
    tokenizer->token_to_id = hash_table_new(tokenizer->hash_size);
    if (!tokenizer->token_to_id) return -1;

    /* 解析词表 */
    p = vocab_start + 1;
    int count = 0;
    depth = 1;
    in_string = false;

    while (*p && depth > 0 && count < vocab_size) {
        /* 跳过空白和非键字符 */
        while (*p && depth > 0) {
            if (in_string) {
                break;  /* 在字符串内，开始解析 token */
            }
            if (*p == '"') {
                in_string = true;
                p++;
                break;
            }
            if (*p == '{') depth++;
            else if (*p == '}') {
                depth--;
                if (depth == 0) break;
            }
            p++;
        }

        if (depth == 0 || !*p) break;
        if (!in_string) continue;

        /* 解析 token 字符串 */
        const char* token_start = p;
        while (*p) {
            if (*p == '\\' && *(p+1)) {
                p += 2;
                continue;
            }
            if (*p == '"') break;
            p++;
        }
        if (!*p) break;

        size_t token_len = p - token_start;
        char* token = (char*)malloc(token_len + 1);
        if (!token) break;

        /* 处理转义字符 */
        size_t j = 0;
        for (size_t i = 0; i < token_len; i++) {
            if (token_start[i] == '\\' && i + 1 < token_len) {
                i++;
                switch (token_start[i]) {
                    case 'n': token[j++] = '\n'; break;
                    case 'r': token[j++] = '\r'; break;
                    case 't': token[j++] = '\t'; break;
                    case '\\': token[j++] = '\\'; break;
                    case '"': token[j++] = '"'; break;
                    case 'u': {
                        /* Unicode escape \uXXXX */
                        if (i + 4 < token_len) {
                            char hex[5] = {token_start[i+1], token_start[i+2],
                                          token_start[i+3], token_start[i+4], 0};
                            unsigned int codepoint = (unsigned int)strtol(hex, NULL, 16);
                            /* 简化：只处理 ASCII 和基本多语言平面 */
                            if (codepoint < 0x80) {
                                token[j++] = (char)codepoint;
                            } else if (codepoint < 0x800) {
                                token[j++] = (char)(0xC0 | (codepoint >> 6));
                                token[j++] = (char)(0x80 | (codepoint & 0x3F));
                            } else {
                                token[j++] = (char)(0xE0 | (codepoint >> 12));
                                token[j++] = (char)(0x80 | ((codepoint >> 6) & 0x3F));
                                token[j++] = (char)(0x80 | (codepoint & 0x3F));
                            }
                            i += 4;
                        }
                        break;
                    }
                    default: token[j++] = token_start[i]; break;
                }
            } else {
                token[j++] = token_start[i];
            }
        }
        token[j] = '\0';

        p++;  /* 跳过结束引号 */
        in_string = false;

        /* 跳过空白到冒号 */
        while (*p && *p != ':' && *p != '}' && *p != '"') p++;
        if (*p != ':') {
            free(token);
            continue;
        }
        p++;  /* 跳过冒号 */

        /* 跳过空白 */
        while (*p && (*p == ' ' || *p == '\t' || *p == '\n' || *p == '\r')) p++;

        /* 读取 ID */
        int id = (int)strtol(p, (char**)&p, 10);

        if (id >= 0 && id < vocab_size) {
            tokenizer->vocab[id] = token;
            hash_table_insert(tokenizer->token_to_id, tokenizer->hash_size, token, id);
            count++;
        } else {
            free(token);
        }

        /* 跳过逗号和空白 */
        while (*p && (*p == ',' || *p == ' ' || *p == '\t' || *p == '\n' || *p == '\r')) p++;
    }

    tokenizer->vocab_size = count;
    fprintf(stderr, "Tokenizer: loaded %d tokens\n", count);
    return count;
}

/* ============================================================================
 * 合并规则解析
 * ============================================================================ */

static int parse_merges(const char* json, BPETokenizer* tokenizer) {
    const char* merges_start = find_key(json, "merges");
    if (!merges_start) return 0;

    merges_start = strchr(merges_start, '[');
    if (!merges_start) return 0;
    merges_start++;

    int num_merges = 0;
    const char* p = merges_start;
    while (*p && *p != ']') {
        if (*p == '"') num_merges++;
        p++;
    }

    if (num_merges == 0 || num_merges > MAX_MERGES) return 0;

    tokenizer->num_merges = num_merges;
    tokenizer->merge_tokens = (char**)calloc(num_merges, sizeof(char*));
    tokenizer->merge_ranks = (int32_t*)calloc(num_merges, sizeof(int32_t));
    if (!tokenizer->merge_tokens || !tokenizer->merge_ranks) return -1;

    p = merges_start;
    int count = 0;
    while (*p && *p != ']' && count < num_merges) {
        p = skip_ws(p);
        if (*p != '"') {
            p++;
            continue;
        }
        p++;

        const char* merge_start = p;
        while (*p && *p != '"') {
            if (*p == '\\') p++;
            if (*p) p++;
        }
        if (!*p) break;

        size_t merge_len = p - merge_start;
        char* merge = (char*)malloc(merge_len + 1);
        if (!merge) break;

        memcpy(merge, merge_start, merge_len);
        merge[merge_len] = '\0';

        tokenizer->merge_tokens[count] = merge;
        tokenizer->merge_ranks[count] = count;
        count++;

        p++;
    }

    tokenizer->num_merges = count;
    return count;
}

/* ============================================================================
 * 特殊 token 解析
 * ============================================================================ */

static void parse_special_tokens(const char* json, BPETokenizer* tokenizer) {
    char* bos = extract_json_string(json, "bos_token");
    char* eos = extract_json_string(json, "eos_token");
    char* unk = extract_json_string(json, "unk_token");
    char* pad = extract_json_string(json, "pad_token");

    if (bos) {
        tokenizer->bos_token = bos;
        tokenizer->bos_id = hash_table_lookup(tokenizer->token_to_id, tokenizer->hash_size, bos, -1);
    }
    if (eos) {
        tokenizer->eos_token = eos;
        tokenizer->eos_id = hash_table_lookup(tokenizer->token_to_id, tokenizer->hash_size, eos, -1);
    }
    if (unk) {
        tokenizer->unk_token = unk;
        tokenizer->unk_id = hash_table_lookup(tokenizer->token_to_id, tokenizer->hash_size, unk, 0);
    }
    if (pad) {
        tokenizer->pad_token = pad;
        tokenizer->pad_id = hash_table_lookup(tokenizer->token_to_id, tokenizer->hash_size, pad, 0);
    }

    /* 收集特殊 token IDs */
    int count = 0;
    if (tokenizer->bos_id >= 0) count++;
    if (tokenizer->eos_id >= 0) count++;
    if (tokenizer->unk_id >= 0) count++;
    if (tokenizer->pad_id >= 0) count++;

    if (count > 0) {
        tokenizer->special_ids = (int32_t*)malloc(count * sizeof(int32_t));
        if (tokenizer->special_ids) {
            int idx = 0;
            if (tokenizer->bos_id >= 0) tokenizer->special_ids[idx++] = tokenizer->bos_id;
            if (tokenizer->eos_id >= 0) tokenizer->special_ids[idx++] = tokenizer->eos_id;
            if (tokenizer->unk_id >= 0) tokenizer->special_ids[idx++] = tokenizer->unk_id;
            if (tokenizer->pad_id >= 0) tokenizer->special_ids[idx++] = tokenizer->pad_id;
            tokenizer->num_special_ids = idx;
        }
    }
}

/* ============================================================================
 * 生命周期管理
 * ============================================================================ */

BPETokenizer* bpe_tokenizer_new(void) {
    BPETokenizer* tokenizer = (BPETokenizer*)calloc(1, sizeof(BPETokenizer));
    if (!tokenizer) return NULL;

    tokenizer->vocab_size = 0;
    tokenizer->vocab = NULL;
    tokenizer->token_to_id = NULL;
    tokenizer->hash_size = 0;
    tokenizer->merge_tokens = NULL;
    tokenizer->merge_ranks = NULL;
    tokenizer->num_merges = 0;
    tokenizer->bos_id = -1;
    tokenizer->eos_id = -1;
    tokenizer->pad_id = 0;
    tokenizer->unk_id = 0;
    tokenizer->bos_token = NULL;
    tokenizer->eos_token = NULL;
    tokenizer->pad_token = NULL;
    tokenizer->unk_token = NULL;
    tokenizer->special_ids = NULL;
    tokenizer->num_special_ids = 0;

    return tokenizer;
}

BPETokenizer* bpe_tokenizer_from_file(const char* path) {
    if (!path) return NULL;

    FILE* f = fopen(path, "r");
    if (!f) {
        fprintf(stderr, "Tokenizer: cannot open file '%s'\n", path);
        return NULL;
    }

    fseek(f, 0, SEEK_END);
    long size = ftell(f);
    fseek(f, 0, SEEK_SET);

    char* json = (char*)malloc(size + 1);
    if (!json) {
        fclose(f);
        return NULL;
    }

    if (fread(json, 1, size, f) != (size_t)size) {
        free(json);
        fclose(f);
        return NULL;
    }
    json[size] = '\0';
    fclose(f);

    BPETokenizer* tokenizer = bpe_tokenizer_new();
    if (!tokenizer) {
        free(json);
        return NULL;
    }

    if (parse_vocab(json, tokenizer) < 0) {
        free(json);
        bpe_tokenizer_free(tokenizer);
        return NULL;
    }

    parse_merges(json, tokenizer);
    parse_special_tokens(json, tokenizer);

    free(json);
    return tokenizer;
}

BPETokenizer* bpe_tokenizer_from_dir(const char* model_dir) {
    if (!model_dir) return NULL;

    char path[1024];
    snprintf(path, sizeof(path), "%s/tokenizer.json", model_dir);

    BPETokenizer* tokenizer = bpe_tokenizer_from_file(path);
    if (!tokenizer) return NULL;

    /* 尝试从 tokenizer_config.json 加载特殊 token */
    snprintf(path, sizeof(path), "%s/tokenizer_config.json", model_dir);
    FILE* f = fopen(path, "r");
    if (f) {
        fseek(f, 0, SEEK_END);
        long size = ftell(f);
        fseek(f, 0, SEEK_SET);

        char* config = (char*)malloc(size + 1);
        if (config) {
            if (fread(config, 1, size, f) == (size_t)size) {
                config[size] = '\0';
                parse_special_tokens(config, tokenizer);
            }
            free(config);
        }
        fclose(f);
    }

    return tokenizer;
}

void bpe_tokenizer_free(BPETokenizer* tokenizer) {
    if (!tokenizer) return;

    if (tokenizer->vocab) {
        for (int i = 0; i < tokenizer->vocab_size; i++) {
            free(tokenizer->vocab[i]);
        }
        free(tokenizer->vocab);
    }

    if (tokenizer->token_to_id) {
        hash_table_free(tokenizer->token_to_id, tokenizer->hash_size);
    }

    if (tokenizer->merge_tokens) {
        for (int i = 0; i < tokenizer->num_merges; i++) {
            free(tokenizer->merge_tokens[i]);
        }
        free(tokenizer->merge_tokens);
    }
    free(tokenizer->merge_ranks);

    free(tokenizer->bos_token);
    free(tokenizer->eos_token);
    free(tokenizer->pad_token);
    free(tokenizer->unk_token);
    free(tokenizer->special_ids);

    free(tokenizer);
}

/* ============================================================================
 * 词表操作
 * ============================================================================ */

size_t bpe_tokenizer_vocab_size(const BPETokenizer* tokenizer) {
    return tokenizer ? tokenizer->vocab_size : 0;
}

int32_t bpe_tokenizer_token_to_id(const BPETokenizer* tokenizer, const char* token) {
    if (!tokenizer || !token) return tokenizer ? tokenizer->unk_id : 0;
    return hash_table_lookup(tokenizer->token_to_id, tokenizer->hash_size, token, tokenizer->unk_id);
}

const char* bpe_tokenizer_id_to_token(const BPETokenizer* tokenizer, int32_t id) {
    if (!tokenizer || id < 0 || id >= tokenizer->vocab_size) return NULL;
    return tokenizer->vocab[id];
}

/* ============================================================================
 * 特殊 Token
 * ============================================================================ */

int32_t bpe_tokenizer_bos_id(const BPETokenizer* tokenizer) {
    return tokenizer ? tokenizer->bos_id : -1;
}

int32_t bpe_tokenizer_eos_id(const BPETokenizer* tokenizer) {
    return tokenizer ? tokenizer->eos_id : -1;
}

int32_t bpe_tokenizer_pad_id(const BPETokenizer* tokenizer) {
    return tokenizer ? tokenizer->pad_id : 0;
}

int32_t bpe_tokenizer_unk_id(const BPETokenizer* tokenizer) {
    return tokenizer ? tokenizer->unk_id : 0;
}

const char* bpe_tokenizer_bos_token(const BPETokenizer* tokenizer) {
    return tokenizer ? tokenizer->bos_token : NULL;
}

const char* bpe_tokenizer_eos_token(const BPETokenizer* tokenizer) {
    return tokenizer ? tokenizer->eos_token : NULL;
}

bool bpe_tokenizer_is_special_token(const BPETokenizer* tokenizer, int32_t id) {
    if (!tokenizer) return false;

    for (int i = 0; i < tokenizer->num_special_ids; i++) {
        if (tokenizer->special_ids[i] == id) return true;
    }
    return false;
}

/* ============================================================================
 * Byte-level BPE 辅助函数
 * ============================================================================ */

/**
 * @brief GPT-2 风格的 byte-to-unicode 映射
 *
 * 将字节值转换为 Unicode 码点:
 * - 可打印 ASCII (33-126) 保持不变
 * - Latin-1 字符 (161-172, 174-255) 保持不变
 * - 其他字节 (0-32, 127-160, 173) 映射到 256+
 */
static int byte_to_unicode_cp(int byte) {
    /* GPT-2 byte-to-unicode 映射:
     * 1. Bytes 33-126 (94 bytes) -> themselves
     * 2. Bytes 161-172 (12 bytes) -> themselves
     * 3. Bytes 174-255 (82 bytes) -> themselves
     * 4. Other 68 bytes -> 256+
     */
    byte &= 0xFF;

    /* Range 1: 33-126 -> 33-126 */
    if (byte >= 33 && byte <= 126) {
        return byte;
    }
    /* Range 2: 161-172 -> 161-172 */
    if (byte >= 161 && byte <= 172) {
        return byte;
    }
    /* Range 3: 174-255 -> 174-255 */
    if (byte >= 174 && byte <= 255) {
        return byte;
    }

    /* Range 4: other bytes map to 256+
     * Order: 0-32, 127-160, 173
     * - 0-32 (33 bytes) -> 256-288
     * - 127-160 (34 bytes) -> 289-322
     * - 173 (1 byte) -> 323
     */
    if (byte <= 32) {
        return 256 + byte;  /* 0-32 -> 256-288 */
    }
    if (byte >= 127 && byte <= 160) {
        return 256 + 33 + (byte - 127);  /* 127-160 -> 289-322 */
    }
    if (byte == 173) {
        return 323;  /* 173 -> 323 */
    }

    return byte;
}

/**
 * @brief 将 Unicode 码点编码为 UTF-8
 * @return 写入的字节数
 */
static int encode_utf8(int cp, char* buf) {
    if (cp < 0x80) {
        buf[0] = (char)cp;
        return 1;
    } else if (cp < 0x800) {
        buf[0] = (char)(0xC0 | (cp >> 6));
        buf[1] = (char)(0x80 | (cp & 0x3F));
        return 2;
    } else if (cp < 0x10000) {
        buf[0] = (char)(0xE0 | (cp >> 12));
        buf[1] = (char)(0x80 | ((cp >> 6) & 0x3F));
        buf[2] = (char)(0x80 | (cp & 0x3F));
        return 3;
    } else {
        buf[0] = (char)(0xF0 | (cp >> 18));
        buf[1] = (char)(0x80 | ((cp >> 12) & 0x3F));
        buf[2] = (char)(0x80 | ((cp >> 6) & 0x3F));
        buf[3] = (char)(0x80 | (cp & 0x3F));
        return 4;
    }
}

/**
 * @brief 从 UTF-8 解码一个 Unicode 码点
 * @param str 输入字符串
 * @param cp 输出码点
 * @return 消耗的字节数
 */
static int decode_utf8(const char* str, int* cp) {
    unsigned char c = (unsigned char)str[0];
    if ((c & 0x80) == 0) {
        *cp = c;
        return 1;
    } else if ((c & 0xE0) == 0xC0) {
        *cp = ((c & 0x1F) << 6) | (str[1] & 0x3F);
        return 2;
    } else if ((c & 0xF0) == 0xE0) {
        *cp = ((c & 0x0F) << 12) | ((str[1] & 0x3F) << 6) | (str[2] & 0x3F);
        return 3;
    } else if ((c & 0xF8) == 0xF0) {
        *cp = ((c & 0x07) << 18) | ((str[1] & 0x3F) << 12) | ((str[2] & 0x3F) << 6) | (str[3] & 0x3F);
        return 4;
    }
    *cp = c;
    return 1;
}

/**
 * @brief Unicode 码点转回字节值 (byte_to_unicode_cp 的逆函数)
 */
static int unicode_cp_to_byte(int cp) {
    /* 直接映射的码点范围 */
    if (cp >= 33 && cp <= 126) return cp;       /* ASCII printable */
    if (cp >= 161 && cp <= 172) return cp;      /* Latin-1 supplement */
    if (cp >= 174 && cp <= 255) return cp;      /* Latin-1 supplement */

    /* 映射到 256+ 的码点，需要逆向计算 */
    if (cp >= 256 && cp <= 288) return cp - 256;           /* bytes 0-32 */
    if (cp >= 289 && cp <= 322) return 127 + (cp - 289);   /* bytes 127-160 */
    if (cp == 323) return 173;                             /* byte 173 */

    return cp;
}

/**
 * @brief 将字节序列转换为 token 字符串表示
 */
static void bytes_to_token_str(const unsigned char* bytes, size_t len, char* out, size_t max_out) {
    size_t pos = 0;
    for (size_t i = 0; i < len && pos < max_out - 4; i++) {
        int cp = byte_to_unicode_cp(bytes[i]);
        pos += encode_utf8(cp, out + pos);
    }
    out[pos] = '\0';
}

/* ============================================================================
 * 编码实现
 * ============================================================================ */

int bpe_tokenizer_encode(
    const BPETokenizer* tokenizer,
    const char* text,
    bool add_special_tokens,
    int32_t* ids,
    size_t max_ids
) {
    if (!tokenizer || !text || !ids || max_ids == 0) return -1;

    size_t count = 0;

    /* 添加 BOS */
    if (add_special_tokens && tokenizer->bos_id >= 0) {
        if (count < max_ids) ids[count++] = tokenizer->bos_id;
    }

    /* Byte-level BPE 编码 */
    size_t text_len = strlen(text);
    if (text_len == 0) {
        if (add_special_tokens && tokenizer->eos_id >= 0) {
            if (count < max_ids) ids[count++] = tokenizer->eos_id;
        }
        return (int)count;
    }

    /* 步骤 1: 将文本转换为字节级别的 token 字符串数组 */
    #define MAX_TOKENS 8192
    char** tokens = (char**)malloc(MAX_TOKENS * sizeof(char*));
    int num_tokens = 0;

    for (size_t i = 0; i < text_len && num_tokens < MAX_TOKENS; i++) {
        unsigned char byte = (unsigned char)text[i];
        char* token_str = (char*)malloc(8);
        int cp = byte_to_unicode_cp(byte);
        encode_utf8(cp, token_str);
        token_str[encode_utf8(cp, token_str)] = '\0';
        tokens[num_tokens++] = token_str;
    }

    /* 步骤 2: 应用 BPE 合并 */
    if (tokenizer->num_merges > 0 && num_tokens > 1) {
        /* 构建 merge 查找表 */
        for (int m = 0; m < tokenizer->num_merges && num_tokens > 1; m++) {
            const char* merge = tokenizer->merge_tokens[m];
            if (!merge) continue;

            /* 解析合并规则: "token1 token2" */
            char first[256] = {0};
            char second[256] = {0};
            const char* space = strchr(merge, ' ');
            if (!space) continue;

            size_t first_len = space - merge;
            if (first_len >= sizeof(first)) continue;
            strncpy(first, merge, first_len);
            first[first_len] = '\0';

            const char* second_start = space + 1;
            strncpy(second, second_start, sizeof(second) - 1);

            /* 查找并应用合并 */
            for (int i = 0; i < num_tokens - 1; i++) {
                if (tokens[i] && tokens[i + 1] &&
                    strcmp(tokens[i], first) == 0 &&
                    strcmp(tokens[i + 1], second) == 0) {

                    /* 合并两个 token */
                    size_t new_len = strlen(tokens[i]) + strlen(tokens[i + 1]) + 1;
                    char* merged = (char*)malloc(new_len);
                    snprintf(merged, new_len, "%s%s", tokens[i], tokens[i + 1]);

                    free(tokens[i]);
                    free(tokens[i + 1]);
                    tokens[i] = merged;

                    /* 移动后续 token */
                    for (int j = i + 1; j < num_tokens - 1; j++) {
                        tokens[j] = tokens[j + 1];
                    }
                    tokens[num_tokens - 1] = NULL;
                    num_tokens--;

                    /* 继续从当前位置查找 */
                    i--;
                }
            }
        }
    }

    /* 步骤 3: 将 token 字符串转换为 ID */
    for (int i = 0; i < num_tokens && count < max_ids; i++) {
        if (!tokens[i]) continue;

        int32_t id = hash_table_lookup(tokenizer->token_to_id, tokenizer->hash_size, tokens[i], -1);
        if (id >= 0) {
            ids[count++] = id;
        } else {
            /* 如果整个 token 找不到，尝试逐字节查找 */
            for (char* p = tokens[i]; *p && count < max_ids; ) {
                char buf[8];
                int char_len = utf8_char_len(p);
                if (char_len <= 0 || char_len > 4) char_len = 1;
                memcpy(buf, p, char_len);
                buf[char_len] = '\0';

                id = hash_table_lookup(tokenizer->token_to_id, tokenizer->hash_size, buf, -1);
                if (id >= 0) {
                    ids[count++] = id;
                } else {
                    ids[count++] = tokenizer->unk_id;
                }
                p += char_len;
            }
        }
    }

    /* 释放临时 token 数组 */
    for (int i = 0; i < num_tokens; i++) {
        free(tokens[i]);
    }
    free(tokens);
    #undef MAX_TOKENS

    /* 添加 EOS */
    if (add_special_tokens && tokenizer->eos_id >= 0) {
        if (count < max_ids) ids[count++] = tokenizer->eos_id;
    }

    return (int)count;
}

/* ============================================================================
 * 解码实现
 * ============================================================================ */

int bpe_tokenizer_decode(
    const BPETokenizer* tokenizer,
    const int32_t* ids,
    size_t num_ids,
    bool skip_special_tokens,
    char* out,
    size_t max_out
) {
    if (!tokenizer || !ids || !out || max_out == 0) return -1;

    size_t out_pos = 0;

    for (size_t i = 0; i < num_ids && out_pos < max_out - 1; i++) {
        int32_t id = ids[i];

        if (skip_special_tokens && bpe_tokenizer_is_special_token(tokenizer, id)) {
            continue;
        }

        const char* token = bpe_tokenizer_id_to_token(tokenizer, id);
        if (!token) continue;

        /* 将 byte-level unicode 转换回实际字节 */
        const char* p = token;
        while (*p && out_pos < max_out - 1) {
            int cp;
            int char_len = decode_utf8(p, &cp);

            /* 将 unicode 码点转换回原始字节 */
            int byte = unicode_cp_to_byte(cp);
            out[out_pos++] = (char)byte;

            p += char_len;
        }
    }

    out[out_pos] = '\0';
    return (int)out_pos;
}
