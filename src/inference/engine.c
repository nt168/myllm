/**
 * @file engine.c
 * @brief 推理引擎实现
 *
 * 整合所有模块，提供完整的文本生成功能
 */

#include "inference/engine.h"
#include "models/llama/llama.h"
#include "tokenizer/tokenizer.h"
#include "sampler/sampler.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

/* ============================================================================
 * 内部结构定义
 * ============================================================================ */

/**
 * @brief 推理引擎内部结构
 */
struct InferenceEngine {
    /* 组件 */
    WeightLoader* loader;           /**< 权重加载器 */
    BPETokenizer* tokenizer;        /**< 分词器 */
    Sampler* sampler;               /**< 采样器 */
    ModelConfig config;             /**< 模型配置 */

    /* 模型实例 (根据架构类型选择) */
    LlamaModel* model;              /**< 模型指针 (目前只支持 LLaMA 架构) */
    int model_type;                 /**< 模型类型: 0=LLaMA, 1=Qwen2, 2=Qwen3 */

    /* 状态 */
    size_t current_position;        /**< 当前生成位置 */
    bool owns_loader;               /**< 是否拥有加载器 */
    bool owns_tokenizer;            /**< 是否拥有分词器 */
    bool owns_model;                /**< 是否拥有模型 */
};

/* ============================================================================
 * 时间工具
 * ============================================================================ */

/**
 * @brief 获取当前时间 (毫秒)
 */
static double get_time_ms(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return ts.tv_sec * 1000.0 + ts.tv_nsec / 1000000.0;
}

/* ============================================================================
 * 模型权重加载辅助函数
 * ============================================================================ */

/**
 * @brief 将张量数据复制到线性层权重
 */
static int load_linear_weight(Linear* layer, WeightLoader* loader, const char* prefix) {
    char name[256];

    /* 加载 weight */
    snprintf(name, sizeof(name), "%s.weight", prefix);
    Tensor* weight = weight_loader_load_f32(loader, name);
    if (!weight) {
        fprintf(stderr, "Warning: Failed to load %s\n", name);
        return -1;
    }

    /* 复制权重数据 */
    if (layer->weight && layer->weight->data && weight->data) {
        size_t weight_size = layer->out_features * layer->in_features * sizeof(float);
        memcpy(layer->weight->data, weight->data, weight_size);
    }

    tensor_free(weight);

    /* 加载 bias (如果存在) */
    if (layer->use_bias) {
        snprintf(name, sizeof(name), "%s.bias", prefix);
        Tensor* bias = weight_loader_load_f32(loader, name);
        if (bias && layer->bias && layer->bias->data && bias->data) {
            memcpy(layer->bias->data, bias->data, layer->out_features * sizeof(float));
            tensor_free(bias);
        }
    }

    return 0;
}

/**
 * @brief 将张量数据复制到嵌入层权重
 */
static int load_embedding_weight(Embedding* embed, WeightLoader* loader, const char* prefix) {
    char name[256];
    snprintf(name, sizeof(name), "%s.weight", prefix);
    Tensor* weight = weight_loader_load_f32(loader, name);
    if (!weight) {
        fprintf(stderr, "Warning: Failed to load %s\n", name);
        return -1;
    }

    if (embed->weight && embed->weight->data && weight->data) {
        size_t weight_size = embed->num_embeddings * embed->embedding_dim * sizeof(float);
        memcpy(embed->weight->data, weight->data, weight_size);
    }

    tensor_free(weight);
    return 0;
}

/**
 * @brief 将张量数据复制到 RMSNorm 权重
 */
static int load_rmsnorm_weight(RMSNorm* norm, WeightLoader* loader, const char* prefix) {
    char name[256];
    snprintf(name, sizeof(name), "%s.weight", prefix);
    Tensor* weight = weight_loader_load_f32(loader, name);
    if (!weight) {
        fprintf(stderr, "Warning: Failed to load %s\n", name);
        return -1;
    }

    if (norm->weight && norm->weight->data && weight->data) {
        size_t weight_size = norm->hidden_dim * sizeof(float);
        memcpy(norm->weight->data, weight->data, weight_size);
    }

    tensor_free(weight);
    return 0;
}

/**
 * @brief 加载 LLaMA 模型权重
 */
static int load_llama_weights(LlamaModel* model, WeightLoader* loader) {
    char name[256];
    int ret = 0;

    /* 1. 加载 token 嵌入层 */
    ret = load_embedding_weight(model->embed_tokens, loader, "model.embed_tokens");
    if (ret != 0) {
        /* 尝试其他命名格式 */
        ret = load_embedding_weight(model->embed_tokens, loader, "embed_tokens");
    }

    /* 2. 加载各 Transformer 层 */
    for (size_t i = 0; i < model->num_layers; i++) {
        LlamaTransformerBlock* layer = model->layers[i];
        snprintf(name, sizeof(name), "model.layers.%zu", i);

        /* 输入归一化 */
        char norm_name[300];
        snprintf(norm_name, sizeof(norm_name), "%s.input_layernorm", name);
        load_rmsnorm_weight(layer->input_norm, loader, norm_name);

        /* 注意力层 */
        char proj_name[300];
        snprintf(proj_name, sizeof(proj_name), "%s.self_attn.q_proj", name);
        load_linear_weight(llama_attention_q_proj(layer->attention), loader, proj_name);
        snprintf(proj_name, sizeof(proj_name), "%s.self_attn.k_proj", name);
        load_linear_weight(llama_attention_k_proj(layer->attention), loader, proj_name);
        snprintf(proj_name, sizeof(proj_name), "%s.self_attn.v_proj", name);
        load_linear_weight(llama_attention_v_proj(layer->attention), loader, proj_name);
        snprintf(proj_name, sizeof(proj_name), "%s.self_attn.o_proj", name);
        load_linear_weight(llama_attention_o_proj(layer->attention), loader, proj_name);

        /* 后注意力归一化 */
        snprintf(norm_name, sizeof(norm_name), "%s.post_attention_layernorm", name);
        load_rmsnorm_weight(layer->post_attention_norm, loader, norm_name);

        /* MLP 层 */
        snprintf(norm_name, sizeof(norm_name), "%s.mlp.gate_proj", name);
        load_linear_weight(layer->mlp->gate_proj, loader, norm_name);
        snprintf(norm_name, sizeof(norm_name), "%s.mlp.up_proj", name);
        load_linear_weight(layer->mlp->up_proj, loader, norm_name);
        snprintf(norm_name, sizeof(norm_name), "%s.mlp.down_proj", name);
        load_linear_weight(layer->mlp->down_proj, loader, norm_name);
    }

    /* 3. 加载最终归一化层 */
    load_rmsnorm_weight(model->norm, loader, "model.norm");

    /* 4. 加载 lm_head (如果存在且不共享权重) */
    if (model->lm_head) {
        load_linear_weight(model->lm_head, loader, "lm_head");
    }

    return 0;
}

/* ============================================================================
 * 生成配置
 * ============================================================================ */

GenerateConfig generate_config_default(void) {
    GenerateConfig config = {
        .max_tokens = ENGINE_DEFAULT_MAX_TOKENS,
        .temperature = SAMPLER_DEFAULT_TEMPERATURE,
        .top_k = SAMPLER_DEFAULT_TOP_K,
        .top_p = SAMPLER_DEFAULT_TOP_P,
        .repetition_penalty = SAMPLER_DEFAULT_REPETITION_PENALTY,
        .stop_tokens = NULL,
        .num_stop_tokens = 0,
        .add_bos = true,
        .add_eos = false,
        .stream = false,
        .seed = (uint64_t)time(NULL)
    };
    return config;
}

/* ============================================================================
 * 生成结果
 * ============================================================================ */

void generate_result_free(GenerateResult* result) {
    if (!result) return;

    if (result->tokens) {
        free(result->tokens);
        result->tokens = NULL;
    }
    if (result->text) {
        free(result->text);
        result->text = NULL;
    }
    result->num_tokens = 0;
    result->text_len = 0;
}

/* ============================================================================
 * 生命周期管理
 * ============================================================================ */

InferenceEngine* engine_new(const char* model_dir) {
    if (!model_dir) return NULL;

    InferenceEngine* engine = (InferenceEngine*)calloc(1, sizeof(InferenceEngine));
    if (!engine) return NULL;

    /* 加载权重 */
    engine->loader = weight_loader_new(model_dir);
    if (!engine->loader) {
        free(engine);
        return NULL;
    }
    engine->owns_loader = true;

    /* 复制配置 */
    const ModelConfig* cfg = weight_loader_config(engine->loader);
    if (cfg) {
        memcpy(&engine->config, cfg, sizeof(ModelConfig));
    }

    /* 加载分词器 */
    engine->tokenizer = bpe_tokenizer_from_dir(model_dir);
    if (!engine->tokenizer) {
        /* 尝试 tokenizer.json */
        char tokenizer_path[1024];
        snprintf(tokenizer_path, sizeof(tokenizer_path), "%s/tokenizer.json", model_dir);
        engine->tokenizer = bpe_tokenizer_from_file(tokenizer_path);
    }
    engine->owns_tokenizer = (engine->tokenizer != NULL);

    /* 创建采样器 */
    engine->sampler = sampler_new();
    if (!engine->sampler) {
        if (engine->owns_loader) weight_loader_free(engine->loader);
        if (engine->owns_tokenizer) bpe_tokenizer_free(engine->tokenizer);
        free(engine);
        return NULL;
    }

    /* 根据架构创建模型 */
    if (strstr(engine->config.architecture, "Llama") ||
        strstr(engine->config.architecture, "LLaMA") ||
        strstr(engine->config.model_type, "llama")) {
        engine->model_type = 0;
    }
    else if (strstr(engine->config.architecture, "Qwen2") ||
             strstr(engine->config.model_type, "qwen2")) {
        engine->model_type = 1;
    }
    else if (strstr(engine->config.architecture, "Qwen3") ||
             strstr(engine->config.model_type, "qwen3")) {
        engine->model_type = 2;
    }
    else {
        /* 默认使用 LLaMA 架构 */
        engine->model_type = 0;
    }

    /* 创建 LlamaModel 并加载权重 */
    LlamaConfig llama_config;
    llama_config.hidden_size = engine->config.hidden_size;
    llama_config.intermediate_size = engine->config.intermediate_size;
    llama_config.num_attention_heads = engine->config.num_attention_heads;
    llama_config.num_hidden_layers = engine->config.num_hidden_layers;
    llama_config.vocab_size = engine->config.vocab_size;
    llama_config.num_key_value_heads = engine->config.num_key_value_heads;
    llama_config.head_dim = engine->config.head_dim;
    llama_config.rope_theta = engine->config.rope_theta;
    llama_config.max_position_embeddings = engine->config.max_position_embeddings;
    llama_config.rms_norm_eps = engine->config.rms_norm_eps;
    llama_config.torch_dtype = engine->config.torch_dtype;
    llama_config.tie_word_embeddings = engine->config.tie_word_embeddings;

    /* 创建模型 (带 KV 缓存，限制缓存大小避免 OOM) */
    size_t cache_max_len = llama_config.max_position_embeddings;
    if (cache_max_len > ENGINE_MAX_CACHE_LEN) {
        cache_max_len = ENGINE_MAX_CACHE_LEN;
    }
    LlamaConfig cache_config = llama_config;
    cache_config.max_position_embeddings = cache_max_len;

    engine->model = llama_model_new_with_cache(&cache_config, 1);
    if (!engine->model) {
        fprintf(stderr, "Warning: Failed to create LlamaModel\n");
        /* 即使模型创建失败，也返回引擎，允许其他功能使用 */
    } else {
        engine->owns_model = true;

        /* 加载权重到模型 */
        if (load_llama_weights(engine->model, engine->loader) != 0) {
            fprintf(stderr, "Warning: Failed to load some weights\n");
        }
    }

    engine->current_position = 0;

    return engine;
}

InferenceEngine* engine_new_with_components(
    WeightLoader* loader,
    BPETokenizer* tokenizer,
    const ModelConfig* config
) {
    if (!loader) return NULL;

    InferenceEngine* engine = (InferenceEngine*)calloc(1, sizeof(InferenceEngine));
    if (!engine) return NULL;

    engine->loader = loader;
    engine->tokenizer = tokenizer;
    engine->owns_loader = false;
    engine->owns_tokenizer = false;

    if (config) {
        memcpy(&engine->config, config, sizeof(ModelConfig));
    }

    engine->sampler = sampler_new();
    if (!engine->sampler) {
        free(engine);
        return NULL;
    }

    engine->model = NULL;
    engine->model_type = 0;
    engine->current_position = 0;

    return engine;
}

void engine_free(InferenceEngine* engine) {
    if (!engine) return;

    if (engine->sampler) {
        sampler_free(engine->sampler);
    }

    if (engine->owns_model && engine->model) {
        llama_model_free(engine->model);
    }

    if (engine->owns_loader && engine->loader) {
        weight_loader_free(engine->loader);
    }

    if (engine->owns_tokenizer && engine->tokenizer) {
        bpe_tokenizer_free(engine->tokenizer);
    }

    free(engine);
}

/* ============================================================================
 * 模型信息
 * ============================================================================ */

const ModelConfig* engine_get_config(const InferenceEngine* engine) {
    return engine ? &engine->config : NULL;
}

size_t engine_vocab_size(const InferenceEngine* engine) {
    return engine ? engine->config.vocab_size : 0;
}

size_t engine_num_layers(const InferenceEngine* engine) {
    return engine ? engine->config.num_hidden_layers : 0;
}

size_t engine_hidden_size(const InferenceEngine* engine) {
    return engine ? engine->config.hidden_size : 0;
}

/* ============================================================================
 * 分词器接口
 * ============================================================================ */

int engine_encode(
    const InferenceEngine* engine,
    const char* text,
    bool add_special,
    int32_t* ids,
    size_t max_ids
) {
    if (!engine || !engine->tokenizer || !text || !ids) {
        return -1;
    }
    return bpe_tokenizer_encode(engine->tokenizer, text, add_special, ids, max_ids);
}

int engine_decode(
    const InferenceEngine* engine,
    const int32_t* ids,
    size_t num_ids,
    bool skip_special,
    char* out,
    size_t max_out
) {
    if (!engine || !engine->tokenizer || !ids || !out) {
        return -1;
    }
    return bpe_tokenizer_decode(engine->tokenizer, ids, num_ids, skip_special, out, max_out);
}

int32_t engine_bos_id(const InferenceEngine* engine) {
    return engine && engine->tokenizer ? bpe_tokenizer_bos_id(engine->tokenizer) : -1;
}

int32_t engine_eos_id(const InferenceEngine* engine) {
    return engine && engine->tokenizer ? bpe_tokenizer_eos_id(engine->tokenizer) : -1;
}

int32_t engine_pad_id(const InferenceEngine* engine) {
    return engine && engine->tokenizer ? bpe_tokenizer_pad_id(engine->tokenizer) : -1;
}

/* ============================================================================
 * 推理接口
 * ============================================================================ */

Tensor* engine_prefill(
    InferenceEngine* engine,
    const int32_t* tokens,
    size_t num_tokens
) {
    if (!engine || !tokens || num_tokens == 0) {
        return NULL;
    }

    /* 如果没有模型实例，返回 NULL */
    if (!engine->model) {
        engine->current_position = num_tokens;
        return NULL;
    }

    /* 调用 LlamaModel 的 prefill */
    Tensor* logits = llama_model_prefill(engine->model, tokens, num_tokens);
    engine->current_position = num_tokens;

    return logits;
}

Tensor* engine_decode_step(
    InferenceEngine* engine,
    int32_t token
) {
    if (!engine) {
        return NULL;
    }

    /* 如果没有模型实例，返回 NULL */
    if (!engine->model) {
        engine->current_position++;
        return NULL;
    }

    /* 调用 LlamaModel 的 decode_step */
    Tensor* logits = llama_model_decode_step(engine->model, token, engine->current_position);
    engine->current_position++;

    return logits;
}

void engine_reset_cache(InferenceEngine* engine) {
    if (!engine) return;
    engine->current_position = 0;
    /* 如果有模型，也需要重置模型的 KV 缓存 */
}

size_t engine_cache_len(const InferenceEngine* engine) {
    return engine ? engine->current_position : 0;
}

/* ============================================================================
 * 生成接口
 * ============================================================================ */

/**
 * @brief 检查是否为停止 token
 */
static bool is_stop_token(int32_t token, const int32_t* stop_tokens, size_t num_stop_tokens) {
    for (size_t i = 0; i < num_stop_tokens; i++) {
        if (token == stop_tokens[i]) {
            return true;
        }
    }
    return false;
}

GenerateResult engine_generate(
    InferenceEngine* engine,
    const char* prompt,
    const GenerateConfig* config
) {
    GenerateResult result = {0};

    if (!engine || !prompt) {
        return result;
    }

    /* 编码输入 */
    int32_t prompt_tokens[4096];
    int num_prompt = engine_encode(engine, prompt, config ? config->add_bos : true,
                                    prompt_tokens, 4096);
    if (num_prompt <= 0) {
        return result;
    }

    /* 使用 token 生成 */
    GenerateConfig token_config = config ? *config : generate_config_default();

    return engine_generate_from_tokens(engine, prompt_tokens, (size_t)num_prompt, &token_config);
}

GenerateResult engine_generate_from_tokens(
    InferenceEngine* engine,
    const int32_t* prompt_tokens,
    size_t num_prompt_tokens,
    const GenerateConfig* config
) {
    GenerateResult result = {0};

    if (!engine || !prompt_tokens || num_prompt_tokens == 0) {
        return result;
    }

    GenerateConfig cfg = config ? *config : generate_config_default();

    /* 设置采样器参数 */
    sampler_set_temperature(engine->sampler, cfg.temperature);
    sampler_set_top_k(engine->sampler, cfg.top_k);
    sampler_set_top_p(engine->sampler, cfg.top_p);
    sampler_set_repetition_penalty(engine->sampler, cfg.repetition_penalty);
    sampler_set_seed(engine->sampler, cfg.seed);

    /* 重置缓存 */
    engine_reset_cache(engine);

    /* 如果没有模型，返回空结果 */
    if (!engine->model) {
        return result;
    }

    /* 分配输出 token 数组 */
    size_t max_output = cfg.max_tokens + num_prompt_tokens + 1;
    int32_t* all_tokens = (int32_t*)malloc(max_output * sizeof(int32_t));
    if (!all_tokens) return result;

    memcpy(all_tokens, prompt_tokens, num_prompt_tokens * sizeof(int32_t));
    size_t num_all_tokens = num_prompt_tokens;

    /* 时间统计 */
    double prefill_start = get_time_ms();
    double time_to_first_token = 0.0;
    double total_decode_time = 0.0;

    /* === Prefill 阶段 === */
    Tensor* logits = engine_prefill(engine, prompt_tokens, num_prompt_tokens);
    if (!logits) {
        free(all_tokens);
        return result;
    }

    /* 采样第一个 token (简化版，使用 argmax) */
    /* 注意：实际实现需要从 logits 中采样 */
    int32_t next_token = 0;  /* 占位符 */

    time_to_first_token = get_time_ms() - prefill_start;

    /* 检查 EOS */
    if (is_stop_token(next_token, cfg.stop_tokens, cfg.num_stop_tokens)) {
        result.tokens = all_tokens;
        result.num_tokens = num_all_tokens;
        result.finished = true;
        result.stop_token = next_token;
        result.prefill_time_ms = time_to_first_token;
        return result;
    }

    all_tokens[num_all_tokens++] = next_token;

    /* === Decode 循环 === */
    size_t num_generated = 1;

    while (num_generated < (size_t)cfg.max_tokens) {
        double decode_start = get_time_ms();

        logits = engine_decode_step(engine, next_token);
        if (!logits) break;

        /* 采样下一个 token (简化版) */
        next_token = 0;  /* 占位符 */

        double decode_time = get_time_ms() - decode_start;
        total_decode_time += decode_time;

        /* 检查停止条件 */
        if (is_stop_token(next_token, cfg.stop_tokens, cfg.num_stop_tokens)) {
            result.stop_token = next_token;
            break;
        }

        all_tokens[num_all_tokens++] = next_token;
        num_generated++;
    }

    result.finished = true;
    if (num_generated >= (size_t)cfg.max_tokens) {
        result.stop_token = -1;  /* 达到最大 token 数 */
    }

    /* 计算统计信息 */
    result.prefill_time_ms = time_to_first_token;
    result.decode_time_ms = num_generated > 0 ? total_decode_time / num_generated : 0.0;

    /* 只返回生成的 token (不含 prompt) */
    size_t num_generated_tokens = num_all_tokens - num_prompt_tokens;
    result.tokens = (int32_t*)malloc(num_generated_tokens * sizeof(int32_t));
    if (result.tokens) {
        memcpy(result.tokens, all_tokens + num_prompt_tokens, num_generated_tokens * sizeof(int32_t));
        result.num_tokens = num_generated_tokens;
    }
    free(all_tokens);

    /* 解码为文本 */
    if (engine->tokenizer && result.tokens && result.num_tokens > 0) {
        size_t text_buf_size = result.num_tokens * 8 + 1;  /* 每个 token 最多 8 字节 */
        result.text = (char*)malloc(text_buf_size);
        if (result.text) {
            int decode_len = engine_decode(engine, result.tokens, result.num_tokens,
                                           true, result.text, text_buf_size);
            result.text_len = decode_len > 0 ? (size_t)decode_len : 0;
        }
    }

    return result;
}

size_t engine_generate_stream(
    InferenceEngine* engine,
    const char* prompt,
    const GenerateConfig* config,
    GenerateCallback callback,
    void* user_data
) {
    if (!engine || !prompt || !callback) {
        return 0;
    }

    GenerateConfig cfg = config ? *config : generate_config_default();

    /* 编码输入 */
    int32_t prompt_tokens[4096];
    int num_prompt = engine_encode(engine, prompt, cfg.add_bos, prompt_tokens, 4096);
    if (num_prompt <= 0) {
        return 0;
    }

    /* 设置采样器 */
    sampler_set_temperature(engine->sampler, cfg.temperature);
    sampler_set_top_k(engine->sampler, cfg.top_k);
    sampler_set_top_p(engine->sampler, cfg.top_p);
    sampler_set_repetition_penalty(engine->sampler, cfg.repetition_penalty);
    sampler_set_seed(engine->sampler, cfg.seed);

    /* 重置 */
    engine_reset_cache(engine);

    /* 如果没有模型，返回 0 */
    if (!engine->model) {
        return 0;
    }

    /* Prefill */
    Tensor* logits_tensor = engine_prefill(engine, prompt_tokens, (size_t)num_prompt);
    if (!logits_tensor) return 0;

    /* 从 logits 采样第一个 token */
    float* logits = (float*)logits_tensor->data;
    size_t vocab_size = engine->config.vocab_size;
    int32_t next_token = sampler_sample(engine->sampler, logits, vocab_size);

    /* 释放 logits 张量 */
    tensor_free(logits_tensor);

    if (is_stop_token(next_token, cfg.stop_tokens, cfg.num_stop_tokens)) {
        return 0;
    }

    /* 输出第一个 token */
    char token_text[256] = {0};
    if (engine->tokenizer) {
        engine_decode(engine, &next_token, 1, true, token_text, sizeof(token_text));
    }
    if (!callback(next_token, token_text, user_data)) {
        return 1;
    }

    size_t num_generated = 1;

    /* Decode 循环 */
    while (num_generated < (size_t)cfg.max_tokens) {
        logits_tensor = engine_decode_step(engine, next_token);
        if (!logits_tensor) break;

        /* 从 logits 采样下一个 token */
        logits = (float*)logits_tensor->data;
        next_token = sampler_sample(engine->sampler, logits, vocab_size);

        /* 释放 logits 张量 */
        tensor_free(logits_tensor);

        if (is_stop_token(next_token, cfg.stop_tokens, cfg.num_stop_tokens)) {
            break;
        }

        /* 输出 token */
        token_text[0] = '\0';
        if (engine->tokenizer) {
            engine_decode(engine, &next_token, 1, true, token_text, sizeof(token_text));
        }
        if (!callback(next_token, token_text, user_data)) {
            num_generated++;
            break;
        }

        num_generated++;
    }

    return num_generated;
}

/* ============================================================================
 * 批处理接口
 * ============================================================================ */

Tensor* engine_prefill_batch(
    InferenceEngine* engine,
    const int32_t** batch_tokens,
    const size_t* batch_sizes,
    size_t batch_size
) {
    (void)batch_tokens;
    (void)batch_sizes;
    (void)batch_size;

    if (!engine) return NULL;

    /* 批处理暂未实现，返回 NULL */
    return NULL;
}
