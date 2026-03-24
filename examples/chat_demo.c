/**
 * @file chat_demo.c
 * @brief 完整的聊天推理演示程序
 *
 * 演示 myllm 推理引擎的完整流程:
 * 1. 加载模型权重
 * 2. 加载分词器
 * 3. 编码输入文本
 * 4. 执行推理 (Prefill + Decode)
 * 5. 采样生成 token
 * 6. 解码输出文本
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <myllm.h>
#include "inference/engine.h"
#include "loader/loader.h"
#include "tokenizer/tokenizer.h"
#include "sampler/sampler.h"
#include "models/llama/llama.h"

/* ============================================================================
 * 配置
 * ============================================================================ */

/** 默认模型路径 */
#define DEFAULT_MODEL_PATH "./models/llama-7b"

/** 最大生成长度 */
#define MAX_GEN_TOKENS 128

/** 最大输入长度 */
#define MAX_INPUT_LEN 4096

/* ============================================================================
 * 辅助函数
 * ============================================================================ */

/**
 * @brief 打印使用说明
 */
static void print_usage(const char* program) {
    printf("Usage: %s [options]\n\n", program);
    printf("Options:\n");
    printf("  -m <path>    Model directory path (default: %s)\n", DEFAULT_MODEL_PATH);
    printf("  -p <prompt>  Input prompt text\n");
    printf("  -t <temp>    Sampling temperature (default: 0.7)\n");
    printf("  -k <top_k>   Top-K sampling (default: 40)\n");
    printf("  -n <tokens>  Max tokens to generate (default: %d)\n", MAX_GEN_TOKENS);
    printf("  -h           Show this help message\n");
}

/**
 * @brief 生成回调函数
 */
static bool on_token_generated(int32_t token, const char* token_str, void* user_data) {
    (void)user_data;
    if (token_str) {
        printf("%s", token_str);
        fflush(stdout);
    }
    return true;  /* 继续生成 */
}

/* ============================================================================
 * 简化版推理演示
 * ============================================================================ */

/**
 * @brief 演示模型加载和基本信息
 */
static int demo_model_info(const char* model_path) {
    printf("\n=== Model Information Demo ===\n\n");

    /* 1. 加载权重加载器 */
    printf("[1] Loading model weights from: %s\n", model_path);
    WeightLoader* loader = weight_loader_new(model_path);
    if (!loader) {
        fprintf(stderr, "Error: Failed to load model from '%s'\n", model_path);
        fprintf(stderr, "Make sure the directory contains:\n");
        fprintf(stderr, "  - config.json\n");
        fprintf(stderr, "  - model.safetensors (or model-00001-of-00001.safetensors)\n");
        return -1;
    }

    /* 2. 获取模型配置 */
    const ModelConfig* config = weight_loader_config(loader);
    if (!config) {
        fprintf(stderr, "Error: Failed to get model config\n");
        weight_loader_free(loader);
        return -1;
    }

    /* 3. 打印模型信息 */
    printf("\n[2] Model Configuration:\n");
    printf("    Architecture:     %s\n", config->architecture);
    printf("    Model Type:       %s\n", config->model_type);
    printf("    Hidden Size:      %zu\n", config->hidden_size);
    printf("    Intermediate Size:%zu\n", config->intermediate_size);
    printf("    Num Layers:       %zu\n", config->num_hidden_layers);
    printf("    Num Heads:        %zu\n", config->num_attention_heads);
    printf("    Num KV Heads:     %zu\n", config->num_key_value_heads);
    printf("    Head Dimension:   %zu\n", config->head_dim);
    printf("    Vocab Size:       %zu\n", config->vocab_size);
    printf("    Max Position:     %zu\n", config->max_position_embeddings);
    printf("    RoPE Theta:       %.1f\n", config->rope_theta);
    printf("    RMS Norm Eps:     %.2e\n", config->rms_norm_eps);

    /* 4. 尝试加载分词器 */
    printf("\n[3] Loading Tokenizer...\n");
    BPETokenizer* tokenizer = bpe_tokenizer_from_dir(model_path);
    if (!tokenizer) {
        char tokenizer_path[1024];
        snprintf(tokenizer_path, sizeof(tokenizer_path), "%s/tokenizer.json", model_path);
        tokenizer = bpe_tokenizer_from_file(tokenizer_path);
    }

    if (tokenizer) {
        printf("    Tokenizer loaded successfully!\n");
        printf("    Vocab Size:       %zu\n", bpe_tokenizer_vocab_size(tokenizer));
        printf("    BOS Token ID:     %d\n", bpe_tokenizer_bos_id(tokenizer));
        printf("    EOS Token ID:     %d\n", bpe_tokenizer_eos_id(tokenizer));

        /* 演示编码和解码 */
        const char* test_text = "Hello, world!";
        int32_t tokens[256];
        int num_tokens = bpe_tokenizer_encode(tokenizer, test_text, true, tokens, 256);
        if (num_tokens > 0) {
            printf("\n    Encoding test: \"%s\"\n", test_text);
            printf("    Tokens: [");
            for (int i = 0; i < num_tokens && i < 10; i++) {
                printf("%d", tokens[i]);
                if (i < num_tokens - 1) printf(", ");
            }
            if (num_tokens > 10) printf(", ...");
            printf("] (count: %d)\n", num_tokens);

            /* 解码回来 */
            char decoded[512];
            int decoded_len = bpe_tokenizer_decode(tokenizer, tokens, num_tokens, false, decoded, sizeof(decoded));
            if (decoded_len > 0) {
                printf("    Decoded back: \"%s\"\n", decoded);
            }
        }

        bpe_tokenizer_free(tokenizer);
    } else {
        printf("    Warning: Could not load tokenizer\n");
    }

    /* 5. 检查模型权重 */
    printf("\n[4] Checking model weights...\n");

    /* 检查嵌入层权重 */
    if (weight_loader_has_tensor(loader, "model.embed_tokens.weight")) {
        Tensor* embed = weight_loader_load(loader, "model.embed_tokens.weight");
        if (embed) {
            printf("    embed_tokens: shape [");
            for (size_t i = 0; i < embed->shape.ndim; i++) {
                printf("%zu", embed->shape.dims[i]);
                if (i < embed->shape.ndim - 1) printf(", ");
            }
            printf("]\n");
            tensor_free(embed);
        }
    }

    /* 检查第一层权重 */
    if (weight_loader_has_tensor(loader, "model.layers.0.self_attn.q_proj.weight")) {
        Tensor* q_proj = weight_loader_load(loader, "model.layers.0.self_attn.q_proj.weight");
        if (q_proj) {
            printf("    layers.0.q_proj: shape [");
            for (size_t i = 0; i < q_proj->shape.ndim; i++) {
                printf("%zu", q_proj->shape.dims[i]);
                if (i < q_proj->shape.ndim - 1) printf(", ");
            }
            printf("]\n");
            tensor_free(q_proj);
        }
    }

    printf("\n[5] Model loaded successfully!\n");

    weight_loader_free(loader);
    return 0;
}

/**
 * @brief 演示创建 LLaMA 模型并加载权重
 */
static int demo_llama_inference(const char* model_path) {
    printf("\n=== LLaMA Inference Demo ===\n\n");

    /* 1. 加载权重 */
    printf("[1] Loading weights...\n");
    WeightLoader* loader = weight_loader_new(model_path);
    if (!loader) {
        fprintf(stderr, "Error: Failed to load model\n");
        return -1;
    }

    const ModelConfig* config = weight_loader_config(loader);

    /* 2. 创建 LLaMA 配置 */
    printf("[2] Creating LLaMA model...\n");
    LlamaConfig llama_config;
    llama_config.vocab_size = config->vocab_size;
    llama_config.hidden_size = config->hidden_size;
    llama_config.intermediate_size = config->intermediate_size;
    llama_config.num_hidden_layers = config->num_hidden_layers;
    llama_config.num_attention_heads = config->num_attention_heads;
    llama_config.num_key_value_heads = config->num_key_value_heads;
    llama_config.head_dim = config->head_dim;
    llama_config.max_position_embeddings = config->max_position_embeddings;
    llama_config.rope_theta = config->rope_theta;
    llama_config.rms_norm_eps = config->rms_norm_eps;
    llama_config.tie_word_embeddings = config->tie_word_embeddings;

    /* 3. 创建模型 (带 KV 缓存) */
    LlamaModel* model = llama_model_new_with_cache(&llama_config, 1);
    if (!model) {
        fprintf(stderr, "Error: Failed to create LLaMA model\n");
        weight_loader_free(loader);
        return -1;
    }

    printf("    Model created with %zu layers\n", llama_model_num_layers(model));
    printf("    KV cache enabled: %s\n", llama_model_has_cache(model) ? "yes" : "no");

    /* 4. 加载权重到模型 */
    printf("[3] Loading weights into model...\n");
    printf("    Note: Weight loading is a simplified demonstration.\n");
    printf("    In a full implementation, all layer weights would be loaded.\n");

    /* 5. 演示推理流程 (不实际执行，因为权重未完全加载) */
    printf("[4] Inference flow demonstration:\n");
    printf("    a) Token embedding: input_ids [batch, seq] -> hidden [batch, seq, hidden_dim]\n");
    printf("    b) For each transformer layer:\n");
    printf("       - Input RMSNorm\n");
    printf("       - GQA attention with RoPE and KV cache\n");
    printf("       - Residual connection\n");
    printf("       - Post-attention RMSNorm\n");
    printf("       - MLP (SwiGLU: gate * up -> down)\n");
    printf("       - Residual connection\n");
    printf("    c) Final RMSNorm\n");
    printf("    d) LM head projection -> logits [batch, seq, vocab_size]\n");
    printf("    e) Sampling from logits\n");

    /* 6. 清理 */
    printf("[5] Cleaning up...\n");
    llama_model_free(model);
    weight_loader_free(loader);

    printf("\nDemo completed!\n");
    return 0;
}

/**
 * @brief 使用推理引擎进行完整演示
 */
static int demo_engine_inference(const char* model_path, const char* prompt, float temperature, int top_k, int max_tokens) {
    printf("\n=== Engine Inference Demo ===\n\n");

    /* 1. 创建推理引擎 */
    printf("[1] Creating inference engine...\n");
    InferenceEngine* engine = engine_new(model_path);
    if (!engine) {
        fprintf(stderr, "Error: Failed to create inference engine\n");
        return -1;
    }

    /* 2. 获取配置 */
    const ModelConfig* config = engine_get_config(engine);
    if (config) {
        printf("    Model: %s (%s)\n", config->model_type, config->architecture);
        printf("    Hidden size: %zu\n", config->hidden_size);
        printf("    Layers: %zu\n", config->num_hidden_layers);
    }

    /* 3. 编码输入 */
    printf("\n[2] Encoding prompt: \"%s\"\n", prompt);
    int32_t tokens[4096];
    int num_tokens = engine_encode(engine, prompt, true, tokens, 4096);
    if (num_tokens <= 0) {
        fprintf(stderr, "Error: Failed to encode prompt\n");
        engine_free(engine);
        return -1;
    }
    printf("    Encoded to %d tokens\n", num_tokens);

    /* 4. 配置生成 */
    GenerateConfig gen_config = generate_config_default();
    gen_config.max_tokens = max_tokens;
    gen_config.temperature = temperature;
    gen_config.top_k = top_k;
    gen_config.top_p = 0.9f;
    gen_config.stream = true;

    printf("\n[3] Generating response (max %d tokens, temp=%.2f, top_k=%d)...\n",
           max_tokens, temperature, top_k);
    printf("\n--- Response ---\n");

    /* 5. 流式生成 */
    size_t generated = engine_generate_stream(
        engine,
        prompt,
        &gen_config,
        on_token_generated,
        NULL
    );

    printf("\n--- End (%zu tokens) ---\n", generated);

    /* 6. 清理 */
    engine_free(engine);

    printf("\nEngine demo completed!\n");
    return 0;
}

/* ============================================================================
 * 主函数
 * ============================================================================ */

int main(int argc, char* argv[]) {
    /* 默认参数 */
    const char* model_path = DEFAULT_MODEL_PATH;
    const char* prompt = "Hello, how are you?";
    float temperature = 0.7f;
    int top_k = 40;
    int max_tokens = MAX_GEN_TOKENS;

    /* 解析命令行参数 */
    for (int i = 1; i < argc; i++) {
        if (strcmp(argv[i], "-m") == 0 && i + 1 < argc) {
            model_path = argv[++i];
        } else if (strcmp(argv[i], "-p") == 0 && i + 1 < argc) {
            prompt = argv[++i];
        } else if (strcmp(argv[i], "-t") == 0 && i + 1 < argc) {
            temperature = atof(argv[++i]);
        } else if (strcmp(argv[i], "-k") == 0 && i + 1 < argc) {
            top_k = atoi(argv[++i]);
        } else if (strcmp(argv[i], "-n") == 0 && i + 1 < argc) {
            max_tokens = atoi(argv[++i]);
        } else if (strcmp(argv[i], "-h") == 0) {
            print_usage(argv[0]);
            return 0;
        } else {
            fprintf(stderr, "Unknown option: %s\n", argv[i]);
            print_usage(argv[0]);
            return 1;
        }
    }

    /* 打印欢迎信息 */
    printf("╔════════════════════════════════════════════════════════════╗\n");
    printf("║                    MyLLM Chat Demo                          ║\n");
    printf("║              Pure C LLM Inference Engine                    ║\n");
    printf("║                    Version %s                               ║\n", myllm_version());
    printf("╚════════════════════════════════════════════════════════════╝\n");

    /* 运行演示 */
    int ret = 0;

    /* 演示 1: 模型信息 */
    ret = demo_model_info(model_path);
    if (ret != 0) {
        printf("\n[Note] To run this demo, please provide a valid model path:\n");
        printf("       %s -m /path/to/your/model\n\n", argv[0]);
        return ret;
    }

    /* 演示 2: LLaMA 模型结构 */
    ret = demo_llama_inference(model_path);
    if (ret != 0) {
        return ret;
    }

    /* 演示 3: 完整推理 */
    ret = demo_engine_inference(model_path, prompt, temperature, top_k, max_tokens);

    return ret;
}
