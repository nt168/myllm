cmake_minimum_required(VERSION 3.14)
project(myllm VERSION 0.1.0 LANGUAGES C)

set(CMAKE_C_STANDARD 11)
set(CMAKE_C_STANDARD_REQUIRED ON)

# 编译选项
set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -Wall -Wextra -Werror=return-type")
set(CMAKE_C_FLAGS_DEBUG "-g -O0 -DDEBUG")
set(CMAKE_C_FLAGS_RELEASE "-O3 -DNDEBUG")

# 架构特定优化
if(CMAKE_SYSTEM_PROCESSOR MATCHES "x86_64|AMD64")
    set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -mavx2 -mfma")
elseif(CMAKE_SYSTEM_PROCESSOR MATCHES "aarch64|arm64")
    set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS}")
endif()

# 源文件
set(MYLLM_SOURCES
    # 张量模块
    src/tensor/shape.c
    src/tensor/tensor.c

    # 运算模块
    src/ops/elementwise.c
    src/ops/activation.c
    src/ops/normalization.c
    src/ops/matmul.c
    src/ops/rope.c

    # 神经网络层模块 (nn_ 前缀，与 models 目录的实现共存)
    src/nn/linear.c
    src/nn/embedding.c
    src/nn/rmsnorm.c
    src/nn/mlp.c

    # KV 缓存模块
    src/kv/kv.c

    # 加载器模块
    src/loader/safetensors.c
    src/loader/loader.c

    # 分词器模块
    src/tokenizer/tokenizer.c

    # 采样器模块
    src/sampler/sampler.c

    # 推理引擎模块
    src/inference/engine.c

    # 量化模块
    src/quant/config.c
    src/quant/fp8.c
    src/quant/quant.c

    # 工具模块
    src/utils/log.c
    src/utils/timer.c
    src/utils/error.c
    src/utils/memory.c
    src/utils/utils.c

    # LLaMA 模型 (包含 linear, rmsnorm, mlp, embedding 实现)
    src/models/llama/attention.c
    src/models/llama/block.c
    src/models/llama/model.c

    # Qwen2 模型
    src/models/qwen2/model.c

    # Qwen3 模型
    src/models/qwen3/model.c
)

# 静态库
add_library(myllm_static STATIC ${MYLLM_SOURCES})
target_include_directories(myllm_static PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}/include
)
set_target_properties(myllm_static PROPERTIES OUTPUT_NAME myllm)

# 动态库
add_library(myllm_shared SHARED ${MYLLM_SOURCES})
target_include_directories(myllm_shared PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}/include
)
set_target_properties(myllm_shared PROPERTIES OUTPUT_NAME myllm)

# 测试
option(MYLLM_BUILD_TESTS "Build tests" ON)
if(MYLLM_BUILD_TESTS)
    enable_testing()

    # Tensor 模块测试
    add_executable(test_tensor tests/unit/tensor/test_tensor.c)
    target_include_directories(test_tensor PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}/include
        ${CMAKE_CURRENT_SOURCE_DIR}/tests
    )
    target_link_libraries(test_tensor myllm_static m)
    add_test(NAME test_tensor COMMAND test_tensor)

    # Ops 模块测试
    add_executable(test_ops tests/unit/ops/test_ops.c)
    target_include_directories(test_ops PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}/include
        ${CMAKE_CURRENT_SOURCE_DIR}/tests
    )
    target_link_libraries(test_ops myllm_static m)
    add_test(NAME test_ops COMMAND test_ops)

    # NN 模块测试
    add_executable(test_nn tests/unit/nn/test_nn.c)
    target_include_directories(test_nn PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}/include
        ${CMAKE_CURRENT_SOURCE_DIR}/tests
    )
    target_link_libraries(test_nn myllm_static m)
    add_test(NAME test_nn COMMAND test_nn)

    # KV 模块测试
    add_executable(test_kv tests/unit/kv/test_kv.c)
    target_include_directories(test_kv PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}/include
        ${CMAKE_CURRENT_SOURCE_DIR}/tests
    )
    target_link_libraries(test_kv myllm_static m)
    add_test(NAME test_kv COMMAND test_kv)

    # Loader 模块测试
    add_executable(test_loader tests/unit/loader/test_loader.c)
    target_include_directories(test_loader PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}/include
        ${CMAKE_CURRENT_SOURCE_DIR}/tests
    )
    target_link_libraries(test_loader myllm_static m)
    add_test(NAME test_loader COMMAND test_loader)

    # Tokenizer 模块测试
    add_executable(test_tokenizer tests/unit/tokenizer/test_tokenizer.c)
    target_include_directories(test_tokenizer PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}/include
        ${CMAKE_CURRENT_SOURCE_DIR}/tests
    )
    target_link_libraries(test_tokenizer myllm_static m)
    add_test(NAME test_tokenizer COMMAND test_tokenizer)

    # Sampler 模块测试
    add_executable(test_sampler tests/unit/sampler/test_sampler.c)
    target_include_directories(test_sampler PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}/include
        ${CMAKE_CURRENT_SOURCE_DIR}/tests
    )
    target_link_libraries(test_sampler myllm_static m)
    add_test(NAME test_sampler COMMAND test_sampler)

    # Inference 模块测试
    add_executable(test_inference tests/unit/inference/test_inference.c)
    target_include_directories(test_inference PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}/include
        ${CMAKE_CURRENT_SOURCE_DIR}/tests
    )
    target_link_libraries(test_inference myllm_static m)
    add_test(NAME test_inference COMMAND test_inference)

    # Quant 模块测试
    add_executable(test_quant tests/unit/quant/test_quant.c)
    target_include_directories(test_quant PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}/include
        ${CMAKE_CURRENT_SOURCE_DIR}/tests
    )
    target_link_libraries(test_quant myllm_static m)
    add_test(NAME test_quant COMMAND test_quant)

    # Utils 模块测试
    add_executable(test_utils tests/unit/utils/test_utils.c)
    target_include_directories(test_utils PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}/include
        ${CMAKE_CURRENT_SOURCE_DIR}/tests
    )
    target_link_libraries(test_utils myllm_static m)
    add_test(NAME test_utils COMMAND test_utils)

    add_executable(test_llama tests/unit/models/test_llama.c)
    target_include_directories(test_llama PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}/include
        ${CMAKE_CURRENT_SOURCE_DIR}/tests
    )
    target_link_libraries(test_llama myllm_static m)
    add_test(NAME test_llama COMMAND test_llama)

    add_executable(test_qwen2 tests/unit/models/test_qwen2.c)
    target_include_directories(test_qwen2 PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}/include
        ${CMAKE_CURRENT_SOURCE_DIR}/tests
    )
    target_link_libraries(test_qwen2 myllm_static m)
    add_test(NAME test_qwen2 COMMAND test_qwen2)

    # Qwen3 测试暂时禁用（缺少 ops_rope 等实现）
    # add_executable(test_qwen3 tests/unit/models/test_qwen3.c)
    # target_include_directories(test_qwen3 PRIVATE
    #     ${CMAKE_CURRENT_SOURCE_DIR}/include
    #     ${CMAKE_CURRENT_SOURCE_DIR}/tests
    # )
    # target_link_libraries(test_qwen3 myllm_static m)
    # add_test(NAME test_qwen3 COMMAND test_qwen3)
endif()

# 示例
option(MYLLM_BUILD_EXAMPLES "Build examples" ON)
if(MYLLM_BUILD_EXAMPLES)
    add_executable(chat_demo examples/chat_demo.c)
    target_link_libraries(chat_demo myllm_static m)
endif()

# 安装
include(GNUInstallDirs)

install(TARGETS myllm_static myllm_shared
    ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR}
    LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
)

install(DIRECTORY include/
    DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}
)

# 配置文件
configure_file(
    ${CMAKE_CURRENT_SOURCE_DIR}/cmake/myllmConfig.cmake.in
    ${CMAKE_CURRENT_BINARY_DIR}/myllmConfig.cmake
    @ONLY
)

install(FILES ${CMAKE_CURRENT_BINARY_DIR}/myllmConfig.cmake
    DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/myllm
)
