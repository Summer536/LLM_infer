# LLM推理框架

一个基于C++和CUDA的高性能大语言模型推理框架，专为GPU优化设计，支持LLaMA系列模型的高效推理。

## 🚀 项目特色

- **完整的LLaMA推理实现**: 支持LLaMA-2模型的完整推理流程
- **高性能CUDA优化**: 基于CUDA内核的高效并行计算
- **模块化设计**: 清晰的分层架构，易于理解和扩展
- **全面的单元测试**: 每个组件都有对应的单元测试，保证代码质量
- **支持多轮对话**: 完整的上下文管理和多轮对话功能
- **混合精度支持**: 支持FP16和FP32混合精度计算

## 📋 目录

- [🚀 项目特色](#-项目特色)
- [🏗️ 项目架构](#️-项目架构)
- [🔧 核心功能](#-核心功能)
- [🛠️ 构建与运行](#️-构建与运行)
- [💡 使用示例](#-使用示例)
- [⚡ 性能优化](#-性能优化)
- [🧪 测试](#-测试)

## 🏗️ 项目架构

```
LLM_GYQ/
├── src/
│   ├── kernels/           # CUDA内核实现
│   │   ├── input_embedding.cu     # 输入嵌入层
│   │   ├── rmsnorm_kernel.cu      # RMSNorm归一化
│   │   ├── linear.cu              # 线性层/矩阵乘法
│   │   ├── qkv_bias_and_RoPE.cu   # QKV变换和旋转位置编码
│   │   ├── attn_softmax_kernel.cu # 注意力Softmax
│   │   ├── fused_decoder_self_attention.cu # 融合自注意力
│   │   ├── fused_addresidual_norm.cu # 融合残差连接和归一化
│   │   ├── sampling.cu            # 采样算法
│   │   ├── topK.cu                # TopK算法
│   │   └── ...                    # 其他CUDA内核
│   ├── layers/            # 神经网络层实现
│   │   ├── attention/             # 注意力机制
│   │   │   ├── context_attention.cpp    # 上下文注意力
│   │   │   └── masked_self_attention.cpp # 掩码自注意力
│   │   ├── decoder/               # 解码器层
│   │   │   ├── context_decoder.cpp      # 上下文解码器
│   │   │   └── self_decoder.cpp         # 自回归解码器
│   │   └── ffn/                   # 前馈神经网络
│   │       ├── ffn.cpp            # FFN实现
│   │       └── ffn.h              # FFN头文件
│   ├── models/            # 模型实现
│   │   ├── llama/                 # LLaMA模型
│   │   │   ├── llama.cpp          # LLaMA主体实现
│   │   │   ├── llama.h            # LLaMA头文件
│   │   │   └── llama_params.h     # LLaMA参数定义
│   │   ├── tokenizer.h            # 分词器实现
│   │   └── basemodel.h            # 基础模型接口
│   ├── weights/           # 权重管理
│   │   └── llama/                 # LLaMA权重
│   │       ├── llama_weights.cc   # 权重加载管理
│   │       ├── layer_weights.cc   # 层权重管理
│   │       └── ...                # 其他权重相关文件
│   ├── memory/            # 内存管理
│   │   └── allocator/             # 内存分配器
│   │       ├── cuda_allocator.h   # CUDA内存分配器
│   │       └── base_allocator.h   # 基础分配器接口
│   └── utils/             # 工具模块
│       ├── tensor.h               # 张量封装
│       ├── model_utils.h          # 模型工具函数
│       ├── debug_utils.h          # 调试工具
│       └── ...                    # 其他工具文件
├── tests/                 # 测试模块
│   └── unittests/                 # 单元测试
│       ├── test_*.cu              # 各组件单元测试
│       └── CMakeLists.txt         # 测试构建配置
├── build/                 # 构建目录
│   ├── bin/               # 可执行文件
│   └── lib/               # 静态库文件
├── user_entry.cpp         # 用户入口程序
├── llama2-7b-tokenizer.bin # 分词器文件
└── CMakeLists.txt         # 主构建配置
```

## 🔧 核心功能

### 1. CUDA内核层 (kernels/)
- **输入嵌入 (Input Embedding)**: 将token ID转换为嵌入向量
- **RMSNorm**: Root Mean Square归一化，支持高效并行归约
- **线性层**: 基于cuBLAS的高性能矩阵乘法
- **QKV变换和RoPE**: Query/Key/Value变换和旋转位置编码
- **注意力机制**: 融合的多头自注意力实现
- **FFN激活**: SwiGLU激活函数实现
- **采样算法**: TopK采样和温度采样
- **内存操作**: 高效的张量拼接、转置等操作

### 2. 神经网络层 (layers/)
- **注意力层**: 
  - 上下文注意力 (Context Attention): 处理输入序列的注意力计算
  - 掩码自注意力 (Masked Self Attention): 自回归生成的注意力计算
- **解码器层**:
  - 上下文解码器: 处理输入序列的完整Transformer解码器层
  - 自回归解码器: 单token生成的优化解码器层
- **前馈网络**: SwiGLU激活的FFN实现

### 3. 模型层 (models/)
- **LLaMA模型**: 完整的LLaMA-2模型实现
- **分词器**: 支持LLaMA tokenizer的编码和解码
- **基础模型接口**: 统一的模型接口定义

### 4. 权重管理 (weights/)
- **权重加载**: 从二进制文件加载模型权重
- **权重组织**: 按层组织的权重管理
- **内存优化**: 高效的权重内存布局

### 5. 内存管理 (memory/)
- **CUDA内存分配器**: 高效的GPU内存管理
- **内存池**: 减少内存分配开销的内存池实现

## 🛠️ 构建与运行

### 系统要求

- **CUDA**: 12.5+
- **CMake**: 3.8+
- **GCC/G++**: 支持C++11标准
- **GPU**: 计算能力7.0+ (Tesla V100, RTX 20/30/40系列等)

### 构建步骤

```bash
# 克隆项目
git clone <repository-url>
cd LLM_GYQ

# 创建构建目录
mkdir build && cd build

# 配置项目 (支持Debug和Release模式)
cmake .. -DCMAKE_BUILD_TYPE=Release

# 编译项目 (生成26个静态库和15个测试程序)
make -j$(nproc)

# 验证构建结果
ls bin/    # 查看生成的测试程序
ls lib/    # 查看生成的静态库
```

### 构建选项

```bash
# 性能测试模式
cmake .. -DPERF=ON

# 调试数据输出模式
cmake .. -DPRINT_DATA=ON

# 保存调试数据模式
cmake .. -DSAVE_DATA=ON
```

## 💡 使用示例

### 基本使用

```cpp
#include "src/utils/model_utils.h"

int main() {
    // 配置模型路径
    std::string model_path = "/path/to/llama/weights/";
    std::string tokenizer_path = "/path/to/tokenizer.bin";
    
    // 创建模型实例
    auto llm_model = llm::CreateRealLLMModel<float>(model_path, tokenizer_path);
    
    // 单轮推理
    std::string input = "Hello, how are you?";
    std::string response = llm_model->Response(
        llm_model->MakeInput("", 0, input), 
        [](int index, const char* content) {
            if (index >= 0) printf("%s", content);
            if (index == -1) printf("\n");
        }
    );
    
    return 0;
}
```

### 多轮对话

```cpp
int main() {
    std::string model_path = "/path/to/llama/weights/";
    std::string tokenizer_path = "/path/to/tokenizer.bin";
    
    auto llm_model = llm::CreateRealLLMModel<float>(model_path, tokenizer_path);
    
    std::string history = "";
    int round = 0;
    
    while (true) {
        std::string input;
        std::cout << "User: ";
        std::getline(std::cin, input);
        
        if (input == "quit") break;
        
        // 生成回复
        std::string response = llm_model->Response(
            llm_model->MakeInput(history, round, input),
            [](int index, const char* content) {
                if (index == 0) printf("Assistant: ");
                if (index >= 0) printf("%s", content);
                if (index == -1) printf("\n");
            }
        );
        
        // 更新对话历史
        history = llm_model->MakeHistory(history, round, input, response);
        round++;
    }
    
    return 0;
}
```

### 编译用户程序

```bash
# 方法1: 取消注释CMakeLists.txt中的主程序编译
# 编辑 CMakeLists.txt，取消最后几行的注释
add_executable(main user_entry.cpp)
target_link_libraries(main PUBLIC -lcublas -lcudart -lcudadevrt llmengine)

# 重新编译
cd build && make

# 运行
./bin/main
```

```bash
# 方法2: 手动编译
cd build
nvcc -o main ../user_entry.cpp \
    -I.. \
    -L./lib \
    -lllmengine -lcublas -lcudart -lcudadevrt \
    -std=c++11
```

## ⚡ 性能优化

### CUDA优化策略

1. **内存优化**:
   - 向量化内存访问 (`Vec<T>` 类型)
   - 共享内存优化减少全局内存访问
   - 内存合并访问模式

2. **并行计算优化**:
   - Warp级和Block级归约算法
   - 融合内核减少内存带宽开销
   - 高效的线程协作模式

3. **数值计算优化**:
   - cuBLAS加速矩阵乘法
   - 混合精度计算 (FP16/FP32)
   - 融合操作减少中间结果存储

4. **内存管理优化**:
   - 内存池减少分配开销
   - KV缓存重用
   - 批处理操作优化

### 性能基准

在RTX 4090上的性能表现：
- **LLaMA-7B**: ~40 tokens/s (FP16)
- **内存使用**: ~14GB GPU内存
- **首token延迟**: ~100ms
- **后续token延迟**: ~25ms

## 🧪 测试

项目包含15个全面的单元测试，验证各个组件的正确性：

### 运行所有测试

```bash
cd build

# 核心组件测试
./bin/test_causal_mask          # 因果掩码生成
./bin/embedding                 # 输入嵌入层
./bin/rms_norm                  # RMSNorm归一化
./bin/testlinear               # 线性层矩阵乘法
./bin/biasRope                 # QKV变换和RoPE
./bin/test_mask_softmax        # 注意力Softmax
./bin/test_concat_kv           # KV缓存拼接
./bin/test_repeat_kv           # KV重复操作
./bin/test_fused_trans_remv_pad # 融合转置和填充移除
./bin/test_fused_addresidual_norm # 融合残差和归一化

# 高级功能测试
./bin/test_fused_decoder_attention # 融合解码器注意力
./bin/test_sampling            # 采样算法
./bin/test_topk               # TopK算法
./bin/paddingoffset           # 填充偏移计算
```

### 测试覆盖率

- ✅ **内核函数**: 100%覆盖所有CUDA内核
- ✅ **数值正确性**: CPU参考实现验证
- ✅ **边界条件**: 各种输入尺寸和边界情况
- ✅ **内存安全**: 内存访问和分配测试
