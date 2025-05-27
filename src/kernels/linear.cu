#include "src/kernels/linear.h"
#include <iostream>
#include <fstream>
#include "src/utils/cuda_debug_utils.cuh"

// all matmul cases:
// A(输入) * B(weigths) = C(输出)
//一、注意力机制相关线性变换：
// 1.ctx（contextdecoder）      A:qkv lienar: [num_tokens, qhiddenunits] *  B:[q hiddenunits, hiddenunits]     = C:{num_tokens, qkv_head_num,  head_size}
//   将输入特征转换为查询(Q)、键(K)、值(V)三种表示，用于自注意力计算

// 2.ctx attn output linear:   A:{num_tokens, head_num, head_size}      *  B:{q hidden units, q hidden units}  = C:{num_tokens, q hidden units}
//   将多头注意力的输出重新映射回原始维度空间

// 3.self qkv linear:          A:[bs, q hidden units]                   *  B:[qhiddenunits, hiddenunits]       = C:{bs, qkv_head_num,  head_size}}
//   类似ctx qkv，但处理的是自回归生成阶段的输入

// 4.self attn output linear:  A:{batch_size, q hidden_units}           *  B:[qhiddenunits, qhiddenunits]      = C:[bs, q hiddenunits]
//   类似ctx attn output，但处理的是自回归生成阶段的注意力输出

//二、前馈神经网络(FFN)相关的线性变换：
// 5.gate:                     A:[bs/token nums, q hidden units]        *  B:[q hidden units, inter size]      = C:[bs/token nums, inter size]
//   FFN中的门控机制，控制信息流动

// 6.up:                       A:[bs/token nums, q hidden units]        *  B:[q hidden units, inter size]      = C:[bs/token nums, inter size]
//   FFN中的上投影，将特征映射到更高维度

// 7.fusedGateUpGemm:          A:[bs/token nums, q hidden units]        *  B:[q hidden units, 2 * inter size]  = C:[bs/token nums, 2 * inter size] 
//   将gate和up两个操作融合为一个GEMM操作，提高计算效率                                                                                                    

// 8.down:                     A:[bs/token nums, inter size]            *  B:[q hidden units, inter size]      = C:[bs/token nums, q hidden units]
//   FFN中的下投影，将特征映射回原始维度

//三、输出层：
// 9.lmhead linear:            A:[bs, q hidden units]                   *  B:[vocab size, q hiden units], need transpose B(这个需要做个转置！)
//   将模型的隐藏状态映射到词汇表大小的logits，用于最终的token预测

template <typename T>
void launchLinearGemm(TensorWrapper<T> *input,
                      BaseWeight<T> &weight,
                      TensorWrapper<T> *output,
                      cublasWrapper *cublas_wrapper, //cubals_wrapper：封装关于cubals库的一切东西 （相关文档2.8.12：https://docs.nvidia.com/cuda/archive/12.3.1/cublas/index.html#cublasgemmex）
                      bool trans_a,
                      bool trans_b) 
{
    int Am = weight.shape[1];
    int Ak = weight.shape[0];
    int Bk = input->shape[1];
    int Bn = input->shape[0];
    int Cm = output->shape[1];
    int Cn = output->shape[0];

    Cm = output->shape.size() == 3 ? output->shape[1] * output->shape[2] : output->shape[1];
    Bk = input->shape.size() == 3 ? input->shape[1] * input->shape[2] : input->shape[1];

    int lda = Am;
    int ldb = Bk;
    int ldc = Cm;

    cublasOperation_t transA = trans_b ? CUBLAS_OP_T : CUBLAS_OP_N;
    cublasOperation_t transB = trans_a ? CUBLAS_OP_T : CUBLAS_OP_N;

    cublas_wrapper->Gemm(transA, transB,
                         trans_b ? Ak : Am,
                         Cn, Bk,
                         weight.data, lda,
                         input->data, ldb,
                         output->data, ldc,
                         1.0f, 0.0f);
}

//对于维度超过2的矩阵，首先需要先经过stride函数来将矩阵的最后两维取出，例如对qk矩阵，取出[seqlen, seqlen]
template <typename T>
void launchLinearStridedBatchGemm(TensorWrapper<T> *input1,
                                  TensorWrapper<T> *input2,
                                  TensorWrapper<T> *output,
                                  cublasWrapper *cublas_wrapper,
                                  bool trans_a,
                                  bool trans_b)
{
    int Bm = input1->shape[2];
    int Bk = input1->shape[3];
    int Ak = input2->shape[2];
    int An = input2->shape[3];
    int Cm = output->shape[2];
    int Cn = output->shape[3];
    int lda = An;
    int ldb = Bk;
    int ldc = Cn;

    int64_t strideA = Ak * An;
    int64_t strideB = Bm * Bk;
    int64_t strideC = Cm * Cn;

    int batchCount = input1->shape[0] * input1->shape[1];

    cublasOperation_t transA = trans_b ? CUBLAS_OP_T : CUBLAS_OP_N;
    cublasOperation_t transB = trans_a ? CUBLAS_OP_T : CUBLAS_OP_N;

    cublas_wrapper->strideBatchedGemm(transA, transB,
                                      Cn, Cm, Bk, 
                                      input2->data, lda, strideA,
                                      input1->data, ldb, strideB,
                                      output->data, ldc, strideC,
                                      batchCount,
                                      1.0f, 0.0f);
}



template void launchLinearGemm<float>(TensorWrapper<float> *input,
                                      BaseWeight<float> &weight,
                                      TensorWrapper<float> *output,
                                      cublasWrapper *cublas_wrapper,
                                      bool trans_a,
                                      bool trans_b);
                                      
template void launchLinearStridedBatchGemm<float>(TensorWrapper<float> *input1,
                                                    TensorWrapper<float> *input2,
                                                    TensorWrapper<float> *output,
                                                    cublasWrapper *cublas_wrapper,
                                                    bool trans_a,
                                                    bool trans_b);
                                                    
template void launchLinearGemm<half>(TensorWrapper<half> *input,
                                     BaseWeight<half> &weight,
                                     TensorWrapper<half> *output,
                                     cublasWrapper *cublas_wrapper,
                                     bool trans_a,
                                     bool trans_b);
                                     
template void launchLinearStridedBatchGemm<half>(TensorWrapper<half> *input1,
                                                   TensorWrapper<half> *input2,
                                                   TensorWrapper<half> *output,
                                                   cublasWrapper *cublas_wrapper,
                                                   bool trans_a,
                                                   bool trans_b);

template void launchLinearGemm<int8_t>(TensorWrapper<int8_t> *input,
                                     BaseWeight<int8_t> &weight,
                                     TensorWrapper<int8_t> *output,
                                     cublasWrapper *cublas_wrapper,
                                     bool trans_a,
                                     bool trans_b);
                                     
template void launchLinearStridedBatchGemm<int8_t>(TensorWrapper<int8_t> *input1,
                                                   TensorWrapper<int8_t> *input2,
                                                   TensorWrapper<int8_t> *output,
                                                   cublasWrapper *cublas_wrapper,
                                                   bool trans_a,
                                                   bool trans_b);
                                                    