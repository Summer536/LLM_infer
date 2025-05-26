#pragma once
#include <cuda_runtime.h>
#include <cuda.h>
#include <cuda_fp16.h>
#include "src/utils/tensor.h"
#include "src/utils/vectorize_utils.h"
template <typename T>
void launchAddResidual( // residual.shape = [num tokens, hidden_units], batch_size = num tokens, n_dims = hidden_units
    TensorWrapper<T> *residual,  //输入：来自上一个RMSnorm的残差（上一个RMSnorm是融合算子Fused AddbiasResidual nad RMSnorm,我们取Fused AddbiasResidual的输出值作为这边的输入！为什么呢？因为残差就是指RMSnorm之前的原始值）
    TensorWrapper<T> *decoder_out, //输入兼输出：来自Downliear的输出 // [num tokens, hidden_units]
    bool is_print=false //这个是后面debug时候需要的参数，不需要管
    );
