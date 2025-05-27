#include <algorithm>  
#include <iostream>   
#include <math.h>   
#include <stdlib.h>    
#include <string>      
#include <vector>      
#include <stdio.h>
#include <fstream>
#include <random>

#include "src/utils/macro.h"
#include "src/weights/base_weights.h"
#include "src/kernels/linear.h"

// Usage:
// `./test_linear` to test fp32 kernel
// `./test_linear fp16` to test fp16 kernel
// `./test_linear int8` to test int8 kernel  //这个test没通过，可能是我的GPU型号不支持这个int8

template<typename T>
void CPUlinear(T *input, T *weight, float *output, // Output is float for CPU reference
               int m, int k, int n) {
    for(int i = 0; i < m; i++){
        for(int j = 0; j < n; j++){
            float sum = 0.0f;
            for(int l = 0; l < k; l++){
                sum += (float)input[i * k + l] * (float)weight[l * n + j];
            }
            output[i * n + j] = sum;
        }
    }
}

template<typename T>
bool CheckResult(float *CPUoutput, T *GPUoutput, int output_size) {
    float tolerance = 1e-3f;
    if (std::is_same<T, half>::value) {
        tolerance = 1e-1f; // Looser tolerance for FP16
    } else if (std::is_same<T, int8_t>::value) {
        tolerance = 1.0f;  // Looser tolerance for INT8, can be significant
    }

    for(int i = 0; i < output_size; i++){
        if(i < 5){
            printf("Result %d: CPU = %f, GPU = %f\n", i, CPUoutput[i], (float)GPUoutput[i]);
        }
        if(fabs(CPUoutput[i] - (float)GPUoutput[i]) > tolerance){
            printf("Mismatch at %d, CPU = %f, GPU = %f (diff: %f)\n", i, CPUoutput[i], (float)GPUoutput[i], fabs(CPUoutput[i] - (float)GPUoutput[i]));
            return false;
        }
    }
    return true;
}

template<typename T>
void test_linear_kernel(){
    const int seqlen = 13;
    const int hidden_units = 64; // Reduced for faster testing, original 4096
    // const int vocab_size = 32; // Not used in this specific test
    // const int inter_size = 10; // Not used in this specific test
    int matrix_k = hidden_units;
    int matrix_n = hidden_units;

    int input_matrix_m = seqlen;
    int input_matrix_k = hidden_units;

    int weight_matrix_m = hidden_units;
    int weight_matrix_k = matrix_k; //This is K for weight, M for input
    int weight_matrix_n = matrix_n;

    int output_matrix_m = seqlen;
    int output_matrix_n = matrix_n;

    int weight_size = weight_matrix_m * weight_matrix_n; // K * N for weight
    int input_size = input_matrix_m * input_matrix_k; // M * K for input
    int output_size = output_matrix_m * output_matrix_n; // M * N for output
    
    printf("Testing %s linear kernel\n", typeid(T).name());
    printf("Input: %d x %d, Weight: %d x %d, Output: %d x %d\n", 
           input_matrix_m, input_matrix_k, weight_matrix_k, weight_matrix_n, output_matrix_m, output_matrix_n);

    T *h_w = (T *)malloc(weight_size * sizeof(T));
    T *d_w;
    cudaMalloc(&d_w, weight_size * sizeof(T));

    std::mt19937 gen(3407);
    
    for(int i = 0; i < weight_size; i++){
        if (std::is_same<T, int8_t>::value) {
             std::uniform_int_distribution<> distrib(-5, 5);
             h_w[i] = (T)distrib(gen);
        } else {
            std::uniform_real_distribution<> distrib(0.0f, 1.0f);
            h_w[i] = (T)distrib(gen);
        }
    }

    T* h_in = (T*) malloc(sizeof(T) * input_size);
    T* d_in;
    cudaMalloc((void**)&d_in, sizeof(T) * input_size);
    for(int i = 0; i < input_size; i++) {
       if (std::is_same<T, int8_t>::value) {
            std::uniform_int_distribution<> distrib(-5, 5);
            h_in[i] = (T)distrib(gen);
        } else {
            std::uniform_real_distribution<> distrib(0.0f, 1.0f);
            h_in[i] = (T)distrib(gen);
        }
    }

    T *h_out_gpu = (T *)malloc(output_size * sizeof(T));
    T *d_out;
    cudaMalloc(&d_out, output_size * sizeof(T));
    
    CHECK(cudaMemcpy(d_w, h_w, weight_size * sizeof(T), cudaMemcpyHostToDevice));
    CHECK(cudaMemcpy(d_in, h_in, sizeof(T) * input_size, cudaMemcpyHostToDevice));

    DataType type = getTensorType<T>();
    WeightType wtype = getWeightType<T>();
    
    TensorWrapper<T> *in = new TensorWrapper<T>(Device::GPU, type, {input_matrix_m, input_matrix_k}, d_in);
    BaseWeight<T> weight;
    // Note: cuBLAS expects column-major for weights if not transposed.
    // GEMM: C = op(A) * op(B)
    // Here, A is weight, B is input.
    // If weight is [K, N] and input is [M, K], and we want M x N output.
    // launchLinearGemm arguments: (transA, transB, M_C, N_C, K_C, ...)
    // M_C = N_weight (if transB=N) or M_weight (if transB=T)
    // N_C = M_input (if transA=N) or N_input (if transA=T)
    // K_C = M_weight (if transB=N) or N_weight (if transB=T)
    // Default call in launchLinearGemm implies weight is A, input is B
    // And it computes C = B * A (if trans_a=N, trans_b=N for cublas)
    // But the parameter names in launchLinearGemm are a bit confusing (transA for weight, transB for input)
    // cublas: C_mn = A_mk * B_kn  => (m, n, k)
    // launchLinearGemm (transA for weight, transB for input):
    // cublas_wrapper->Gemm(transB, transA, N_input, M_weight, K_input, ...)
    // For C_output[M_input, N_weight] = Input[M_input, K_input] * Weight[K_input, N_weight]
    // Weight shape {K, N} -> Ak=K, Am=N in launchLinearGemm
    // Input shape {M, K} -> Bk=K, Bn=M in launchLinearGemm
    // Output shape {M, N}
    // Arguments for cublas_wrapper->Gemm(trans_input, trans_weight, N_weight, M_input, K_input, ...)
    // This seems to map to: cublas_wrapper->Gemm(transB=CUBLAS_OP_N, transA=CUBLAS_OP_N, N_w, M_in, K_common, weight.data, K_common, input.data, K_common, output.data, N_w)
    // With weight.shape[0] = K, weight.shape[1] = N. (Original code has Am=N, Ak=K)
    // input.shape[0] = M, input.shape[1] = K. (Original code has Bn=M, Bk=K)
    // The original launchLinearGemm has: (transA=op(weight), transB=op(input))
    // cublas_wrapper->Gemm(transB, transA, M_C= (trans_b ? Ak : Am) , N_C=Cn, K_C=Bk, ...)
    // Ak = weight.shape[0] (K_w), Am = weight.shape[1] (N_w)
    // Bk = input.shape[1] (K_in), Bn = input.shape[0] (M_in)
    // Cm = output.shape[1], Cn = output.shape[0]
    // If weight is KxN (Ak x Am) and input is MxK (Bn x Bk), output MxN (Cn x Cm)
    // cublas(op(B_input), op(A_weight), N_output, M_output, K_common, ...)
    // op(A_weight) = trans_a, op(B_input) = trans_b
    // M_C = N_output, N_C = M_output, K_C = K_common
    // In launchLinearGemm: Gemm(op(input), op(weight), output_N_dim, output_M_dim, common_K_dim, ...)
    // M_gemm = output_N_dim = (trans_a ? weight.shape[0] : weight.shape[1])  // N_weight
    // N_gemm = output_M_dim = input.shape[0] // M_input
    // K_gemm = input.shape[1] // K_input (common_K)
    // (trans_a for weight, trans_b for input in launchLinearGemm corresponds to transB, transA in cublas call)
    // The mapping seems to be: C_MxN = Input_MxK * Weight_KxN
    // C(output_matrix_m, output_matrix_n) = Input(input_matrix_m, input_matrix_k) * Weight(input_matrix_k, output_matrix_n)
    // So weight.shape = {input_matrix_k, output_matrix_n}

    weight.shape = {input_matrix_k, output_matrix_n};
    weight.data = d_w;
    weight.type = wtype;

    TensorWrapper<T> *out = new TensorWrapper<T>(Device::GPU, type, {output_matrix_m, output_matrix_n}, d_out);

    cublasHandle_t cublas_handle;
    cublasLtHandle_t cublaslt_handle;
    cublasCreate(&cublas_handle);
    cublasLtCreate(&cublaslt_handle); // Initialize cublasLtHandle
    cublasSetMathMode(cublas_handle, CUBLAS_DEFAULT_MATH); // or CUBLAS_TENSOR_OP_MATH for tensor cores
    cublasWrapper *cublas_wrapper = new cublasWrapper(cublas_handle, cublaslt_handle);
    
    if (std::is_same<T, float>::value) {
        cublas_wrapper->setFP32GemmConfig();
    } else if (std::is_same<T, half>::value) {
        cublas_wrapper->setFP16GemmConfig();
    } else if (std::is_same<T, int8_t>::value) {
        cublas_wrapper->setINT8GemmConfig();
    }

    std::cout << "before launch linear gemm kernel" << std::endl;
    // For C = A * B (Input * Weight), A is M x K, B is K x N. No transpose needed for A or B.
    // In launchLinearGemm, input is B, weight is A in cublas call C = B*A
    // So if our conceptual model is C_out = Input_MxK * Weight_KxN
    // launchLinearGemm(Input_B, Weight_A, Output_C, cublas_wrapper, trans_Weight_A=false, trans_Input_B=false)
    // cublas call: Gemm(trans_Input_B, trans_Weight_A, N_weight, M_input, K_common, ...)
    launchLinearGemm(in, weight, out, cublas_wrapper, false, false); // trans_weight=false, trans_input=false
    std::cout << "after launch linear gemm kernel" << std::endl;

    CHECK(cudaMemcpy(h_out_gpu, d_out, output_size * sizeof(T), cudaMemcpyDeviceToHost));
    float* CPUout_ref = (float*) malloc(output_size * sizeof(float)); // CPU reference is always float
    
    CPUlinear(h_in, h_w, CPUout_ref, input_matrix_m, input_matrix_k, output_matrix_n);

    bool check_result = CheckResult<T>(CPUout_ref, h_out_gpu, output_size);
    if(check_result){
        std::cout << "Test passed" << std::endl;
    } else {
        std::cout << "Test failed" << std::endl;
    }

    free(h_w);
    free(h_in);
    free(h_out_gpu);
    free(CPUout_ref);
    cudaFree(d_w);
    cudaFree(d_in);
    cudaFree(d_out);
    delete in;
    delete out;
    // delete weight; // weight.data is d_w, managed elsewhere
    cublasDestroy(cublas_handle);
    cublasLtDestroy(cublaslt_handle); // Destroy cublasLtHandle
    delete cublas_wrapper;
}

int main(int argc, char *argv[]){
    if (argc > 1) {
        std::string arg = argv[1];
        if (arg == "fp16") {
            test_linear_kernel<half>();
        } else if (arg == "int8") {
            test_linear_kernel<int8_t>();
        } else {
            test_linear_kernel<float>(); // Default or if arg is not fp16/int8
        }
    } else {
        test_linear_kernel<float>();
    }
    return 0;
}

