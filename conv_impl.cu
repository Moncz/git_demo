#include <iostream>
#include <vector>
#include <chrono>
#include <cmath>
#include <cudnn.h>
#include <cuda_runtime.h>
#include <cuda_fp16.h>

#define CHECK_CUDA(call) {                                         \
    cudaError_t status_ = call;                                    \
    if (status_ != cudaSuccess) {                                  \
        std::cerr << "CUDA Error: " << cudaGetErrorString(status_) << " at " << __FILE__ << ":" << __LINE__ << std::endl; \
        exit(EXIT_FAILURE);                                        \
    }                                                              \
}

#define CHECK_CUDNN(call) {                                        \
    cudnnStatus_t status_ = call;                                  \
    if (status_ != CUDNN_STATUS_SUCCESS) {                         \
        std::cerr << "cuDNN Error: " << cudnnGetErrorString(status_) << " at " << __FILE__ << ":" << __LINE__ << std::endl; \
        exit(EXIT_FAILURE);                                        \
    }                                                              \
}

struct ConvParam {
    int batch, in_c, h, w, out_c, k_h, k_w, pad_h, pad_w, stride_h, stride_w;
};

// CPU参考实现（FP32计算保障精度）
void cpu_conv_ref(const float* input, const float* filter, float* output, const ConvParam& p) {
    int out_h = (p.h - p.k_h + 2 * p.pad_h) / p.stride_h + 1;
    int out_w = (p.w - p.k_w + 2 * p.pad_w) / p.stride_w + 1;
    
    for (int n = 0; n < p.batch; ++n) {
        for (int oc = 0; oc < p.out_c; ++oc) {
            for (int oh = 0; oh < out_h; ++oh) {
                for (int ow = 0; ow < out_w; ++ow) {
                    float sum = 0.0f;
                    for (int ic = 0; ic < p.in_c; ++ic) {
                        for (int kh = 0; kh < p.k_h; ++kh) {
                            for (int kw = 0; kw < p.k_w; ++kw) {
                                int ih = oh * p.stride_h - p.pad_h + kh;
                                int iw = ow * p.stride_w - p.pad_w + kw;
                                if (ih >= 0 && ih < p.h && iw >= 0 && iw < p.w) {
                                    size_t input_idx = n * p.in_c * p.h * p.w + ic * p.h * p.w + ih * p.w + iw;
                                    size_t filter_idx = oc * p.in_c * p.k_h * p.k_w + ic * p.k_h * p.k_w + kh * p.k_w + kw;
                                    sum += input[input_idx] * filter[filter_idx];
                                }
                            }
                        }
                    }
                    size_t output_idx = n * p.out_c * out_h * out_w + oc * out_h * out_w + oh * out_w + ow;
                    output[output_idx] = sum;
                }
            }
        }
    }
}

// FP16数值验证（允许1%的相对误差）
bool verify(const __half* gpu_result, const float* cpu_ref, size_t size) {
    const float eps = 0.01f; // 1% 误差容限
    int error_count = 0;
    const int max_errors_to_print = 10;
    
    for (size_t i = 0; i < size; ++i) {
        float gpu_val = __half2float(gpu_result[i]);
        float ref_val = cpu_ref[i];
        
        // 处理特殊情况
        if (std::isnan(gpu_val) || std::isnan(ref_val)) {
            if (error_count < max_errors_to_print) {
                printf("Error at index %zu: GPU=NaN, CPU=%.6f\n", i, ref_val);
            }
            error_count++;
            continue;
        }
        
        // 计算相对误差
        float abs_error = fabs(gpu_val - ref_val);
        float rel_error = (fabs(ref_val) > 1e-6) ? abs_error / fabs(ref_val) : abs_error;
        
        if (rel_error > eps) {
            if (error_count < max_errors_to_print) {
                printf("Error at index %zu: GPU=%.6f, CPU=%.6f, rel_error=%.6f\n", 
                       i, gpu_val, ref_val, rel_error);
            }
            error_count++;
        }
    }
    
    if (error_count > 0) {
        printf("Total errors: %d/%zu (%.2f%%)\n", error_count, size, 
               (float)error_count / size * 100.0f);
        return false;
    }
    return true;
}

void run_benchmark(const ConvParam& param) {
    // 维度计算
    int out_h = (param.h - param.k_h + 2 * param.pad_h) / param.stride_h + 1;
    int out_w = (param.w - param.k_w + 2 * param.pad_w) / param.stride_w + 1;

    size_t input_size = param.batch * param.in_c * param.h * param.w;
    size_t filter_size = param.out_c * param.in_c * param.k_h * param.k_w;
    size_t output_size = param.batch * param.out_c * out_h * out_w;

    // 主机内存分配（FP32用于参考计算）
    float* h_input = new float[input_size];
    float* h_filter = new float[filter_size];
    float* h_cpu_out = new float[output_size];
    __half* h_gpu_out = new __half[output_size];
    __half *d_input, *d_filter, *d_output;

    // 初始化数据（限制在[-0.1, 0.1]范围内避免FP16溢出）
    srand(42); // 固定随机种子以便复现
    for (size_t i = 0; i < input_size; ++i) 
        h_input[i] = (float)rand() / RAND_MAX * 0.2f - 0.1f;
    for (size_t i = 0; i < filter_size; ++i) 
        h_filter[i] = (float)rand() / RAND_MAX * 0.2f - 0.1f;

    // 打印输入数据的统计信息
    float input_min = 1.0f, input_max = -1.0f;
    float filter_min = 1.0f, filter_max = -1.0f;
    for (size_t i = 0; i < input_size; ++i) {
        input_min = std::min(input_min, h_input[i]);
        input_max = std::max(input_max, h_input[i]);
    }
    for (size_t i = 0; i < filter_size; ++i) {
        filter_min = std::min(filter_min, h_filter[i]);
        filter_max = std::max(filter_max, h_filter[i]);
    }
    printf("Input range: [%.6f, %.6f]\n", input_min, input_max);
    printf("Filter range: [%.6f, %.6f]\n", filter_min, filter_max);

    // 设备内存分配
    CHECK_CUDA(cudaMalloc(&d_input, input_size * sizeof(__half)));
    CHECK_CUDA(cudaMalloc(&d_filter, filter_size * sizeof(__half)));
    CHECK_CUDA(cudaMalloc(&d_output, output_size * sizeof(__half)));

    // 转换到FP16并拷贝到设备
    std::vector<__half> h_input_half(input_size), h_filter_half(filter_size);
    for (size_t i = 0; i < input_size; ++i) 
        h_input_half[i] = __float2half(h_input[i]);
    for (size_t i = 0; i < filter_size; ++i) 
        h_filter_half[i] = __float2half(h_filter[i]);
    
    CHECK_CUDA(cudaMemcpy(d_input, h_input_half.data(), input_size * sizeof(__half), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(d_filter, h_filter_half.data(), filter_size * sizeof(__half), cudaMemcpyHostToDevice));

    // CuDNN初始化
    cudnnHandle_t cudnn;
    cudnnCreate(&cudnn);

    // 创建描述符（统一使用NCHW布局）
    cudnnTensorDescriptor_t input_desc, output_desc;
    cudnnFilterDescriptor_t filter_desc;
    cudnnConvolutionDescriptor_t conv_desc;

    // 创建描述符（统一使用NCHW布局）
    cudnnCreateTensorDescriptor(&input_desc);
    CHECK_CUDNN(cudnnSetTensor4dDescriptor(input_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_HALF,
                                          param.batch, param.in_c, param.h, param.w));
    
    cudnnCreateFilterDescriptor(&filter_desc);
    CHECK_CUDNN(cudnnSetFilter4dDescriptor(filter_desc, CUDNN_DATA_HALF, CUDNN_TENSOR_NCHW,
                                          param.out_c, param.in_c, param.k_h, param.k_w));
    
    cudnnCreateConvolutionDescriptor(&conv_desc);
    CHECK_CUDNN(cudnnSetConvolution2dDescriptor(conv_desc, 
        param.pad_h, param.pad_w, 
        param.stride_h, param.stride_w, 
        1, 1, // dilation
        CUDNN_CROSS_CORRELATION, CUDNN_DATA_HALF));
    
    // 使用cudnnGetConvolution2dForwardOutputDim计算输出维度
    int conv_out_n, conv_out_c, conv_out_h, conv_out_w;
    CHECK_CUDNN(cudnnGetConvolution2dForwardOutputDim(conv_desc, input_desc, filter_desc,
                                                     &conv_out_n, &conv_out_c, &conv_out_h, &conv_out_w));
    
    cudnnCreateTensorDescriptor(&output_desc);
    CHECK_CUDNN(cudnnSetTensor4dDescriptor(output_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_HALF,
                                          conv_out_n, conv_out_c, conv_out_h, conv_out_w));

    // 验证输出维度
    cudnnDataType_t dataType;
    int n, c, h, w, nStride, cStride, hStride, wStride;
    CHECK_CUDNN(cudnnGetTensor4dDescriptor(output_desc, &dataType, &n, &c, &h, &w, 
                                          &nStride, &cStride, &hStride, &wStride));
    
    if (n != conv_out_n || c != conv_out_c || h != conv_out_h || w != conv_out_w) {
        std::cerr << "Error: Output tensor dimensions mismatch!" << std::endl;
        std::cerr << "Expected: " << conv_out_n << "x" << conv_out_c << "x" << conv_out_h << "x" << conv_out_w << std::endl;
        std::cerr << "Got: " << n << "x" << c << "x" << h << "x" << w << std::endl;
        exit(EXIT_FAILURE);
    }

    // 自动选取算法
    cudnnConvolutionFwdAlgoPerf_t perfResults[4];
    int returnedAlgoCount = 0;
    CHECK_CUDNN(cudnnFindConvolutionForwardAlgorithmEx(
        cudnn, input_desc, d_input,
        filter_desc, d_filter,
        conv_desc,
        output_desc, d_output,
        4, &returnedAlgoCount, perfResults,
        nullptr, 0));

    if (returnedAlgoCount == 0) {
        std::cerr << "Error: No suitable convolution algorithm found!" << std::endl;
        exit(EXIT_FAILURE);
    }

    cudnnConvolutionFwdAlgo_t algo = perfResults[0].algo;
    size_t workspace_size = perfResults[0].memory;
    void* d_workspace = nullptr;
    if (workspace_size > 0) {
        CHECK_CUDA(cudaMalloc(&d_workspace, workspace_size));
    }

    // 运行GPU卷积（预热+计时）
    float alpha = 1.0f, beta = 0.0f; // alpha、beta始终为float类型
    
    // 预热
    CHECK_CUDNN(cudnnConvolutionForward(cudnn, &alpha, 
        input_desc, d_input, 
        filter_desc, d_filter, 
        conv_desc, algo, 
        d_workspace, workspace_size, 
        &beta, output_desc, d_output));
    
    // 计时循环（运行100次求平均）
    const int repeats = 100;
    cudaEvent_t start, stop;
    CHECK_CUDA(cudaEventCreate(&start));
    CHECK_CUDA(cudaEventCreate(&stop));
    
    CHECK_CUDA(cudaEventRecord(start));
    for (int i = 0; i < repeats; ++i) {
        CHECK_CUDNN(cudnnConvolutionForward(cudnn, &alpha, 
            input_desc, d_input, 
            filter_desc, d_filter, 
            conv_desc, algo, 
            d_workspace, workspace_size, 
            &beta, output_desc, d_output));
    }
    CHECK_CUDA(cudaEventRecord(stop));
    CHECK_CUDA(cudaEventSynchronize(stop));
    
    float total_ms;
    CHECK_CUDA(cudaEventElapsedTime(&total_ms, start, stop));
    float avg_us = (total_ms * 1000.0f) / repeats; // 平均微秒数

    // 将结果拷贝回主机
    CHECK_CUDA(cudaMemcpy(h_gpu_out, d_output, output_size * sizeof(__half), cudaMemcpyDeviceToHost));

    // CPU参考计算
    cpu_conv_ref(h_input, h_filter, h_cpu_out, param);

    // 打印CPU结果的统计信息
    float cpu_min = 1.0f, cpu_max = -1.0f;
    for (size_t i = 0; i < output_size; ++i) {
        cpu_min = std::min(cpu_min, h_cpu_out[i]);
        cpu_max = std::max(cpu_max, h_cpu_out[i]);
    }
    printf("CPU output range: [%.6f, %.6f]\n", cpu_min, cpu_max);

    // 验证结果
    bool success = verify(h_gpu_out, h_cpu_out, output_size);

    // 打印报告
    printf("Param=[N=%d, C_in=%d, H=%d, W=%d, C_out=%d, K=%dx%d, pad_h=%d, pad_w=%d, stride_h=%d, stride_w=%d]\n",
           param.batch, param.in_c, param.h, param.w, param.out_c, 
           param.k_h, param.k_w, param.pad_h, param.pad_w, param.stride_h, param.stride_w);
    printf("  Time: %.2f us\tResult: %s\n", avg_us, success ? "Pass" : "Fail");

    // 资源清理
    delete[] h_input; delete[] h_filter; delete[] h_cpu_out; delete[] h_gpu_out;
    CHECK_CUDA(cudaFree(d_input)); CHECK_CUDA(cudaFree(d_filter)); 
    CHECK_CUDA(cudaFree(d_output)); if (d_workspace) CHECK_CUDA(cudaFree(d_workspace));
    
    cudnnDestroyTensorDescriptor(input_desc);
    cudnnDestroyTensorDescriptor(output_desc);
    cudnnDestroyFilterDescriptor(filter_desc);
    cudnnDestroyConvolutionDescriptor(conv_desc);
    cudnnDestroy(cudnn);
}

int main() {
    // 六组不同的卷积参数（使用完整的参数结构）
    std::vector<ConvParam> params = {
        {16, 128, 64, 64, 27, 3,3,1,1,1,1},
        {16, 256, 32,32, 256,3,3,1,1,1,1},
        {16, 64, 128, 128, 64,3,3,1,1,1,1},
        {2, 1920, 32, 32,640,3,3,1,1,1,1},
        {2, 640,64,64,640,3,3,1,1,1,1},
        {2, 320,64,64,4,3,3,1,1,1,1}
    };

    for (const auto& param : params) {
        run_benchmark(param);
    }

    return 0;
}
