// ============================================================================
//  cuda_check.h  --  CUDA 错误处理宏
// ----------------------------------------------------------------------------
//  本项目的约定：每一次 cudaMalloc / cudaMemcpy / kernel 启动后面都跟一次检查。
//  一次静默失败的启动只会印出一张全黑图，却要花一小时去查，所以一次都不跳过。
// ============================================================================
#pragma once

#include <stdio.h>
#include <stdlib.h>

// 检查单个 CUDA 运行时调用；失败时打印「文件:行号」并退出。
#define CUDA_CHECK(call)                                                          \
    do {                                                                          \
        const cudaError_t err__ = (call);                                          \
        if (err__ != cudaSuccess) {                                                \
            fprintf(stderr, "\n[CUDA ERROR] %s:%d\n  call      : %s\n  code      : %d (%s)\n  message   : %s\n", \
                    __FILE__, __LINE__, #call, (int)err__,                            \
                    cudaGetErrorName(err__), cudaGetErrorString(err__));               \
            exit(EXIT_FAILURE);                                                    \
        }                                                                          \
    } while (0)

// 同上，但多打一行「上下文」说明（kernel 启动之后特别有用）。
#define CUDA_CHECK_MSG(call, msg)                                                  \
    do {                                                                          \
        const cudaError_t err__ = (call);                                          \
        if (err__ != cudaSuccess) {                                                \
            fprintf(stderr, "\n[CUDA ERROR] %s:%d\n  context   : %s\n  call      : %s\n  code      : %d (%s)\n  message   : %s\n", \
                    __FILE__, __LINE__, (msg), #call, (int)err__,                      \
                    cudaGetErrorName(err__), cudaGetErrorString(err__));               \
            exit(EXIT_FAILURE);                                                    \
        }                                                                          \
    } while (0)

// ---------------------------------------------------------------------------
//  下面两个是「纯转发」宏：它们**不自己套 do{ }while(0)**。
//
//  为什么统一成「都不包裹」：CUDA_CHECK_MSG 展开后本身就是一个 do{ }while(0)，
//  转发宏再包一层只是重复。原来 CUDA_CHECK_LAST 不包、CUDA_CHECK_SYNC 包，
//  读起来像两种不同的东西，实际行为完全一样 —— 统一之后，转发宏的意图很清楚：
//  它只负责把 call 和 msg 换掉，语句封装的活交给被转发的那个宏。
// ---------------------------------------------------------------------------

// 启动 kernel 之后立刻调用：把「启动配置错误」（寄存器过多、grid 尺寸非法…）
// 在它变成一张黑图之前就报出来。
#define CUDA_CHECK_LAST(msg) CUDA_CHECK_MSG(cudaGetLastError(), msg)

// 等待设备，并报告异步执行之后才暴露出来的错误。
#define CUDA_CHECK_SYNC(msg) CUDA_CHECK_MSG(cudaDeviceSynchronize(), msg)
