// ============================================================================
//  interactive.h  --  Win32 实时预览（可选功能，GDI 实现，无需 OpenGL）
// ----------------------------------------------------------------------------
//  为什么用 GDI 而不是 OpenGL：
//    CUDA-OpenGL Interop 需要打开额外的环境/依赖（GL 头、上下文、报备），
//    而 Windows 自带的 GDI 一行就能把一块 32 位 DIB 拉伸贴到窗口上：
//        StretchDIBits(hdc, 0,0, winW,winH, 0,0, srcW,srcH, pixels, &bmi, DIB_RGB_COLORS, SRCCOPY);
//    GDI 的 32 位 DIB 是 BGRA 顺序，而渲染器输出 RGBA，所以贴图前换一次 R/B。
//
//  为什么是「轨道相机」而不是自由飞行：
//    场景固定在原点附近，轨道相机（yaw / pitch / distance + 一个目标点）不会迷路，
//    按键也少，正好够展示「实时预览」这件事。
//
//  按键（窗口标题栏里也会写一遍）：
//      左/右 或 A/D : yaw（绕 Y 轴转）      上/下 或 W/S : pitch（抬高/俯视）
//      Q / E        : 拉远 / 拉近           R            : 强制重渲染
//      ESC / 关窗   : 退出
//
//  为什么用 PeekMessage 而不是 GetMessage：
//    预览循环里「取消息 -> 需要就重渲染 -> 睡一会儿」都要做，GetMessage 会一直阻塞
//    到有消息为止，重渲染就没机会发生。PeekMessage + Sleep(10) 是这种小工具的标准写法。
//
//  只在按键后重渲染（320x180 / spp1 一次只要几毫秒），所以不会有「拖着卡」的问题。
// ============================================================================
#pragma once

#if !defined(_WIN32)

// 非 Windows 平台：给一个同名空实现，保证这份代码本身不会被平台问题卡住。
inline int runInteractivePreview(int, int, void*,
                                 void (*)(void*, const Vec3&, const Vec3&, unsigned char*),
                                 const Vec3&, const Vec3&, const char*)
{
    return 0;
}

#else

#ifndef WIN32_LEAN_AND_MEAN
#  define WIN32_LEAN_AND_MEAN      // 少拖一堆用不上的头（OLE/壳层），避免宏污染
#endif
#ifndef NOMINMAX
#  define NOMINMAX                 // 不许 windows.h 定义 min/max 宏：它会和 C++ 标准库打架
#endif
#include <windows.h>

#include <stdio.h>
#include <stdlib.h>     // malloc / free：预览缓冲
#include <string.h>
#include <math.h>

#include "vec3.h"

// 渲染回调：把 (eye, target) 渲成一张 RGBA 图（分辨率、spp 由 userData 里的设置决定）。
typedef void (*PreviewRenderFn)(void* userData, const Vec3& eye, const Vec3& target, unsigned char* rgbaOut);

// 单窗口小程序：状态直接放文件级静态变量，避免把 WndProc 写成需要指针传递的复杂结构。
struct PreviewState
{
    HWND           hwnd;
    int            srcW, srcH;         // 渲染分辨率（会被拉伸到窗口客户区）
    unsigned char* bgra;               // 显示用缓冲（BGRA）
    BITMAPINFO     bmi;
    Vec3           target;             // 轨道中心
    float          yaw, pitch, dist;
    bool           dirty;              // 需要重渲染
    bool           quit;
    float          lastMs;
    void*          userData;
    PreviewRenderFn renderFn;
};

static PreviewState g_preview;

static void previewBlit(HDC hdc)
{
    StretchDIBits(hdc,
                  0, 0, g_preview.srcW * 2, g_preview.srcH * 2,   // 目标：客户区 2 倍面积
                  0, 0, g_preview.srcW, g_preview.srcH,           // 源：整张渲染图
                  g_preview.bgra, &g_preview.bmi, DIB_RGB_COLORS, SRCCOPY);
}

static void previewUpdateTitle()
{
    char title[256];
    snprintf(title, sizeof(title),
             "whitted preview  %.1f ms  yaw %.0f  pitch %.0f  dist %.1f  (arrows/WASD move, Q/E zoom, R redraw, ESC quit)",
             g_preview.lastMs,
             g_preview.yaw * 57.29578f, g_preview.pitch * 57.29578f, g_preview.dist);
    SetWindowTextA(g_preview.hwnd, title);
}

static LRESULT CALLBACK previewWndProc(HWND hwnd, UINT msg, WPARAM wp, LPARAM lp)
{
    (void)lp;

    switch (msg)
    {
    case WM_DESTROY:
        g_preview.quit = true;
        PostQuitMessage(0);
        return 0;

    case WM_PAINT:
    {
        PAINTSTRUCT ps;
        HDC hdc = BeginPaint(hwnd, &ps);
        previewBlit(hdc);
        EndPaint(hwnd, &ps);
        return 0;
    }

    case WM_KEYDOWN:
    {
        const float dYaw = 0.09f;
        const float dPitch = 0.07f;
        const float dDist = 0.6f;

        switch (wp)
        {
        case VK_LEFT:  case 'A': g_preview.yaw   -= dYaw;   g_preview.dirty = true; break;
        case VK_RIGHT: case 'D': g_preview.yaw   += dYaw;   g_preview.dirty = true; break;
        case VK_UP:    case 'W': g_preview.pitch -= dPitch; g_preview.dirty = true; break;
        case VK_DOWN:  case 'S': g_preview.pitch += dPitch; g_preview.dirty = true; break;
        case 'Q':                g_preview.dist  += dDist;  g_preview.dirty = true; break;
        case 'E':                g_preview.dist  -= dDist;  g_preview.dirty = true; break;
        case 'R':                g_preview.dirty = true; break;
        case VK_ESCAPE:          DestroyWindow(hwnd); break;
        default: break;
        }

        // 夹紧：俯仰角贴到 ±83 度（再大视线会和 up 向量平行，相机基退化）
        if (g_preview.pitch >  1.45f) g_preview.pitch =  1.45f;
        if (g_preview.pitch < -1.45f) g_preview.pitch = -1.45f;
        if (g_preview.dist < 2.0f)    g_preview.dist = 2.0f;
        if (g_preview.dist > 60.0f)   g_preview.dist = 60.0f;
        return 0;
    }

    default:
        break;
    }
    return DefWindowProcA(hwnd, msg, wp, lp);
}

// 打开预览窗口并进入消息循环。
// width/height 是「渲染分辨率」，窗口客户区固定是它的 2 倍（低分辨率渲染 + 放大显示）。
// 返回 0 表示正常退出，非 0 表示窗口创建失败。
inline int runInteractivePreview(int width, int height, void* userData, PreviewRenderFn renderFn,
                                 const Vec3& initialEye, const Vec3& initialTarget, const char* title)
{
    if (width < 32 || height < 32) return 1;

    memset(&g_preview, 0, sizeof(g_preview));
    g_preview.srcW = width;
    g_preview.srcH = height;
    g_preview.userData = userData;
    g_preview.renderFn = renderFn;
    g_preview.target = initialTarget;
    g_preview.dirty = true;

    // 由初始相机位置反推轨道参数（yaw / pitch / distance）
    const Vec3  offset = initialEye - initialTarget;
    const float horizontal = sqrtf(offset.x * offset.x + offset.z * offset.z);
    g_preview.dist = length(offset);
    g_preview.yaw = atan2f(offset.x, offset.z);
    g_preview.pitch = atan2f(offset.y, horizontal);

    g_preview.bgra = (unsigned char*)malloc((size_t)width * (size_t)height * 4u);
    if (g_preview.bgra == NULL) return 1;

    memset(&g_preview.bmi, 0, sizeof(g_preview.bmi));
    g_preview.bmi.bmiHeader.biSize = sizeof(BITMAPINFOHEADER);
    g_preview.bmi.bmiHeader.biWidth = width;
    g_preview.bmi.bmiHeader.biHeight = -height;    // 负数 = 自上而下：和渲染缓冲的行序一致
    g_preview.bmi.bmiHeader.biPlanes = 1;
    g_preview.bmi.bmiHeader.biBitCount = 32;
    g_preview.bmi.bmiHeader.biCompression = BI_RGB;

    WNDCLASSEXA wc;
    memset(&wc, 0, sizeof(wc));
    wc.cbSize = sizeof(wc);
    wc.style = CS_OWNDC;
    wc.lpfnWndProc = previewWndProc;
    wc.hInstance = GetModuleHandleA(NULL);
    wc.hCursor = LoadCursor(NULL, IDC_ARROW);
    wc.lpszClassName = "WhittedPreviewWindow";
    if (RegisterClassExA(&wc) == 0 && GetLastError() != ERROR_CLASS_ALREADY_EXISTS) return 1;

    // 让「客户区」正好是 2 倍渲染分辨率
    RECT rc;
    rc.left = 0; rc.top = 0; rc.right = width * 2; rc.bottom = height * 2;
    AdjustWindowRect(&rc, WS_OVERLAPPEDWINDOW, FALSE);

    g_preview.hwnd = CreateWindowExA(0, wc.lpszClassName,
                                     (title != NULL) ? title : "whitted preview",
                                     WS_OVERLAPPEDWINDOW,
                                     CW_USEDEFAULT, CW_USEDEFAULT,
                                     rc.right - rc.left, rc.bottom - rc.top,
                                     NULL, NULL, wc.hInstance, NULL);
    if (g_preview.hwnd == NULL) { free(g_preview.bgra); g_preview.bgra = NULL; return 1; }

    ShowWindow(g_preview.hwnd, SW_SHOW);
    UpdateWindow(g_preview.hwnd);

    unsigned char* rgba = (unsigned char*)malloc((size_t)width * (size_t)height * 4u);
    if (rgba == NULL) { DestroyWindow(g_preview.hwnd); return 1; }

    printf("  交互预览: 渲染 %d x %d，窗口 %d x %d；按键见窗口标题（ESC 退出）\n",
           width, height, width * 2, height * 2);

    while (!g_preview.quit)
    {
        MSG msg;
        while (PeekMessageA(&msg, NULL, 0, 0, PM_REMOVE))
        {
            if (msg.message == WM_QUIT) { g_preview.quit = true; break; }
            TranslateMessage(&msg);
            DispatchMessageA(&msg);
        }
        if (g_preview.quit) break;

        if (g_preview.dirty)
        {
            // 轨道相机：目标点固定，眼睛在球面上
            const Vec3 eye = g_preview.target + Vec3(cosf(g_preview.pitch) * sinf(g_preview.yaw) * g_preview.dist,
                                                     sinf(g_preview.pitch) * g_preview.dist,
                                                     cosf(g_preview.pitch) * cosf(g_preview.yaw) * g_preview.dist);

            const DWORD t0 = GetTickCount();
            g_preview.renderFn(g_preview.userData, eye, g_preview.target, rgba);
            g_preview.lastMs = (float)(GetTickCount() - t0);

            // RGBA -> BGRA（GDI 的 32 位 DIB 是蓝在前）
            const size_t pixels = (size_t)width * (size_t)height;
            for (size_t i = 0; i < pixels; ++i)
            {
                g_preview.bgra[i * 4 + 0] = rgba[i * 4 + 2];
                g_preview.bgra[i * 4 + 1] = rgba[i * 4 + 1];
                g_preview.bgra[i * 4 + 2] = rgba[i * 4 + 0];
                g_preview.bgra[i * 4 + 3] = 255;
            }

            HDC hdc = GetDC(g_preview.hwnd);
            previewBlit(hdc);
            ReleaseDC(g_preview.hwnd, hdc);
            previewUpdateTitle();

            g_preview.dirty = false;
        }

        Sleep(10);   // 不抢 CPU：预览是「按键才动」，没必要空转
    }

    free(rgba);
    free(g_preview.bgra);
    g_preview.bgra = NULL;
    return 0;
}

#endif   // _WIN32
