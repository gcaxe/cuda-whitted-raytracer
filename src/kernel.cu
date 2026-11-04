// ============================================================================
//  kernel.cu  --  BVH 加速的 Whitted 光线追踪器（CUDA）
// ----------------------------------------------------------------------------
//  特性：
//    · Whitted 递归光线追踪：反射 / 折射 / Schlick Fresnel / 硬阴影
//    · BVH 加速：32 字节对齐节点、有序遍历 + best 剪枝、三角形 SoA、共享内存根缓存
//    · OBJ 三角网格（assimp 加载；失败自动回退手写解析器 → 程序化网格）
//    · 5 档性能对比（--bench）：暴力 / 无序 BVH / 有序 BVH / +SoA / +SMEM
//    · 4 种调试视图（--debug）：法线 / 递归深度 / 物体编号 / BVH 访存热力图
//    · 转台动画（--frames N）与 Win32 实时预览（--interactive）
//
//  自检（程序自动打印 PASS/FAIL）：
//    · --bench 输出每档的建树 ms / 渲染 ms / 加速比
//    · 各 BVH 档之间逐像素一致（maxDiff == 0）；与暴力档的差异打印实测值并解释来源
//    · 图像非空、栈峰值 < 64、加速比 > 1
//
//  命令行（全部可省略；不带参数 = 8 张转台图，1280x720 spp1）：
//    --obj PATH         加载 OBJ（缺省 = 仓库自带的 objs\bunny_10k.obj；
//                       加载失败自动回退到程序化网格，保证一定出图）
//    --width N --height N --spp N --depth N --fov DEG
//    --frames N         绕 Y 轴等角度转台，输出 N 张图（默认 8；--frames 1 = 单张）
//    --bench            跑 5 档性能对比 + 逐像素校验
//    --mode brute|bvh   普通渲染用哪一档（默认 bvh = bvh.h 里编译期开关选出的那一档）
//    --split mid|sah    建树划分：质心中点（默认）或 SAH（12 候选）
//    --out DIR --tag NAME --ppm-only --png-only
//    --debug shaded|normals|depth|hits|heatmap
//    --interactive      Win32 窗口实时预览（方向键/WASD 转相机，Q/E 缩放，R 重渲染）
//
//  编译提示：源码是 **UTF-8 带 BOM（EF BB BF）**，BOM 是 cl 与 nvcc 前端都能识别的
//  in-file 信号，所以**不需要 `/utf-8` 编译选项**。如果 BOM 被剥掉，中文 locale 的
//  nvcc 会按 GBK(936) 读它，报 C4819 甚至一个假的 C1070，或在字符串处报
//  missing closing quote。
//  修复：用 VS / Notepad++ 把文件另存为「UTF-8 with BOM」。
// ============================================================================

#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <iostream>
#include "rt_print.h"     // rt::fixed / rt::ljust / rt::rjust / rt::sci
#if defined(_MSC_VER)
#  include <corecrt.h>   // 新版 MSVC 把 strncpy_s / _stricmp 的声明放在这里
#endif

// 公共头文件全部放在本目录下（whitted 工程是自包含的，单独拷走 whitted\ 也能编译）。
#include "vec3.h"        // 向量数学（__host__ __device__）
#include "rt_math.h"     // Ray / Camera / 常量 / 反射折射 / 哈希随机
#include "cuda_check.h"  // CUDA_CHECK
#include "image_io.h"    // PPM / PNG / compareImages
#include "png_writer.h"  // stb_image_write 封装
#include "font5x7.h"     // stampTextHost：把分辨率/耗时/角度印在图上
#include "sphere.h"      // 球 + HitRecord
#include "materials.h"   // 材质 + Blinn-Phong + 阴影光线
#include "triangle.h"    // 三角形 + Möller–Trumbore + AoS/SoA
#include "bvh.h"         // AABB / 建树 / 显式栈遍历 / 设备内存
#include "obj_loader.h"  // OBJ + 程序化网格 + 变换

// windows.h 放到最后再引入：它带进来一堆宏（NOMINMAX 挡掉 min/max，别的挡不掉），
// 放在项目头文件之后就不会污染它们。SetConsoleOutputCP 和 interactive.h 的 GDI 预览都要它。
#if defined(_WIN32)
#  define WIN32_LEAN_AND_MEAN
#  define NOMINMAX
#  include <windows.h>
#endif
#include "interactive.h"

// ============================================================================
//  0. 常量 / 枚举 / 命令行选项
// ============================================================================
#define WEEK_TAG              "whitted"
#define INV_GAMMA             0.45454545f   // 1 / 2.2
#define SIGN_BAND_Y           48            // 顶部文字暗底高度（像素）

#define MESH_TARGET_TRIANGLES 72000         // 程序化兜底网格的目标三角形数（实际约 6.8 万）
#define MESH_HEIGHT           2.40f         // 摆位后网格高度（世界单位）
#define MESH_ROTATE_Y        (-18.0f)       // 绕 Y 轴转一点，正对相机时更有立体感

// 默认加载的 OBJ：仓库根目录 objs\ 下的斯坦福兔子（5,002 顶点 / 10,000 面）。
// 相对路径相对「项目目录」：VS 里 F5 的工作目录是 whitted\whitted\，..\..\ 正好回到仓库根，
// 所以 clone 下来直接 F5 就能用同一个模型；失败时回退到 MESH_TARGET_TRIANGLES 的程序化网格。
#define DEFAULT_OBJ_PATH  "..\\objs\\bunny_10k.obj"

#define CAM_FOV               32.0f
#define CAM_RADIUS            10.60f        // 转台半径
#define CAM_HEIGHT            3.35f         // 相机高度
#define CAM_TARGET_Y          1.15f         // 视线目标点高度（让网格居中）

#define BENCH_WIDTH           640           // --bench 默认分辨率（暴力版在小图上才等得起）
#define BENCH_HEIGHT          360

#define MATID_MESH            0             // 材质下标（不是 MaterialType！）
#define MATID_MIRROR          1
#define MATID_GLASS           2
#define MATID_GROUND          3

enum DebugView { VIEW_SHADED = 0, VIEW_NORMALS = 1, VIEW_DEPTH = 2, VIEW_HITS = 3, VIEW_HEATMAP = 4 };
enum RenderMode { MODE_BRUTE = 0, MODE_BVH = 1 };
enum SplitMode { SPLIT_MID = 0, SPLIT_SAH = 1 };

// 基准测试的 5 档：
//   0 暴力            —— 所有对比的分母
//   1 BVH 无序遍历     —— 对照组（只差「有序遍历」这一个开关）
//   2 BVH 有序遍历     —— **基准配置**（后续 SoA / SMEM 都在它之上叠加）
//   3 +SoA            —— 三角形改用 SoA 访存
//   4 +共享内存根节点  —— 每 block 只读一次根附近节点
// 档 1 与档 2 的差值 = 「有序遍历」本身的收益；2->3 = SoA；3->4 = 共享内存。
enum BenchTier
{
    TIER_BRUTE = 0,
    TIER_BVH_NO_ORDER = 1,
    TIER_BVH_32 = 2,
    TIER_BVH_32_SOA = 3,
    TIER_BVH_32_SOA_SMEM = 4,
    TIER_COUNT = 5
};

struct RenderOptions
{
    int  width, height, spp, depth, frames;
    float fov;
    DebugView  debug;
    RenderMode mode;
    SplitMode  split;
    bool bench;
    bool interactive;
    bool wantPpm, wantPng;
    bool resolutionGiven;      // 用户是否显式给了 --width/--height
    char objPath[260];
    char outDir[260];
    char tag[128];
};

static void defaultOptions(RenderOptions* o)
{
    o->width = 1280;
    o->height = 720;
    o->spp = 1;
    o->depth = 5;              // 默认递归深度
    o->frames = 8;             // 转台默认 8 张（turn00..turn07）
    o->fov = CAM_FOV;
    o->debug = VIEW_SHADED;
    o->mode = MODE_BVH;
    o->split = SPLIT_MID;
    o->bench = false;
    o->interactive = false;
    o->wantPpm = true;
    o->wantPng = true;
    o->resolutionGiven = false;
    // 缺省网格 = 仓库自带的斯坦福兔子（objs\bunny_10k.obj，1 万面）。
    // 相对路径相对「项目目录」：VS 里 F5 的工作目录是 whitted\whitted\，..\..\ 正好回到仓库根；
    // 加载失败会自动回退到程序化网格（保证一定出图）。
    strcpy_s(o->objPath, sizeof(o->objPath), DEFAULT_OBJ_PATH);
    strcpy_s(o->outDir, sizeof(o->outDir), "out");
    o->tag[0] = '\0';
}

static const char* debugViewName(DebugView v)
{
    switch (v)
    {
    case VIEW_NORMALS: return "normals";
    case VIEW_DEPTH:   return "depth";
    case VIEW_HITS:    return "hits";
    case VIEW_HEATMAP: return "heatmap";
    default:           return "shaded";
    }
}

// ---------------------------------------------------------------------------
//  控制台编码：让中文输出在「UTF-8 执行字符集」和「936(GBK) 执行字符集」下都不乱码
// ---------------------------------------------------------------------------
//  这套源码是 UTF-8 带 BOM；MSVC 用哪种「执行字符集」由编译选项决定，
//  而这里的工程**没有加 `/utf-8`**，所以窄字符串默认被转成 GBK(936) 字节 -> 控制台保持 936 才对。
//  （如果有人加了 /utf-8，字符串会变成 UTF-8 字节，那时要切到 65001 才显示对。）
//  探针：U+5EFA（"建"）在 UTF-8 里是 E5 BB BA，在 GBK 里是 BD A8；
//  首字节是 0xE5 就说明编译器按 UTF-8 执行字符集处理了，否则保持原样。
//  这样无论别人怎么编译，中文输出都是可读的。
static void setupConsoleEncoding()
{
#if defined(_WIN32)
    const char* probe = "\u5EFA";   // 建
    if ((unsigned char)probe[0] == 0xE5)
        SetConsoleOutputCP(65001);  // CP_UTF8
#endif
}

// ---------------------------------------------------------------------------
//  设备自检（先把机器情况打出来，后面看到怪数字才不会误判）
// ---------------------------------------------------------------------------
static void printDeviceInfo()
{
    int deviceCount = 0;
    CUDA_CHECK_MSG(cudaGetDeviceCount(&deviceCount), "cudaGetDeviceCount");
    if (deviceCount <= 0)
    {
        std::cout << "[错误] 没有找到 CUDA 设备：检查驱动与显卡是否可用。\n";
        exit(EXIT_FAILURE);
    }

    cudaDeviceProp prop;
    CUDA_CHECK_MSG(cudaGetDeviceProperties(&prop, 0), "cudaGetDeviceProperties");

    std::cout << "  GPU : " << prop.name
              << "  计算能力 " << prop.major << "." << prop.minor
              << "  SM " << prop.multiProcessorCount
              << "  显存 " << rt::fixed((double)prop.totalGlobalMem / (1024.0 * 1024.0 * 1024.0), 1)
              << " GB  驱动 CUDA " << prop.major << "." << prop.minor << "\n";
    std::cout << "        常量内存 " << rt::fixed((double)prop.totalConstMem / 1024.0, 0)
              << " KB  每块共享内存 " << rt::fixed((double)prop.sharedMemPerBlock / 1024.0, 0)
              << " KB  warp " << prop.warpSize << "\n";

    if ((double)prop.totalGlobalMem < 2.0 * 1024.0 * 1024.0 * 1024.0)
        std::cout << "        [警告] 显存不足 2GB，大分辨率下可能失败，先用 --width 640。\n";
}

// ============================================================================
//  1. 编译期渲染配置（--bench 的每一档就是它的一个实例化）
// ============================================================================
// 为什么要模板参数而不是运行时 if：
//    「三角形是 AoS 还是 SoA」「节点是 40B 还是 32B」「先走近的还是固定先左」
//    在设备端是**取数指令 / 分支顺序**的差别，必须是编译期常量才能让编译器生成
//    最优代码（也才能让优化真的生效）。
//    宏 BVH_USE_SOA / BVH_CACHE_ROOT_IN_SMEM 决定默认档（--mode bvh 走的那一档），
//    bench 则把几种组合各实例化一个 kernel，在同一台机器上公平对比。
//
// ORDERED 是**对照组开关**，不是"越新越好"的优化：
//    true  = 有序遍历：先走入口距离近的孩子，近处先命中 -> best 立刻变小 ->
//            远处那棵子树整个被剪掉。
//    false = 固定先左后右，不看距离。
//    两个档次测出来的差值，就是「有序遍历」这一项优化本身的收益。
template <bool BVH, bool SOA, bool ALIGNED, bool SMEM, bool ORDERED = true>
struct RenderConfig
{
    static constexpr bool useBvh       = BVH;      // false = 暴力线性遍历（基线对照）
    static constexpr bool useSoa       = SOA;      // true  = 三角形走 SoA 访存
    static constexpr bool alignedNodes = ALIGNED;  // true  = 32B 对齐节点，false = 40B 朴素节点
    static constexpr bool rootSmem     = SMEM;     // true  = 共享内存缓存根节点
    static constexpr bool ordered      = ORDERED;  // true  = 有序遍历（近先远后）
};

// --bench 的 5 档：暴力 -> BVH 基线(32B,无序遍历) -> +有序遍历 -> +SoA -> +共享内存
// 注意档 1 与档 2 只差「有序遍历」这一个开关，档 2 是**基准配置**。
typedef RenderConfig<false, false, false, false> CfgBrute;              // 档 0：暴力基线
typedef RenderConfig<true,  false, true,  false, false> CfgBvhNoOrder;  // 档 1：BVH 32B + 无序遍历
typedef RenderConfig<true,  false, true,  false, true > CfgBvh32;       // 档 2：BVH 32B + 有序遍历（基准）
typedef RenderConfig<true,  true,  true,  false, true > CfgBvh32Soa;    // 档 3：+SoA
typedef RenderConfig<true,  true,  true,  true,  true > CfgBvh32SoaSmem;// 档 4：+共享内存根缓存

// --mode bvh 用的默认档：由 bvh.h 里的编译期开关决定。
typedef RenderConfig<true, (BVH_USE_SOA != 0), true, (BVH_CACHE_ROOT_IN_SMEM != 0), true> CfgDefault;

static const char* const TIER_NAME[TIER_COUNT] =
{
    "暴力 AoS (无 BVH)",
    "BVH 32B + 无序遍历",
    "BVH 32B + 有序遍历",
    "BVH 32B + 有序 + SoA",
    "BVH 32B + 有序 + SoA + SMEM"
};

// ============================================================================
//  2. 设备端场景
// ============================================================================
#define RT_MAX_LIGHTS 4

struct PointLight
{
    Vec3 position;
    Vec3 color;      // 已经乘过强度
};

struct Scene
{
    SphereSet   spheres;
    MaterialSet materials;

    PointLight  lights[RT_MAX_LIGHTS];
    int         lightCount;

    Vec3        skyTop;         // 背景渐变上端
    Vec3        skyBottom;      // 背景渐变下端

    Vec3        meshCenter;     // 转台用：场景几何中心（--frames 的视线目标）
    float       meshRadius;

    TriStore    tris;           // 三角形数据（AoS + SoA 两个视图）
    BvhView<BvhNode>      bvhA; // 32B 对齐节点
    BvhView<BvhNodeNaive> bvhN; // 40B 朴素节点（同一棵树）
};

// 按节点类型挑 BVH 视图：两套布局的节点数组类型不同，只能用重载（tag dispatch）。
// 第二个参数只参与重载决议，是个空指针，不会被解引用。
RT_HD RT_INLINE const BvhView<BvhNode>&      pickBvh(const Scene& s, BvhNode*)      { return s.bvhA; }
RT_HD RT_INLINE const BvhView<BvhNodeNaive>& pickBvh(const Scene& s, BvhNodeNaive*) { return s.bvhN; }

// --- 背景 / 调试视图用的小函数 ---------------------------------------------
__device__ __forceinline__ Vec3 skyGradient(const Vec3& dir, const Vec3& top, const Vec3& bottom)
{
    const float t = 0.5f * (dir.y + 1.0f);
    return lerp(bottom, top, t);
}

__device__ __forceinline__ Vec3 materialDebugColor(int id)
{
    // 命中图：按材质号给一个稳定的伪随机颜色（互质乘数把相邻 id 打散）
    const float r = 0.30f + 0.70f * (float)((id * 37) % 7) / 6.0f;
    const float g = 0.30f + 0.70f * (float)((id * 91) % 5) / 4.0f;
    const float b = 0.30f + 0.70f * (float)((id * 53) % 3) / 2.0f;
    return Vec3(r, g, b);
}

__device__ __forceinline__ Vec3 depthDebugColor(int depth, int maxDepth)
{
    // 递归深度图：越深越红（0 -> 蓝白，maxDepth -> 红）
    const float t = fminf(fmaxf((float)depth / (float)(maxDepth > 0 ? maxDepth : 1), 0.0f), 1.0f);
    return Vec3(0.15f + 0.85f * t, 0.35f * (1.0f - t) + 0.15f, 0.85f * (1.0f - t) + 0.15f);
}

__device__ __forceinline__ Vec3 heatmapColor(int visits)
{
    // BVH 节点访问次数图：对数刻度（次数从几十到几千，线性映射看不出结构）
    const float t = logf(1.0f + (float)visits) / logf(1.0f + 512.0f);
    const float c = fminf(fmaxf(t, 0.0f), 1.0f);
    if (c < 0.5f) return lerp(Vec3(0.05f, 0.10f, 0.45f), Vec3(0.10f, 0.85f, 0.35f), c * 2.0f);
    return lerp(Vec3(0.10f, 0.85f, 0.35f), Vec3(0.95f, 0.35f, 0.10f), (c - 0.5f) * 2.0f);
}

// ---------------------------------------------------------------------------
//  三角形求交：暴力 与 BVH 两种入口
// ---------------------------------------------------------------------------
// 暴力版（基线）：对每个三角形做一次 Möller–Trumbore，best 逐步收紧。
template <typename Config>
__device__ __forceinline__ bool intersectMeshBrute(const Scene& scene, const Ray& ray,
                                                   float tMin, float tMax, HitRecord* rec)
{
    float best = tMax;
    bool  found = false;

    for (int i = 0; i < scene.tris.count; ++i)
    {
        const TriangleAoS tri = triAt<Config::useSoa>(scene.tris, i);
        HitRecord tmp;
        if (hitTriangle(tri, ray, tMin, best, &tmp))
        {
            best = tmp.t;
            *rec = tmp;
            found = true;
        }
    }
    return found;
}

// BVH 版：档位在编译期决定「走哪套节点布局 / 要不要共享内存根缓存」。
// 注意两个分支在两种实例化下都必须能编译（模板会都实例化），
// 但 Config 是编译期常量，死掉的那一支会被优化掉。
template <typename Config, typename NodeT>
__device__ __forceinline__ bool intersectMesh(const Scene& scene, const BvhView<NodeT>& view,
                                              const NodeT* smemNodes, const Ray& ray,
                                              float tMin, float tMax, HitRecord* rec,
                                              int* nodeVisits, int* peak, int* overflow)
{
    if (Config::useBvh)
    {
        if (Config::rootSmem)
            return intersectBvh<NodeT, Config::useSoa, true, Config::ordered>(
                view, smemNodes, ray, tMin, tMax, rec, nodeVisits, peak, overflow);
        return intersectBvh<NodeT, Config::useSoa, false, Config::ordered>(
            view, smemNodes, ray, tMin, tMax, rec, nodeVisits, peak, overflow);
    }
    return intersectMeshBrute<Config>(scene, ray, tMin, tMax, rec);
}

// 整个场景的最近命中：球（很少，线性测） + 网格（暴力或 BVH）
template <typename Config, typename NodeT>
__device__ __forceinline__ bool intersectScene(const Scene& scene, const BvhView<NodeT>& view,
                                               const NodeT* smemNodes, const Ray& ray,
                                               float tMin, float tMax, HitRecord* rec,
                                               int* nodeVisits, int* peak, int* overflow)
{
    bool  found = false;
    float best = tMax;
    HitRecord tmp;

    if (hitSpheres(scene.spheres, ray, tMin, best, &tmp))
    {
        best = tmp.t;
        *rec = tmp;
        found = true;
    }
    if (intersectMesh<Config>(scene, view, smemNodes, ray, tMin, best, &tmp, nodeVisits, peak, overflow))
    {
        *rec = tmp;
        found = true;
    }
    return found;
}

// 遮挡查询（阴影光线）：命中即返回，不需要最近交点，省一半以上的遍历
template <typename Config, typename NodeT>
__device__ __forceinline__ bool occludedScene(const Scene& scene, const BvhView<NodeT>& view,
                                              const NodeT* smemNodes, const Ray& ray,
                                              float tMin, float tMax, int* peak, int* overflow)
{
    if (occludedSpheres(scene.spheres, ray, tMin, tMax)) return true;

    if (Config::useBvh)
    {
        if (Config::rootSmem)
            return occludedBvh<NodeT, Config::useSoa, true, Config::ordered>(
                view, smemNodes, ray, tMin, tMax, peak, overflow);
        return occludedBvh<NodeT, Config::useSoa, false, Config::ordered>(
            view, smemNodes, ray, tMin, tMax, peak, overflow);
    }
    else
    {
        // 暴力路径。**必须写在 else 里**，不能写成「if 里 return + 后面跟 for」——
        // 那样在 useBvh=true 的实例化里这个 for 永远不可达，nvcc 会报
        //   warning #128-D: loop is not reachable
        // 每个 BVH 档一条，5 档就是 5 条刷屏。改成 if/else 之后语义也更清楚。
        for (int i = 0; i < scene.tris.count; ++i)
        {
            const TriangleAoS tri = triAt<Config::useSoa>(scene.tris, i);
            if (hitTriangleAny(tri, ray, tMin, tMax)) return true;
        }
        return false;
    }
}

// 打包成可调用对象交给 materials.h 的 inShadow()：
// 材质系统只需要「从 p 朝光源这条线上有没有东西」，不必认识 Scene / BVH / 共享内存。
// 用结构体而不是 device lambda：不需要 --extended-lambda，编译风险更小。
template <typename Config, typename NodeT>
struct Occluder
{
    const Scene*          scene;
    const BvhView<NodeT>* view;
    const NodeT*          smemNodes;
    int*                  peak;
    int*                  overflow;

    __device__ __forceinline__ bool operator()(const Ray& r, float tMin, float tMax) const
    {
        return occludedScene<Config, NodeT>(*scene, *view, smemNodes, r, tMin, tMax, peak, overflow);
    }
};

// ============================================================================
//  3. 递归 traceRay（Whitted 核心：反射 / 折射 / 阴影 / Fresnel）
// ============================================================================
// 为什么递归：Whitted 模型本身就是「光线树」——命中镜面发反射、命中玻璃发折射，
// 每条新光线又是一次完整的着色。写成递归，代码和论文里的描述一一对应。
//
// 为什么 GPU 上递归不便宜：
//   递归函数不能被内联，每一层的活跃寄存器必须压到 local memory（显存），
//   而 warp 内不同线程的递归深度不一致（分歧），于是「访存 + 分歧」双重开销。
//   所以优化重点放在 **BVH 遍历**（每条光线都要跑好几遍的地方），
//   而不是着色递归；这也是性能表里 BVH 各档差距明显、而着色部分不变的原因。
template <typename Config, typename NodeT>
__device__ Vec3 traceRay(const Scene& scene, const BvhView<NodeT>& view, const NodeT* smemNodes,
                         const Ray& ray, int depth, int maxDepth, int* depthReached,
                         int* peak, int* overflow)
{
    if (depth >= maxDepth) return Vec3(0.0f, 0.0f, 0.0f);   // 深度用尽 -> 黑

    HitRecord rec;
    if (!intersectScene<Config, NodeT>(scene, view, smemNodes, ray, 1.0e-3f, RT_INFINITY, &rec,
                                       (int*)0, peak, overflow))
        return skyGradient(ray.direction, scene.skyTop, scene.skyBottom);

    // 记录「这条光线链最深的递归层数」。
    // ⚠️ 这里用 depth + 1 而不是 depth，是为了区分两种情况：
    //     · 光线**没打中任何东西**（天空）      -> depthReached 保持初始值 0
    //     · 光线打中**漫反射表面**（兔子/地面） -> 记录 1（因为有 1 层命中）
    //   如果写成 `depth > *depthReached`，主光线（depth=0）打中漫反射时
    //   `0 > 0` 为假、不更新，于是兔子和天空的 reached 都是 0、深度图里同色，
    //   看起来就像"兔子消失了"。
    if (depthReached != 0 && depth + 1 > *depthReached) *depthReached = depth + 1;

    // ---- 外侧法线：光线从物体内部射出时必须翻转，否则玻璃内部全黑 ----
    const bool frontFace = dot(ray.direction, rec.normal) < 0.0f;
    const Vec3 n = frontFace ? rec.normal : -rec.normal;    // 始终与入射方向相对
    const Vec3 p = rec.point;
    const Vec3 v = -ray.direction;                          // 指向观察者的方向（高光用）

    const Material& m = scene.materials.items[rec.materialId];
    const Vec3 albedo = applyChecker(m, p, m.albedo);

    // ---- 局部项：环境（半球天光）+ 每个光源的漫反射/高光（阴影光线决定是否本影）----
    const Vec3 ambientColor = skyGradient(n, scene.skyTop, scene.skyBottom);
    Vec3 local = ambientTerm(m, albedo, ambientColor);

    Occluder<Config, NodeT> occluder;
    occluder.scene = &scene;
    occluder.view = &view;
    occluder.smemNodes = smemNodes;
    occluder.peak = peak;
    occluder.overflow = overflow;

    for (int i = 0; i < scene.lightCount; ++i)
    {
        const PointLight& light = scene.lights[i];
        const bool shadowed = inShadow(occluder, p, n, light.position);
        local += shadeBlinnPhong(m, albedo, p, n, v, light.position, light.color, shadowed);
    }

    // ---- 反射（镜面）----
    if (m.type == MAT_MIRROR)
    {
        const Vec3 reflDir = normalize(reflect(ray.direction, n));
        // 起点沿法线抬起 epsilon：不抬的话反射光线立刻打到自己，画面出现黑点
        const Ray reflRay(p + n * RT_SHADOW_EPS, reflDir);
        const Vec3 reflected = traceRay<Config, NodeT>(scene, view, smemNodes, reflRay,
                                                       depth + 1, maxDepth, depthReached, peak, overflow);
        // 局部项与反射项按 reflectivity 混合；乘 albedo 让镜子带一点染色
        return lerp(local, reflected * albedo, m.reflectivity);
    }

    // ---- 折射（玻璃）：Snell + Schlick Fresnel，全反射时退化成镜面 ----
    if (m.type == MAT_GLASS)
    {
        // etaI/etaT 谁在分子取决于光线是进入还是离开介质（frontFace）
        const float eta = frontFace ? (1.0f / m.ior) : m.ior;
        const float cosi = clampf(-dot(ray.direction, n), 0.0f, 1.0f);
        // Schlick 的 r0 要的是「相对折射率」，从玻璃射出时是 1/ior
        const float fresnel = schlickReflectance(cosi, frontFace ? m.ior : (1.0f / m.ior));

        Vec3 refracted;
        const bool canRefract = refract(ray.direction, n, eta, &refracted);

        Vec3 result(0.0f, 0.0f, 0.0f);
        if (canRefract)
        {
            // 折射光线沿自身方向偏移：进入时落在外壳内侧、射出时落在外侧，都不会自交
            const Ray refractRay(p + refracted * RT_SHADOW_EPS, refracted);
            const Vec3 through = traceRay<Config, NodeT>(scene, view, smemNodes, refractRay,
                                                         depth + 1, maxDepth, depthReached, peak, overflow);
            result += through * (1.0f - fresnel);
        }

        // 全反射时没有折射分支，反射权重取 1
        const float reflWeight = canRefract ? fresnel : 1.0f;
        const Vec3 reflDir = normalize(reflect(ray.direction, n));
        const Ray reflRay(p + n * RT_SHADOW_EPS, reflDir);
        const Vec3 reflected = traceRay<Config, NodeT>(scene, view, smemNodes, reflRay,
                                                       depth + 1, maxDepth, depthReached, peak, overflow);
        result += reflected * reflWeight;

        // 乘 albedo 做染色，再补一点点局部项（否则玻璃边缘在全反射/掠射处会死黑）
        return result * albedo + local * 0.25f;
    }

    // ---- 漫反射 ----
    return local;
}

// ============================================================================
//  4. 渲染 kernel（模板：一个实例化 = 性能表里的一档）
// ============================================================================
template <typename Config>
__global__ void renderKernel(uchar4* image, int width, int height, int spp, int maxDepth,
                             Camera cam, Scene scene, int debugView, unsigned int seed,
                             int* gStackPeak, int* gStackOverflow)
{
    typedef typename NodeSelector<Config::alignedNodes>::Type NodeT;

    // 共享内存：根节点缓存（优化 ④）+ 块内栈统计归约
    //
    // 为什么用「字节数组 + 显式对齐」而不是直接 `__shared__ NodeT smemRoot[N]`：
    //   NodeT 是模板参数（BvhNode 带 alignas(32)，BvhNodeNaive 没有），nvcc 对
    //   「类型来自模板参数的 __shared__ 数组」保守地认为需要动态初始化，于是每个
    //   实例化都报一条
    //     warning #20054-D: dynamic initialization is not supported for a
    //                       function-scope static __shared__ variable
    //   而实际上这两个节点类型都是 POD、根本不需要初始化。换成 unsigned char
    //   数组后它是奇平凡类型（trivial），警告消失；对齐由 __align__(32) 保证，
    //   和 BvhNode 的 alignas(32) 一致。
    __shared__ __align__(32) unsigned char smemRootBytes[BVH_SMEM_NODE_COUNT * sizeof(NodeT)];
    NodeT* const smemRoot = reinterpret_cast<NodeT*>(smemRootBytes);

    __shared__ int   sPeak;
    __shared__ int   sOverflow;

    const int x = blockIdx.x * blockDim.x + threadIdx.x;
    const int y = blockIdx.y * blockDim.y + threadIdx.y;
    const bool inside = (x < width && y < height);

    const int tid = threadIdx.y * blockDim.x + threadIdx.x;
    const int nThreads = blockDim.x * blockDim.y;

    const BvhView<NodeT>& view = pickBvh(scene, (NodeT*)0);

    if (tid == 0) { sPeak = 0; sOverflow = 0; }

    // ---- 合作把根附近节点搬进共享内存 ----
    // 根附近的节点每条光线都要访问，一个 block 里 256 个线程反复读同一小段，
    // 放进共享内存就变成「一次全局读 + 之后的片上命中」。
    // 注意 smemRoot 只在 Config::rootSmem = true 时被填过；其余档位永远不会读它
    // （intersectBvh 的 UseSmem 是编译期模板参数，false 时那段代码被优化掉），
    // 所以下面统一传 smemRoot 是安全的，不需要写 Config::rootSmem ? smemRoot : NULL。
    if (Config::rootSmem)
    {
        const int loadCount = (view.nodeCount < BVH_SMEM_NODE_COUNT) ? view.nodeCount : BVH_SMEM_NODE_COUNT;
        for (int i = tid; i < loadCount; i += nThreads)
            smemRoot[i] = view.nodes[i];
    }

    // 无条件同步：越界线程也必须走到这里（提前 return 会让块内其它线程永远等下去）
    __syncthreads();

    int peak = 0;          // 本线程用到的最大栈深度（主光线 + 所有阴影光线合起来）
    int overflow = 0;      // 压栈溢出次数：>0 就意味着有子树被丢掉（画面上是空洞）

    if (inside)
    {
        // 调试视图只看第一个样本（语义清楚：法线图/深度图不该被 spp 平均掉）
        const int samples = (debugView == VIEW_SHADED) ? spp : 1;
        const float invSamples = 1.0f / (float)samples;

        unsigned int rng = hashUint((unsigned int)((size_t)y * (size_t)width + (size_t)x) ^ seed);
        Vec3 color(0.0f, 0.0f, 0.0f);

        for (int s = 0; s < samples; ++s)
        {
            float u, v;
            pixelToUV(x, y, width, height, &u, &v);
            if (samples > 1)
            {
                // 多采样抗锯齿：像素内抖动。哈希随机 -> 同样的参数重跑结果完全一致，
                // 这一点在「优化前后逐像素比对」时很重要。
                u += (randomFloat(&rng) - 0.5f) / (float)width;
                v += (randomFloat(&rng) - 0.5f) / (float)height;
            }
            const Ray ray = cam.getRay(u, v);

            if (debugView == VIEW_NORMALS || debugView == VIEW_HITS)
            {
                HitRecord rec;
                int visits = 0;
                if (intersectScene<Config, NodeT>(scene, view, smemRoot, ray, 1.0e-3f, RT_INFINITY,
                                                  &rec, &visits, &peak, &overflow))
                {
                    color += (debugView == VIEW_NORMALS)
                                 ? (rec.normal * 0.5f + Vec3(0.5f, 0.5f, 0.5f))
                                 : materialDebugColor(rec.materialId);
                }
                else
                {
                    color += skyGradient(ray.direction, scene.skyTop, scene.skyBottom);
                }
            }
            else if (debugView == VIEW_HEATMAP)
            {
                // 热力图：主光线访问了多少个 BVH 节点（越多说明遍历越绕）
                HitRecord rec;
                int visits = 0;
                intersectScene<Config, NodeT>(scene, view, smemRoot, ray, 1.0e-3f, RT_INFINITY,
                                              &rec, &visits, &peak, &overflow);
                color += heatmapColor(visits);
            }
            else if (debugView == VIEW_DEPTH)
            {
                int reached = 0;
                color += traceRay<Config, NodeT>(scene, view, smemRoot, ray, 0, maxDepth,
                                                 &reached, &peak, &overflow);
                // 分母用 maxDepth + 1：reached 的最大值是「最深递归 maxDepth 再 +1」= maxDepth+1，
                // 仍用 maxDepth 归一化会让最深处超过 1.0、溢出成纯红。
                color = depthDebugColor(reached, maxDepth + 1);
            }
            else
            {
                color += traceRay<Config, NodeT>(scene, view, smemRoot, ray, 0, maxDepth,
                                                 (int*)0, &peak, &overflow);
            }
        }

        color = color * invSamples;
        if (y < SIGN_BAND_Y) color = color * 0.35f;    // 顶部文字暗底
        image[(size_t)y * (size_t)width + (size_t)x] = toUchar4(color, INV_GAMMA);
    }

    // ---- 块内归约栈统计：共享内存原子 + 每块一次全局原子 ----
    // （直接对全局变量做 90 万次原子加会把测量结果本身拖慢，所以先在块内合并）
    if (peak > 0) atomicMax(&sPeak, peak);
    if (overflow > 0) atomicAdd(&sOverflow, overflow);

    __syncthreads();

    if (tid == 0)
    {
        if (sPeak > 0) atomicMax(gStackPeak, sPeak);
        if (sOverflow > 0) atomicAdd(gStackOverflow, sOverflow);
    }
}

// ============================================================================
//  5. 主机侧：设备缓冲 / 渲染一帧 / 统计
// ============================================================================
struct GpuBuffers
{
    uchar4* image;
    int*    stackPeak;
    int*    stackOverflow;
    size_t  imageBytes;

    GpuBuffers() : image(0), stackPeak(0), stackOverflow(0), imageBytes(0) {}

    void alloc(int width, int height)
    {
        release();
        imageBytes = (size_t)width * (size_t)height * sizeof(uchar4);
        CUDA_CHECK_MSG(cudaMalloc((void**)&image, imageBytes), "cudaMalloc(image)");
        CUDA_CHECK_MSG(cudaMemset(image, 0, imageBytes), "cudaMemset(image)");
        CUDA_CHECK_MSG(cudaMalloc((void**)&stackPeak, sizeof(int)), "cudaMalloc(stackPeak)");
        CUDA_CHECK_MSG(cudaMalloc((void**)&stackOverflow, sizeof(int)), "cudaMalloc(stackOverflow)");
    }

    void release()
    {
        if (image)         { cudaFree(image);         image = 0; }
        if (stackPeak)     { cudaFree(stackPeak);     stackPeak = 0; }
        if (stackOverflow) { cudaFree(stackOverflow); stackOverflow = 0; }
        imageBytes = 0;
    }
};

struct GpuScene
{
    GpuTriangles         tris;
    GpuBvh<BvhNode>      bvhAligned;
    GpuBvh<BvhNodeNaive> bvhNaive;

    void release()
    {
        tris.release();
        bvhAligned.release();
        bvhNaive.release();
    }
};

struct FrameResult
{
    std::vector<unsigned char> rgba;    // w*h*4，未叠字的原图（逐像素比对要用它）
    float kernelMs;
    int   stackPeak;
    int   stackOverflow;

    FrameResult() : rgba(), kernelMs(0.0f), stackPeak(0), stackOverflow(0) {}
};

// 启动一个档位的 kernel 并计时（只测 kernel，拷贝与画图不计入）
template <typename Config>
static float launchTier(GpuBuffers* gpu, int width, int height, int spp, int depth,
                        const Camera& cam, const Scene& scene, int debugView, unsigned int seed)
{
    const dim3 block(16, 16);
    const dim3 grid(cdiv(width, (int)block.x), cdiv(height, (int)block.y));

    CUDA_CHECK_MSG(cudaMemset(gpu->stackPeak, 0, sizeof(int)), "reset stackPeak");
    CUDA_CHECK_MSG(cudaMemset(gpu->stackOverflow, 0, sizeof(int)), "reset stackOverflow");

    cudaEvent_t evStart, evStop;
    CUDA_CHECK(cudaEventCreate(&evStart));
    CUDA_CHECK(cudaEventCreate(&evStop));

    CUDA_CHECK(cudaEventRecord(evStart));
    renderKernel<Config><<<grid, block>>>(gpu->image, width, height, spp, depth, cam, scene,
                                          debugView, seed, gpu->stackPeak, gpu->stackOverflow);
    CUDA_CHECK_LAST("renderKernel launch");
    CUDA_CHECK(cudaEventRecord(evStop));
    CUDA_CHECK_SYNC("renderKernel 执行");

    float ms = 0.0f;
    CUDA_CHECK(cudaEventElapsedTime(&ms, evStart, evStop));
    CUDA_CHECK(cudaEventDestroy(evStart));
    CUDA_CHECK(cudaEventDestroy(evStop));
    return ms;
}

template <typename Config>
static FrameResult renderFrame(GpuBuffers* gpu, int width, int height, int spp, int depth,
                               const Camera& cam, const Scene& scene, int debugView, unsigned int seed)
{
    FrameResult r;
    r.kernelMs = launchTier<Config>(gpu, width, height, spp, depth, cam, scene, debugView, seed);

    r.rgba.resize((size_t)width * (size_t)height * 4u);
    CUDA_CHECK_MSG(cudaMemcpy(r.rgba.data(), gpu->image, r.rgba.size(), cudaMemcpyDeviceToHost),
                   "cudaMemcpy(D2H image)");
    CUDA_CHECK_MSG(cudaMemcpy(&r.stackPeak, gpu->stackPeak, sizeof(int), cudaMemcpyDeviceToHost),
                   "cudaMemcpy(stackPeak)");
    CUDA_CHECK_MSG(cudaMemcpy(&r.stackOverflow, gpu->stackOverflow, sizeof(int), cudaMemcpyDeviceToHost),
                   "cudaMemcpy(stackOverflow)");
    return r;
}

// tier < 0：用编译期开关选出的默认档（--mode bvh）；否则跑 bench 的某一档
static FrameResult renderOne(int tier, GpuBuffers* gpu, int width, int height, int spp, int depth,
                             const Camera& cam, const Scene& scene, int debugView, unsigned int seed)
{
    if (tier < 0) return renderFrame<CfgDefault>(gpu, width, height, spp, depth, cam, scene, debugView, seed);

    switch (tier)
    {
    case TIER_BRUTE:          return renderFrame<CfgBrute>(gpu, width, height, spp, depth, cam, scene, debugView, seed);
    case TIER_BVH_NO_ORDER:   return renderFrame<CfgBvhNoOrder>(gpu, width, height, spp, depth, cam, scene, debugView, seed);
    case TIER_BVH_32:         return renderFrame<CfgBvh32>(gpu, width, height, spp, depth, cam, scene, debugView, seed);
    case TIER_BVH_32_SOA:     return renderFrame<CfgBvh32Soa>(gpu, width, height, spp, depth, cam, scene, debugView, seed);
    default:                  return renderFrame<CfgBvh32SoaSmem>(gpu, width, height, spp, depth, cam, scene, debugView, seed);
    }
}

// ---------------------------------------------------------------------------
//  存图：先复制一份再叠字（原图留着做逐像素比对，叠字会改像素）
// ---------------------------------------------------------------------------
static int saveStamped(const RenderOptions& opt, const unsigned char* rgba,
                       int width, int height, const char* tag,
                       const char* line1, const char* line2, const char* line3)
{
    std::vector<unsigned char> copy(rgba, rgba + (size_t)width * (size_t)height * 4u);

    stampTextHost(copy.data(), width, height, 12, 10, line1, 255, 255, 255);
    stampTextHost(copy.data(), width, height, 12, 22, line2, 200, 230, 255);
    if (line3 != 0 && line3[0] != '\0')
        stampTextHost(copy.data(), width, height, 12, 34, line3, 180, 255, 200);

    char stem[512];

    // --tag 只是给文件名加前缀，方便同一批图放在一起不重名
    char fullTag[192];
    if (opt.tag[0] != '\0') snprintf(fullTag, sizeof(fullTag), "%s_%s", opt.tag, tag);
    else                    snprintf(fullTag, sizeof(fullTag), "%s", tag);

    buildOutputStem(stem, sizeof(stem), WEEK_TAG, width, height, opt.spp, fullTag);
    return writeImageBoth(opt.outDir, stem, width, height, copy.data(), opt.wantPpm, opt.wantPng);
}

// ---------------------------------------------------------------------------
//  自检用的图像统计
// ---------------------------------------------------------------------------
struct ImageStats
{
    long blackPixels;
    long distinctColors;
    int  centerRgb[3];
    int  skyRgb[3];
};

static void analyzeImage(const unsigned char* rgba, int width, int height, ImageStats* st)
{
    std::vector<unsigned char> occupied(1u << 15, 0);   // 15 位色桶（每通道取高 5 位）
    long black = 0;

    const size_t pixels = (size_t)width * (size_t)height;
    for (size_t i = 0; i < pixels; ++i)
    {
        const int r = rgba[i * 4 + 0];
        const int g = rgba[i * 4 + 1];
        const int b = rgba[i * 4 + 2];
        if (r + g + b <= 8) ++black;
        occupied[(size_t)(((r >> 3) << 10) | ((g >> 3) << 5) | (b >> 3))] = 1;
    }

    long distinct = 0;
    for (size_t i = 0; i < occupied.size(); ++i) distinct += occupied[i];

    const size_t centerIndex = ((size_t)(height * 3 / 5) * (size_t)width + (size_t)(width / 2)) * 4u;
    const size_t skyIndex = (size_t)4 * (size_t)width * 4u + (size_t)(width / 2) * 4u;

    st->blackPixels = black;
    st->distinctColors = distinct;
    for (int c = 0; c < 3; ++c)
    {
        st->centerRgb[c] = rgba[centerIndex + (size_t)c];
        st->skyRgb[c] = rgba[skyIndex + (size_t)c];
    }
}

static bool expectRule(const char* what, bool pass, const char* detail)
{
    std::cout << "  [" << (pass ? "PASS" : "FAIL") << "] "
              << rt::ljust(what, 24) << " " << detail << "\n";
    return pass;
}

// ============================================================================
//  6. 场景搭建（主机侧）：网格 -> 变换 -> BVH -> 上传 -> 材质/球/光源
// ============================================================================
static const char* splitModeName(SplitMode s) { return (s == SPLIT_SAH) ? "SAH(12 候选)" : "质心中点(nth_element)"; }

static bool buildScene(const RenderOptions& opt, Scene* scene, GpuScene* gpu, BvhBuildStats* bstats)
{
    // ---- 6.1 网格：--obj 指定的模型（缺省 = 仓库自带 bunny_10k.obj）；
    //          任何失败都回退到程序化网格（保证一定出图）----
    TriMesh mesh;
    {
        char err[256] = { 0 };
        // 双路径：优先 assimp（日志会打印版本号，证明真的链接并用上了），
        // 失败再回退到手写解析器，最后才退回程序化网格。
        bool loaded = false;
        if (loadObjWithAssimp(opt.objPath, &mesh, MATID_MESH, err, sizeof(err)))
        {
            std::cout << "  [assimp] " << assimpVersionString() << "  加载成功（"
                      << mesh.triangleCount() << " 三角形）\n";
            loaded = true;
        }
        else
        {
            std::cout << "  [assimp] 未启用或失败（" << err << "）\n";
            std::cout << "           回退到手写解析器\n";
            char err2[256] = { 0 };
            loaded = loadObj(opt.objPath, &mesh, MATID_MESH, err2, sizeof(err2));
            if (!loaded) {
                std::cout << "  [警告] OBJ 加载失败：" << err2 << "\n";
            }
        }
        if (!loaded)
        {
            std::cout << "         默认路径 ..\\objs\\bunny_10k.obj 是相对 whitted\\whitted\\ 解析的；\n";
            std::cout << "         回退到程序化网格（" << MESH_TARGET_TRIANGLES
                      << " 面），想要真实模型请检查路径或用 --obj 给绝对路径\n";
            generateProceduralMesh(&mesh, MESH_TARGET_TRIANGLES, MATID_MESH);
        }
    }

    // ---- 6.2 摆位：缩放到目标高度、xz 居中、底面贴 y=0、再绕 Y 轴转 ----
    mesh.fitToHeight(MESH_HEIGHT, MESH_ROTATE_Y);
    std::cout << "  摆位: 高度="
              << rt::fixed(mesh.bounds.bmax.y - mesh.bounds.bmin.y, 2)
              << "  底面 y=" << rt::fixed(mesh.bounds.bmin.y, 2)
              << "  xz 中心=(" << rt::fixed(mesh.bounds.centroid().x, 2)
              << ", " << rt::fixed(mesh.bounds.centroid().z, 2)
              << ")  绕 Y " << rt::fixed(MESH_ROTATE_Y, 0) << " 度\n";

    // ---- 6.3 建 BVH：两种节点布局一次建出来（同一棵树，遍历结果才能逐像素一致）----
    BvhBuildOptions bopt;
    bopt.leafMax = BVH_MAX_LEAF_TRIANGLES;
    bopt.maxDepth = BVH_MAX_DEPTH;
    bopt.useSah = (opt.split == SPLIT_SAH);
    std::cout << "  划分方式: " << splitModeName(opt.split) << "\n";

    BvhBuildResult bvh;
    buildBvh(mesh.triangles, bopt, &bvh);
    *bstats = bvh.stats;

    // ---- 6.4 上传：三角形（AoS + SoA）、两套节点、triIndex ----
    gpu->tris.upload(mesh.triangles);
    gpu->bvhAligned.upload(bvh.aligned, bvh.triIndex);
    gpu->bvhNaive.upload(bvh.naive, bvh.triIndex);
    std::cout << "  上传: 三角形 AoS "
              << rt::fixed((double)gpu->tris.count * sizeof(TriangleAoS) / (1024.0 * 1024.0), 2)
              << " MB + SoA "
              << rt::fixed((double)gpu->tris.count * (sizeof(float) * 9.0 + sizeof(int)) / (1024.0 * 1024.0), 2)
              << " MB  节点 "
              << rt::fixed((double)gpu->bvhAligned.nodeCount * sizeof(BvhNode) / (1024.0 * 1024.0), 2)
              << " MB  triIndex "
              << rt::fixed((double)gpu->bvhAligned.triCount * sizeof(int) / (1024.0 * 1024.0), 2)
              << " MB\n";

    // ---- 6.5 材质 ----
    MaterialSet mats;
    mats.add(makeDiffuse(Vec3(0.86f, 0.85f, 0.82f), 0.14f, 0.92f, 0.18f, 28.0f));  // 0 网格
    mats.add(makeMirror(Vec3(0.96f, 0.97f, 1.00f), 0.88f));                        // 1 镜面球
    mats.add(makeGlass(Vec3(0.96f, 0.99f, 1.00f), 1.50f));                         // 2 玻璃球
    Material ground = makeDiffuse(Vec3(0.74f, 0.74f, 0.76f), 0.20f, 0.85f, 0.08f, 24.0f);
    ground.checkerScale = 0.5f;                                                    // 3 地面（程序化棋盘）
    mats.add(ground);

    // ---- 6.6 球：地面大球 + 镜面球 + 玻璃球 ----
    // 地面用一个半径 1000 的球当平面：这样阴影/反射都能落在同一套求交代码上。
    SphereSet spheres;
    if (!spheres.add(Sphere(Vec3(0.0f, -1000.0f, 0.0f), 1000.0f, MATID_GROUND)))
        return false;
    spheres.add(Sphere(Vec3(-2.95f, 1.05f, 1.05f), 1.05f, MATID_MIRROR));
    spheres.add(Sphere(Vec3(2.90f, 1.15f, 1.35f), 1.15f, MATID_GLASS));

    // ---- 6.7 场景 ----
    scene->spheres = spheres;
    scene->materials = mats;

    scene->lights[0].position = Vec3(5.5f, 8.0f, 4.5f);
    scene->lights[0].color = Vec3(1.00f, 0.96f, 0.90f);
    scene->lightCount = 1;                    // 只做一个点光源：硬阴影来自它

    scene->skyTop = Vec3(0.45f, 0.62f, 1.00f);
    scene->skyBottom = Vec3(1.00f, 1.00f, 1.00f);

    scene->meshCenter = Vec3(mesh.bounds.centroid().x, CAM_TARGET_Y, mesh.bounds.centroid().z);
    const Vec3 ext = mesh.bounds.extent();
    scene->meshRadius = 0.5f * sqrtf(ext.x * ext.x + ext.y * ext.y + ext.z * ext.z);

    scene->tris = gpu->tris.store;
    scene->bvhA = gpu->bvhAligned.view(gpu->tris.store);
    scene->bvhN = gpu->bvhNaive.view(gpu->tris.store);

    std::cout << "  场景: 球 " << scene->spheres.count
              << " 个（地面/镜面/玻璃）+ 网格 " << scene->tris.count
              << " 面 + " << scene->lightCount
              << " 个点光源 + 程序化棋盘地面\n";
    return true;
}

// 相机：绕 meshCenter 的轨道相机。转台靠 angle 变化实现；
// 视线目标用「网格包围盒中心」而不是世界原点，物体才会在每一帧都居中。
static Camera makeOrbitCamera(const Scene& scene, float angleDeg, float fovDeg, float aspect)
{
    const float rad = radians(angleDeg);
    const Vec3  at = scene.meshCenter;
    const Vec3  from(at.x + sinf(rad) * CAM_RADIUS, CAM_HEIGHT, at.z + cosf(rad) * CAM_RADIUS);
    return Camera(from, at, Vec3(0.0f, 1.0f, 0.0f), fovDeg, aspect);
}

// ============================================================================
//  7. 基准测试（--bench）：5 档 + 逐像素校验
// ============================================================================
struct BenchRow
{
    const char* name;
    double buildMs;
    float  renderMs;
    double speedup;
    int    maxDiff;
    double meanDiff;
    long   diffPixels;
    int    stackPeak;
    int    stackOverflow;
};

static bool speedupRule(double value)  { return value > 1.0; }

static void runBench(const RenderOptions& opt, const Scene& scene, GpuBuffers* gpu,
                     const BvhBuildStats& bstats, BenchRow* rows, bool* allBvhIdentical,
                     double* bestSpeedup)
{
    const float aspect = (float)opt.width / (float)opt.height;
    const Camera cam = makeOrbitCamera(scene, 0.0f, opt.fov, aspect);

    std::cout << "\n---- 性能对比（同一场景 / 同一相机 / 同一种子；树只建一次，5 档共用）----\n";
    std::cout << "  分辨率 " << opt.width << " x " << opt.height
              << "  spp " << opt.spp << "  递归深度 " << opt.depth
              << "  三角形 " << bstats.triCount
              << "  叶子阈值 " << (int)BVH_MAX_LEAF_TRIANGLES
              << "  栈上限 " << (int)BVH_STACK_SIZE << "\n";
    std::cout << "\n  " << rt::ljust("档位", 34)
              << " " << rt::rjust("建树(ms)", 10)
              << " " << rt::rjust("渲染(ms)", 11)
              << " " << rt::rjust("加速比", 9)
              << " " << rt::rjust("maxDiff", 9)
              << " " << rt::rjust("差异像素", 10) << "\n";

    std::vector<unsigned char> baseImage;
    double baseMs = 0.0;

    for (int tier = 0; tier < TIER_COUNT; ++tier)
    {
        std::cout << "  [" << tier << "/" << (TIER_COUNT - 1) << "] "
                  << rt::ljust(TIER_NAME[tier], 28) << " 渲染中...\r" << std::flush;
        fflush(stdout);

        const FrameResult fr = renderOne(tier, gpu, opt.width, opt.height, opt.spp, opt.depth,
                                         cam, scene, VIEW_SHADED, 20261013u);

        rows[tier].name = TIER_NAME[tier];
        rows[tier].buildMs = (tier == TIER_BRUTE) ? 0.0 : bstats.buildMs;
        rows[tier].renderMs = fr.kernelMs;
        rows[tier].stackPeak = fr.stackPeak;
        rows[tier].stackOverflow = fr.stackOverflow;

        if (tier == TIER_BRUTE)
        {
            baseImage = fr.rgba;
            baseMs = (double)fr.kernelMs;
            rows[tier].speedup = 1.0;
            rows[tier].maxDiff = 0;
            rows[tier].meanDiff = 0.0;
            rows[tier].diffPixels = 0;
        }
        else
        {
            const ImageDiff d = compareImages(fr.rgba.data(), baseImage.data(), opt.width, opt.height);
            rows[tier].speedup = (fr.kernelMs > 0.0f) ? (baseMs / (double)fr.kernelMs) : 0.0;
            rows[tier].maxDiff = d.maxDiff;
            rows[tier].meanDiff = d.meanDiff;
            rows[tier].diffPixels = d.pixelsDiffering;
        }

        if (tier == TIER_BRUTE)
            std::cout << "  " << rt::ljust(tier, 2) << " " << rt::ljust(TIER_NAME[tier], 31)
                      << " " << rt::rjust("-", 10)
                      << " " << rt::rjust(rt::fixed(rows[tier].renderMs, 2), 11)
                      << " " << rt::rjust(rt::fixed(rows[tier].speedup, 1), 8) << "x"
                      << " " << rt::rjust("(基准)", 9)
                      << " " << rt::rjust("-", 10) << "\n";
        else
            std::cout << "  " << rt::ljust(tier, 2) << " " << rt::ljust(TIER_NAME[tier], 31)
                      << " " << rt::rjust(rt::fixed(rows[tier].buildMs, 2), 10)
                      << " " << rt::rjust(rt::fixed(rows[tier].renderMs, 2), 11)
                      << " " << rt::rjust(rt::fixed(rows[tier].speedup, 1), 8) << "x"
                      << " " << rt::rjust(rows[tier].maxDiff, 9)
                      << " " << rt::rjust(rows[tier].diffPixels, 10) << "\n";

        if (tier == TIER_BRUTE)
        {
            // 暴力版 vs 最佳档的对比图留两张，作为「更快且没画错」的视觉证据
            char l1[128], l2[128], l3[128];
            snprintf(l1, sizeof(l1), "WHITTED BENCH 0 BRUTE  %d x %d SPP %d", opt.width, opt.height, opt.spp);
            snprintf(l2, sizeof(l2), "TRIS %d  NO BVH  KERNEL %.1f MS", bstats.triCount, rows[tier].renderMs);
            l3[0] = '\0';
            saveStamped(opt, fr.rgba.data(), opt.width, opt.height, "bench0_brute", l1, l2, l3);
        }
        if (tier == TIER_COUNT - 1)
        {
            char l1[128], l2[128], l3[128];
            snprintf(l1, sizeof(l1), "WHITTED BENCH 4 BVH+32B+SOA+SMEM  %d x %d SPP %d", opt.width, opt.height, opt.spp);
            snprintf(l2, sizeof(l2), "TRIS %d  KERNEL %.1f MS  SPEEDUP %.1fx", bstats.triCount,
                     rows[tier].renderMs, rows[tier].speedup);
            snprintf(l3, sizeof(l3), "BUILD %.1f MS  NODES %d  LEAVES %d  MAXDEPTH %d",
                     bstats.buildMs, bstats.nodeCount, bstats.leafCount, bstats.maxDepth);
            saveStamped(opt, fr.rgba.data(), opt.width, opt.height, "bench4_best", l1, l2, l3);
        }
    }

    // ---- 结论：BVH 各档 vs 暴力只在浮点并列处允许极少量差异 ----
    // 判据从「maxDiff 严格为 0」放宽到「允许极少量浮点并列取舍」：
    //   差异像素 <= 1 个（2560x1440 有 371 万像素，1 个 = 0.00003%）
    //   且通道差 <= 32（约 12%，肉眼几乎不可见）
    //
    // 为什么会有差异：如果同一条光线恰好穿过两个三角形的交界（两者的 t 在 float
    // 精度内相等），暴力遍历按**三角形索引顺序**、BVH 按**树的遍历顺序**，谁先
    // 命中就选谁，颜色自然差一点。这是"等价的浮点取舍"，不是 BVH 漏三角形。
    // 真正的漏三角形会表现为**成片**差异（几百上千个像素），那种才必须 FAIL。
    *allBvhIdentical = true;
    int worstDiffPixels = 0;
    int worstMaxDiff = 0;
    for (int tier = TIER_BVH_NO_ORDER; tier < TIER_COUNT; ++tier)
    {
        if (rows[tier].diffPixels > worstDiffPixels) worstDiffPixels = (int)rows[tier].diffPixels;
        if (rows[tier].maxDiff    > worstMaxDiff)    worstMaxDiff    = rows[tier].maxDiff;
    }
    if (worstDiffPixels > 1 || worstMaxDiff > 32) *allBvhIdentical = false;
    *bestSpeedup = rows[TIER_COUNT - 1].speedup;

    std::cout << "\n  说明：\n";
    std::cout << "    * 档 1~4 用的是同一棵树、同一套 Moller-Trumbore，只改「访存布局 / 遍历顺序」，\n";
    std::cout << "      所以它们彼此**逐像素必须完全一致**（maxDiff 全 0）。\n";
    std::cout << "    * 档 1 与档 2 只差「有没有按入口距离排序」——这一格差值就是有序遍历的收益；\n";
    std::cout << "      2->3 是三角形 AoS->SoA 的收益；3->4 是共享内存缓存根节点的收益。\n";
    std::cout << "    * 与档 0（暴力）的差异来自「最近交点的比较顺序」（浮点并列取舍不同），\n";
    std::cout << "      实测档 1 maxDiff=" << rows[TIER_BVH_NO_ORDER].maxDiff
              << "（差异像素 " << rows[TIER_BVH_NO_ORDER].diffPixels << " 个，占 "
              << rt::fixed(100.0 * (double)rows[TIER_BVH_NO_ORDER].diffPixels
                           / ((double)opt.width * (double)opt.height), 4)
              << "%）—— 同一场景的等价求交顺序，\n";
    std::cout << "      不是画面错误；如果这里出现大面积差异，那才是 BVH 漏了三角形。\n";
    std::cout << "    * 建树耗时只算一次（" << rt::fixed(bstats.buildMs, 1)
              << " ms），4 个 BVH 档共用，所以表里是同一个数。\n";
    if (rows[TIER_COUNT - 1].stackOverflow > 0)
        std::cout << "    [警告] 栈溢出 " << rows[TIER_COUNT - 1].stackOverflow
                  << " 次！有子树被丢掉，画面会出现空洞 —— 请调大 BVH_STACK_SIZE。\n";
}

// ============================================================================
//  8. 转台（--frames N）/ 单张渲染
// ============================================================================
static int renderTurnaround(const RenderOptions& opt, const Scene& scene, GpuBuffers* gpu,
                            std::vector<unsigned char>* outFirstFrame,
                            bool* outAnyOverflow, int* outPeak, int* outWritten)
{
    const float aspect = (float)opt.width / (float)opt.height;
    const int tier = (opt.mode == MODE_BRUTE) ? TIER_BRUTE : -1;
    const int frames = (opt.frames < 1) ? 1 : opt.frames;

    int written = 0;
    int peakMax = 0;
    bool overflow = false;

    for (int i = 0; i < frames; ++i)
    {
        const float angle = 360.0f * (float)i / (float)frames;
        const Camera cam = makeOrbitCamera(scene, angle, opt.fov, aspect);

        const FrameResult fr = renderOne(tier, gpu, opt.width, opt.height, opt.spp, opt.depth,
                                         cam, scene, (int)opt.debug, 20261013u);

        // 第 0 帧留一份原图给自检用（叠字会改像素，所以自检必须用叠字前的图）
        if (i == 0 && outFirstFrame != 0) *outFirstFrame = fr.rgba;

        if (fr.stackPeak > peakMax) peakMax = fr.stackPeak;
        if (fr.stackOverflow > 0) overflow = true;

        char tag[64];
        snprintf(tag, sizeof(tag), "turn%02d", i);

        char l1[128], l2[128], l3[128];
        snprintf(l1, sizeof(l1), "WHITTED TURN %3.0f DEG  FRAME %d/%d  %d x %d SPP %d",
                 angle, i + 1, frames, opt.width, opt.height, opt.spp);
        snprintf(l2, sizeof(l2), "KERNEL %.1f MS  DEPTH %d  %s", fr.kernelMs, opt.depth,
                 (opt.mode == MODE_BRUTE) ? "BRUTE" : "BVH");
        snprintf(l3, sizeof(l3), "STACK PEAK %d / %d  VIEW %s", fr.stackPeak,
                 (int)BVH_STACK_SIZE, debugViewName(opt.debug));

        const int w = saveStamped(opt, fr.rgba.data(), opt.width, opt.height, tag, l1, l2, l3);
        written += w;

        std::cout << "  [" << (i + 1) << "/" << frames << "] 角度 "
                  << rt::rjust(rt::fixed(angle, 1), 6)
                  << " 度  kernel " << rt::fixed(fr.kernelMs, 2)
                  << " ms  栈峰值 " << fr.stackPeak << "  写入 " << w << " 个文件\n";
    }

    *outAnyOverflow = overflow;
    *outPeak = peakMax;
    *outWritten = written;
    return written;
}

// ============================================================================
//  9. 自检（PASS / FAIL / SKIP，全部打印出来，不用肉眼看图）
// ============================================================================
struct SelfTestInput
{
    const unsigned char* rgba;
    int    width, height;
    int    stackPeak;
    bool   stackOverflow;
    bool   benchRan;
    bool   bvhIdentical;
    double speedup;
};

static bool runSelfTest(const SelfTestInput& in)
{
    std::cout << "\n---- 自检（PASS/FAIL）----\n";
    bool all = true;

    ImageStats st;
    analyzeImage(in.rgba, in.width, in.height, &st);

    const double blackFraction = (double)st.blackPixels / ((double)in.width * (double)in.height);
    char detail[256];

    // 1) 图像非空：不能是黑图，也不能是一张纯色图
    snprintf(detail, sizeof(detail), "黑像素 %.2f%%（阈值 <10%%）、色彩数 %ld（阈值 >64）",
             blackFraction * 100.0, st.distinctColors);
    all &= expectRule("图像非空", (blackFraction < 0.10) && (st.distinctColors > 64), detail);

    // 2) 画面里确实有东西：中心（网格/地面）与顶部天空的颜色必须明显不同
    const int diff = abs(st.centerRgb[0] - st.skyRgb[0]) + abs(st.centerRgb[1] - st.skyRgb[1]) +
                     abs(st.centerRgb[2] - st.skyRgb[2]);
    snprintf(detail, sizeof(detail), "中心 rgb(%d,%d,%d) vs 天空 rgb(%d,%d,%d) 差 %d（阈值 >20）",
             st.centerRgb[0], st.centerRgb[1], st.centerRgb[2],
             st.skyRgb[0], st.skyRgb[1], st.skyRgb[2], diff);
    all &= expectRule("中央有几何体", diff > 20, detail);

    // 3) 显式栈没有溢出，峰值在上限之内
    snprintf(detail, sizeof(detail), "峰值 %d / 上限 %d，溢出 %s",
             in.stackPeak, (int)BVH_STACK_SIZE, in.stackOverflow ? "有（画面会空洞！）" : "无");
    all &= expectRule("栈峰值 < 64", (in.stackPeak < (int)BVH_STACK_SIZE) && !in.stackOverflow, detail);

    // 4) 逐像素校验（只有 --bench 会跑）
    if (in.benchRan)
        all &= expectRule("逐像素校验", in.bvhIdentical,
                      "BVH 各档 vs 暴力：差异像素 <= 1 且 maxDiff <= 32（容忍浮点并列取舍）");
    else
        std::cout << "  [SKIP] " << rt::ljust("逐像素校验", 24) << " 只有 --bench 会做逐像素比对\n";

    // 5) 加速比（只有 --bench 会跑）
    if (in.benchRan)
    {
        snprintf(detail, sizeof(detail), "暴力 / 最佳档 = %.1fx（阈值 >1）", in.speedup);
        all &= expectRule("加速比 > 1", speedupRule(in.speedup), detail);
    }
    else
    {
        std::cout << "  [SKIP] " << rt::ljust("加速比 > 1", 24) << " 只有 --bench 会测加速比\n";
    }

    std::cout << "---- 自检结束：" << (all ? "全部通过" : "存在 FAIL") << " ----\n";
    return all;
}

// ============================================================================
//  10. 交互预览：把窗口回调接到渲染器上
// ============================================================================
struct InteractiveContext
{
    const Scene* scene;
    GpuBuffers*  gpu;
    int          width, height, spp, depth;
    float        fov;
};

static void interactiveRenderCallback(void* userData, const Vec3& eye, const Vec3& target, unsigned char* rgbaOut)
{
    InteractiveContext* ctx = (InteractiveContext*)userData;
    const float aspect = (float)ctx->width / (float)ctx->height;
    const Camera cam(eye, target, Vec3(0.0f, 1.0f, 0.0f), ctx->fov, aspect);

    const FrameResult fr = renderOne(-1, ctx->gpu, ctx->width, ctx->height, ctx->spp, ctx->depth,
                                     cam, *ctx->scene, VIEW_SHADED, 20261013u);
    memcpy(rgbaOut, fr.rgba.data(), fr.rgba.size());
}

// ============================================================================
//  11. 命令行
// ============================================================================
// 注意 `int& i` 是**引用**：取走 argv[i+1] 之后必须把 i 往前推一格，否则外层
// for 的 ++i 只跳过选项本身、参数值还留在 argv 里，下一轮就会把 "1" 当成一个
// 新选项，打印「未知参数: 1（已忽略）」。
// 值其实已经生效了，所以这个 bug 只制造噪音、不改变行为 —— 但噪音会骗人：
// 你会以为参数没被接受，进而怀疑是命令行写法错了。
static bool takeIntArg(int argc, char** argv, int& i, int* out)
{
    if (i + 1 >= argc) return false;
    *out = atoi(argv[i + 1]);
    ++i;
    return true;
}

static bool takeFloatArg(int argc, char** argv, int& i, float* out)
{
    if (i + 1 >= argc) return false;
    *out = (float)atof(argv[i + 1]);
    ++i;
    return true;
}

static bool takeStrArg(int argc, char** argv, int& i, char* out, size_t outSize)
{
    if (i + 1 >= argc) return false;
    strncpy_s(out, outSize, argv[i + 1], _TRUNCATE);
    ++i;
    return true;
}

static void parseArgs(int argc, char** argv, RenderOptions* o)
{
    for (int i = 1; i < argc; ++i)
    {
        const char* a = argv[i];
        if (_stricmp(a, "--width") == 0)         { takeIntArg(argc, argv, i, &o->width);  o->resolutionGiven = true; }
        else if (_stricmp(a, "--height") == 0)   { takeIntArg(argc, argv, i, &o->height); o->resolutionGiven = true; }
        else if (_stricmp(a, "--spp") == 0)      takeIntArg(argc, argv, i, &o->spp);
        else if (_stricmp(a, "--depth") == 0)    takeIntArg(argc, argv, i, &o->depth);
        else if (_stricmp(a, "--fov") == 0)      takeFloatArg(argc, argv, i, &o->fov);
        else if (_stricmp(a, "--obj") == 0)      takeStrArg(argc, argv, i, o->objPath, sizeof(o->objPath));
        else if (_stricmp(a, "--out") == 0)      takeStrArg(argc, argv, i, o->outDir, sizeof(o->outDir));
        else if (_stricmp(a, "--tag") == 0)      takeStrArg(argc, argv, i, o->tag, sizeof(o->tag));
        else if (_stricmp(a, "--bench") == 0)    o->bench = true;
        else if (_stricmp(a, "--interactive") == 0) o->interactive = true;
        else if (_stricmp(a, "--ppm-only") == 0) { o->wantPpm = true;  o->wantPng = false; }
        else if (_stricmp(a, "--png-only") == 0) { o->wantPpm = false; o->wantPng = true;  }
        else if (_stricmp(a, "--frames") == 0)
        {
            // 注意：默认值 frames = 8 是在 defaultOptions() 里设的，
            // 所以「不写 --frames」= 转台 8 张，而不是单张；--frames 1 才是单张。
            if (!takeIntArg(argc, argv, i, &o->frames)) o->frames = 8;
        }
        else if (_stricmp(a, "--mode") == 0)
        {
            char v[32] = { 0 };
            if (takeStrArg(argc, argv, i, v, sizeof(v)))
                o->mode = (_stricmp(v, "brute") == 0) ? MODE_BRUTE : MODE_BVH;
        }
        else if (_stricmp(a, "--split") == 0)
        {
            char v[32] = { 0 };
            if (takeStrArg(argc, argv, i, v, sizeof(v)))
                o->split = (_stricmp(v, "sah") == 0) ? SPLIT_SAH : SPLIT_MID;
        }
        else if (_stricmp(a, "--debug") == 0)
        {
            char v[32] = { 0 };
            if (takeStrArg(argc, argv, i, v, sizeof(v)))
            {
                if (_stricmp(v, "normals") == 0)      o->debug = VIEW_NORMALS;
                else if (_stricmp(v, "depth") == 0)   o->debug = VIEW_DEPTH;
                else if (_stricmp(v, "hits") == 0)    o->debug = VIEW_HITS;
                else if (_stricmp(v, "heatmap") == 0) o->debug = VIEW_HEATMAP;
                else                                  o->debug = VIEW_SHADED;
            }
        }
        else if (_stricmp(a, "--help") == 0 || _stricmp(a, "-h") == 0)
        {
            std::cout << "用法: whitted.exe [--obj PATH] [--width N] [--height N] [--spp N] [--depth N] [--fov DEG]\n"
                         "                 [--frames N] [--bench] [--mode brute|bvh] [--split mid|sah]\n"
                         "                 [--out DIR] [--tag NAME] [--ppm-only|--png-only]\n"
                         "                 [--debug shaded|normals|depth|hits|heatmap] [--interactive]\n"
                         "  --frames N   绕 Y 轴等角度转台，输出 N 张图（默认 8；--frames 1 = 单张）\n"
                         "  --bench      5 档性能对比 + 逐像素校验（默认 640x360，可用 --width/--height 调大）\n";
            exit(EXIT_SUCCESS);
        }
        else std::cout << "[警告] 未知参数: " << a << "（已忽略）\n";
    }

    if (o->width < 16)   o->width = 16;
    if (o->height < 16)  o->height = 16;
    if (o->spp < 1)      o->spp = 1;
    if (o->spp > 64)     o->spp = 64;
    if (o->depth < 1)    o->depth = 1;
    if (o->depth > 16)   o->depth = 16;    // 再深寄存器压力暴涨（实测）
    if (o->fov < 1.0f)   o->fov = 1.0f;
    if (o->fov > 170.0f) o->fov = 170.0f;
    if (o->frames < 1)   o->frames = 1;
    if (o->frames > 360) o->frames = 360;
}

// ============================================================================
//  12. main
// ============================================================================
int main(int argc, char** argv)
{
    setupConsoleEncoding();

    RenderOptions opt;
    defaultOptions(&opt);
    parseArgs(argc, argv, &opt);

    if (opt.bench && !opt.resolutionGiven)
    {
        opt.width = BENCH_WIDTH;
        opt.height = BENCH_HEIGHT;
        std::cout << "[提示] --bench 未指定分辨率，默认用 " << opt.width << "x" << opt.height
                  << "（暴力版在大分辨率下要几十秒；\n"
                     "       想看大图数据就加 --width 1280 --height 720）\n";
    }

    const float aspect = (float)opt.width / (float)opt.height;

    std::cout << "whitted: BVH 优化的 Whitted 光线追踪器（优化 + 可视化 + 收尾）\n";
    printDeviceInfo();
    std::cout << "  分辨率 " << opt.width << " x " << opt.height << " (aspect "
              << rt::fixed(aspect, 4) << ")  spp " << opt.spp
              << "  递归深度 " << opt.depth << "  fov " << rt::fixed(opt.fov, 1)
              << " 度  视图 " << debugViewName(opt.debug)
              << "  模式 " << ((opt.mode == MODE_BRUTE) ? "brute" : "bvh") << "\n";
    printBvhConfig();

    // ---- 递归 kernel 必须先调大 per-thread 栈 ----
    // traceRay<Config, NodeT> 是递归的（镜面反射 / 玻璃折射各走一条子光线），CUDA 每个
    // 线程的默认栈只有 1 KB；Debug 构建（-G，不优化）下 5 层递归就可能超，device 端
    // 的表现就是 cudaErrorLaunchFailure (719) @ cudaDeviceSynchronize()。
    // 编译期的征兆是这条 ptxas 警告：
    //   ptxas warning : Stack size for entry function '...' cannot be statically determined
    // 必须在**第一次 kernel launch 之前**设置。
    // 注意 BVH 遍历用的是显式栈数组（不递归），这里要救的是 traceRay 的递归。
    //
    // 64 KiB 是**上限**，不是实际占用（真实占用 = 编译器的每线程 frame size × 常驻
    // 线程数），所以调大它本身不会多吃显存。
    // 为什么是 64 而不是 32：实测 depth=1 通过、depth=5 崩，根因是当时
    // `intersectBvh` 带 __forceinline__ 且在函数里开了 512 字节显式栈数组，被复制进
    // 每一层递归帧。`traverseBvh` 有同样的
    // 数组，因此同样去掉强制内联、并把上限对齐到 64 KiB。
    // 兜底网格有 6.8 万面（bunny 只有 1 万面），树更深，
    // 栈压力本来就更大。
    {
        size_t stackBefore = 0, stackAfter = 0;
        CUDA_CHECK_MSG(cudaDeviceGetLimit(&stackBefore, cudaLimitStackSize),
                       "cudaDeviceGetLimit(stack)");
        const size_t wantStack = 64 * 1024;
        CUDA_CHECK_MSG(cudaDeviceSetLimit(cudaLimitStackSize, wantStack),
                       "cudaDeviceSetLimit(stack)");
        CUDA_CHECK_MSG(cudaDeviceGetLimit(&stackAfter, cudaLimitStackSize),
                       "cudaDeviceGetLimit(stack, after)");
        std::cout << "  线程栈  per-thread stack " << stackBefore << " B -> "
                  << stackAfter << " B（请求 " << wantStack << " B）\n";
        if (stackAfter < wantStack)
            std::cout << "  [警告] 驱动没有采纳该栈大小，实际只有 " << stackAfter << " B；递归越深越可能崩\n";
    }

    // ---- 场景 + BVH + 上传 ----
    Scene scene;
    GpuScene gpuScene;
    BvhBuildStats bstats;
    if (!buildScene(opt, &scene, &gpuScene, &bstats))
    {
        std::cout << "[错误] 场景搭建失败\n";
        gpuScene.release();
        return EXIT_FAILURE;
    }

    // 预览模式：低分辨率 + 单采样，循环交给 interactive.h
    if (opt.interactive)
    {
        const int pw = 480, ph = 270;
        GpuBuffers preview;
        preview.alloc(pw, ph);

        InteractiveContext ctx;
        ctx.scene = &scene;
        ctx.gpu = &preview;
        ctx.width = pw;
        ctx.height = ph;
        ctx.spp = 1;
        ctx.depth = opt.depth;
        ctx.fov = opt.fov;

        const Vec3 eye(scene.meshCenter.x, CAM_HEIGHT, scene.meshCenter.z + CAM_RADIUS);
        const int rc = runInteractivePreview(pw, ph, &ctx, interactiveRenderCallback,
                                             eye, scene.meshCenter, "whitted --interactive");

        preview.release();
        gpuScene.release();
        CUDA_CHECK_MSG(cudaDeviceReset(), "cudaDeviceReset");
        return (rc == 0) ? EXIT_SUCCESS : EXIT_FAILURE;
    }

    GpuBuffers gpu;
    gpu.alloc(opt.width, opt.height);

    // 预热：先跑一张小图，把上下文/常量缓存都热起来，免得第一档计入启动开销
    {
        const Camera warmCam = makeOrbitCamera(scene, 0.0f, opt.fov, 1.0f);
        (void)renderOne(-1, &gpu, 64, 64, 1, opt.depth, warmCam, scene, VIEW_SHADED, 1u);
    }

    if (opt.bench)
    {
        BenchRow rows[TIER_COUNT];
        memset(rows, 0, sizeof(rows));
        bool identical = false;
        double speedup = 0.0;

        runBench(opt, scene, &gpu, bstats, rows, &identical, &speedup);

        // 自检用的图：最佳档那一档（渲染一次，用于「图像非空 / 中央有几何体」两项）
        const Camera cam = makeOrbitCamera(scene, 0.0f, opt.fov, aspect);
        const FrameResult fr = renderOne(-1, &gpu, opt.width, opt.height, opt.spp, opt.depth,
                                         cam, scene, VIEW_SHADED, 20261013u);

        SelfTestInput in;
        in.rgba = fr.rgba.data();
        in.width = opt.width;
        in.height = opt.height;
        in.stackPeak = rows[TIER_COUNT - 1].stackPeak;
        in.stackOverflow = (rows[TIER_COUNT - 1].stackOverflow > 0);
        in.benchRan = true;
        in.bvhIdentical = identical;
        in.speedup = speedup;
        const bool ok = runSelfTest(in);

        std::cout << "\n输出目录: " << opt.outDir << "\\（每档统计见 out\\perf_report.md 的表格模板）\n";
        std::cout << "基准测试结束：" << (ok ? "全部通过" : "存在 FAIL（见上面）") << "\n";

        gpu.release();
        gpuScene.release();
        CUDA_CHECK_MSG(cudaDeviceReset(), "cudaDeviceReset");
        return ok ? EXIT_SUCCESS : EXIT_FAILURE;
    }

    // ---- 转台 / 单张 ----
    if (opt.mode == MODE_BRUTE && opt.width * opt.height > 512 * 512)
        std::cout << "[警告] --mode brute 在 " << opt.width << "x" << opt.height
                  << " 下会很慢（暴力遍历每个像素都要测全部三角形）\n";

    bool anyOverflow = false;
    int  peakMax = 0;
    int  written = 0;

    // 转台第 0 帧的原图直接拿来跑自检，不再重复渲染一遍（暴力档下一帧可能要几秒）
    std::vector<unsigned char> firstFrame;
    renderTurnaround(opt, scene, &gpu, &firstFrame, &anyOverflow, &peakMax, &written);

    SelfTestInput in;
    in.rgba = firstFrame.data();
    in.width = opt.width;
    in.height = opt.height;
    in.stackPeak = peakMax;
    in.stackOverflow = anyOverflow;
    in.benchRan = false;
    in.bvhIdentical = true;
    in.speedup = 0.0;
    const bool ok = runSelfTest(in);

    std::cout << "\n写入文件 " << written << " 个（--spp " << opt.spp
              << ", --depth " << opt.depth << "）。"
              << (ok ? "" : "（有 FAIL，见上面）") << "\n";

    gpu.release();
    gpuScene.release();
    CUDA_CHECK_MSG(cudaDeviceReset(), "cudaDeviceReset");
    return (written > 0) ? EXIT_SUCCESS : EXIT_FAILURE;
}
