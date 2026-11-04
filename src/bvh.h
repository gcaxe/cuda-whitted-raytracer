// ============================================================================
//  bvh.h  --  AABB / slab 求交 + CPU 建树 + GPU 显式栈遍历（的主角）
// ----------------------------------------------------------------------------
//  本文件实现四项优化，每一件都对应性能表里的一行：
//
//    ① 32 字节对齐的节点布局（BvhNode）      —— 一个节点正好占一个 32B sector
//    ② 三角形 SoA 布局（BVH_USE_SOA）        —— warp 内取三角形的地址更连续
//    ③ 有序遍历 + best 剪枝                  —— 近的子节点先走，远的压栈
//    ④ 共享内存缓存根节点（BVH_CACHE_ROOT_IN_SMEM）—— 根附近节点每个 warp 都要读
//
//  为什么「显式栈」是 GPU 上的标准做法：
//    朴素递归遍历要靠函数调用栈，而设备函数的栈通常在 local memory（显存）里，
//    每层递归都要保存/恢复寄存器，warp 内线程的遍历深度还不一致（分歧），
//    代价极高。改成「一个固定大小的 int 栈 + 循环」之后，栈就在寄存器/本地数组
//    里，编译器能把大部分访问优化掉，而且深度上限是编译期已知的。
//
//  栈溢出必须计数：溢出时我们会「丢掉一个还没访问的子树」，画面上就是空洞
//  （漏三角形）。宁可打印告警也不能静默丢，所以每次溢出都累加计数，
//  kernel 把计数和「栈使用峰值」一起拷回主机打印。
// ============================================================================
#pragma once

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <vector>
#include <algorithm>
#include <chrono>

#include <cuda_runtime.h>   // 本文件同时包含主机侧的设备内存管理

#include "vec3.h"
#include "rt_math.h"
#include "cuda_check.h"
#include "triangle.h"

// ---------------------------------------------------------------------------
//  0. 编译期优化开关（改这里的宏可以关掉某一项优化做对比；--bench 会逐档实测）
// ---------------------------------------------------------------------------
#ifndef BVH_USE_SOA
#define BVH_USE_SOA 1                 // 1 = 三角形用 SoA 访存
#endif
#ifndef BVH_CACHE_ROOT_IN_SMEM
#define BVH_CACHE_ROOT_IN_SMEM 1      // 1 = 用 __shared__ 缓存根附近节点
#endif

#define BVH_SMEM_NODE_COUNT     16    // 缓存多少个根节点：16×32B = 512B，
                                      // 一个 warp 一次读满一整行，命中率最高的那批节点
#define BVH_MAX_LEAF_TRIANGLES  4     // 叶子阈值：4 个三角形
#define BVH_MAX_DEPTH           64    // 最大深度：二叉树 + 4 叶阈值，7 万面只要 ~15 层
#define BVH_STACK_SIZE          64    // 显式栈容量（够了：每层最多压 1 个节点）
#define BVH_SAH_CANDIDATES      12    // SAH 候选划分个数：3 轴 × 4 位置

// 运行时可打印的开关状态（一眼看清各优化开了哪些）。
inline void printBvhConfig()
{
    printf("  BVH 编译期开关: SOA=%d  SMEM_ROOT=%d(%d 节点)  叶子阈值=%d  最大深度=%d  栈=%d  SAH 候选=%d\n",
           (int)BVH_USE_SOA, (int)BVH_CACHE_ROOT_IN_SMEM, (int)BVH_SMEM_NODE_COUNT,
           (int)BVH_MAX_LEAF_TRIANGLES, (int)BVH_MAX_DEPTH, (int)BVH_STACK_SIZE,
           (int)BVH_SAH_CANDIDATES);
}

// ============================================================================
//  1. AABB + slab 求交
// ============================================================================
struct AABB
{
    Vec3 bmin, bmax;

    // 默认构造出「空盒子」：min = +inf，max = -inf，grow 任何点都能撑开。
    RT_HD AABB() : bmin(RT_INFINITY, RT_INFINITY, RT_INFINITY),
                   bmax(-RT_INFINITY, -RT_INFINITY, -RT_INFINITY) {}
    RT_HD AABB(const Vec3& lo, const Vec3& hi) : bmin(lo), bmax(hi) {}

    RT_HD void grow(const Vec3& p) { bmin = minVec(bmin, p); bmax = maxVec(bmax, p); }
    RT_HD void grow(const AABB& b) { bmin = minVec(bmin, b.bmin); bmax = maxVec(bmax, b.bmax); }
    RT_HD void pad(float e) { bmin -= Vec3(e, e, e); bmax += Vec3(e, e, e); }

    RT_HD Vec3  centroid() const { return (bmin + bmax) * 0.5f; }
    RT_HD Vec3  extent() const { return bmax - bmin; }

    // 最长轴：中点分割和 SAH 都优先切最长的那个方向，树才不会又高又瘦。
    RT_HD int longestAxis() const
    {
        const Vec3 e = extent();
        if (e.x > e.y) return (e.x > e.z) ? 0 : 2;
        return (e.y > e.z) ? 1 : 2;
    }

    // 表面积启发式（SAH）里的「表面积」：2*(dx*dy + dy*dz + dz*dx)。
    // 负数（空盒子）夹到 0，避免代价变成负数把划分带偏。
    RT_HD float area() const
    {
        const Vec3 d = maxVec(extent(), Vec3(0.0f, 0.0f, 0.0f));
        return 2.0f * (d.x * d.y + d.y * d.z + d.z * d.x);
    }
};

// 按轴取下标的两个小工具（Vec3 没有 operator[]，显式写出来更清楚）。
RT_HD RT_INLINE float axisOf(const Vec3& v, int axis)
{
    return axis == 0 ? v.x : (axis == 1 ? v.y : v.z);
}

RT_HD RT_INLINE void setAxisOf(Vec3* v, int axis, float value)
{
    if (axis == 0) v->x = value; else if (axis == 1) v->y = value; else v->z = value;
}

// 预求倒数方向：slab 求交的每一次盒子测试都要用，光线只算一次。
RT_HD RT_INLINE Vec3 safeInvDir(const Vec3& d)
{
    // 1/0 = ±inf 是 IEEE-754 的规定。故意**不**特判除零：
    // 平行于某个轴时，(bmin-o)*inf 得到 ∓inf、(bmax-o)*inf 得到 ±inf，
    // 三对板求交之后区间要么全空（光线在板外）要么不受影响（在板内），
    // 结果自动正确。写 if (d.x == 0.0f) 反而要多一次分支，还可能漏掉
    // 「方向分量是 -0.0f」这种边界。
    return Vec3(1.0f / d.x, 1.0f / d.y, 1.0f / d.z);
}

// ---------------------------------------------------------------------------
//  slab method：光线与 AABB 求交
// ---------------------------------------------------------------------------
// 把盒子看成三对平行板（x/y/z 各一对），光线与每对板交出区间 [t0, t1]，
// 三个区间求交集就是与盒子的重叠区间 [near, far]：
//
//     near = max(三对的较小值, tMin)      far = min(三对的较大值, tMax)
//     命中 <=> near <= far
//
// 返回 true 时 *tEntry = near（进入盒子的距离）。有序遍历要靠这个值决定
// 「先走哪个孩子」，所以这里多输出一个 float，几乎没有额外代价。
//
// 关于 NaN：当光线原点**正好**落在板面上（bmin-o == 0）且方向分量为 0 时，
// 0 * inf 会得到 NaN。这里用 fminf/fmaxf 是刻意的 —— IEEE 的 minNum/maxNum
// 语义会忽略 NaN 操作数，NaN 不会污染结果，所以不需要额外判断。
RT_HD RT_INLINE bool hitAabb(const AABB& b, const Vec3& o, const Vec3& invD,
                             float tMin, float tMax, float* tEntry)
{
    float t0 = (b.bmin.x - o.x) * invD.x;
    float t1 = (b.bmax.x - o.x) * invD.x;
    float near = fminf(t0, t1);
    float far  = fmaxf(t0, t1);

    t0 = (b.bmin.y - o.y) * invD.y;
    t1 = (b.bmax.y - o.y) * invD.y;
    near = fmaxf(near, fminf(t0, t1));
    far  = fminf(far,  fmaxf(t0, t1));

    t0 = (b.bmin.z - o.z) * invD.z;
    t1 = (b.bmax.z - o.z) * invD.z;
    near = fmaxf(near, fminf(t0, t1));
    far  = fminf(far,  fmaxf(t0, t1));

    // 与光线自身的有效区间求交：tMin 把「盒子在身后」的情况直接剔除。
    near = fmaxf(near, tMin);
    far  = fminf(far, tMax);

    if (near > far) return false;
    if (tEntry != 0) *tEntry = near;
    return true;
}

// 三角形 -> 包围盒。稍微撑大一点（pad）：
// 共面/细长三角形会让 AABB 面积为 0，SAH 代价恒为 0 就选不出划分；
// 顺带也让「光线贴着三角形边缘」的浮点误差不至于漏掉盒子。
RT_HD RT_INLINE AABB triangleBounds(const TriangleAoS& t)
{
    AABB b;
    b.grow(t.v0);
    b.grow(t.v1);
    b.grow(t.v2);
    b.pad(1.0e-4f);
    return b;
}

// ============================================================================
//  2. 两套节点布局
// ============================================================================
//  2a. 「朴素」节点：40 字节，字段按语义直觉排（用于对比对齐前后的显存差别）。
//      40 不是 2 的幂：数组里第 k 个节点的起始偏移是 40k，会跨 32B sector，
//      也会跨 128B 缓存行（128/40 = 3.2），一次节点访问可能要发两次内存事务。
struct BvhNodeNaive
{
    Vec3 bmin;      // 12
    int  left;      // 16  内节点：左孩子下标（右孩子 = left+1）；叶子：-1
    Vec3 bmax;      // 28
    int  triCount;  // 32  0 = 内节点，>0 = 叶子里的三角形数
    int  triStart;  // 36  叶子三角形起始下标（内节点不用）
    int  pad;       // 40
};
static_assert(sizeof(BvhNodeNaive) == 40, "朴素节点按 40 字节设计，改了字段要同步注释");

//  2b. 32 字节对齐节点（优化 ①）
//      字段顺序是故意的，不是随手排的：
//        [ float bmin[3] | int left  ] = 16 字节：float3 + 1 个 int 正好配平，
//        [ float bmax[3] | int count ] = 16 字节：同上
//      两个 16 字节组拼成一个 32 字节节点，也就是 GPU 的一个 sector。
//      遍历时「读包围盒做 slab 测试」和「拿孩子下标 / 三角形范围」都在同一个
//      sector 里，一次 32B 事务就把这次节点访问要的东西全取回来了；
//      如果按 Vec3 bmin / Vec3 bmax / int left / int count 排，字段会跨
//      16 字节边界，编译器可能发两次请求，而且 alignas(32) 之后还会被填到 64。
//      alignas(32) 保证数组里每个节点都落在 sector 边界上（cudaMalloc 返回
//      的基址本来就是 256 字节对齐的）。
struct alignas(32) BvhNode
{
    float bmin[3];   // 12
    int   left;      // 16  内节点：左孩子下标（右孩子 = left+1）
                     //     叶子：三角形起始下标（与内节点复用同一字段）
    float bmax[3];   // 28
    int   count;     // 32  0 = 内节点；>0 = 叶子里的三角形数
};
static_assert(sizeof(BvhNode) == 32, "BvhNode 必须是一个 32 字节 sector");
static_assert(alignof(BvhNode) == 32, "BvhNode 必须 32 字节对齐，否则会跨 sector");

// 「按编译期开关选节点类型」的小工具：
// kernel 里写 typedef NodeSelector<Config::alignedNodes>::Type NodeT; 就能让
// 同一个 kernel 模板在两种节点布局上各实例化一份，不需要写两份遍历代码。
// （刻意不用 std::conditional：那是标准库类型，在 device 代码里用它要跟
//   nvcc 的 host/device 标注打交道，自己写三行更稳。）
template <bool Aligned> struct NodeSelector;
template <> struct NodeSelector<true>  { typedef BvhNode      Type; };
template <> struct NodeSelector<false> { typedef BvhNodeNaive Type; };

// ---- 两套布局的统一读取入口 ----------------------------------------------
// 内节点的「第一个孩子」和叶子的「第一个三角形」都用 nodeFirst()，
// 孩子成对分配，所以右孩子 = nodeFirst()+1，不需要第二个下标字段。
RT_HD RT_INLINE bool nodeIsLeaf(const BvhNode& n)      { return n.count > 0; }
RT_HD RT_INLINE bool nodeIsLeaf(const BvhNodeNaive& n) { return n.triCount > 0; }

RT_HD RT_INLINE int nodeFirst(const BvhNode& n)        { return n.left; }
RT_HD RT_INLINE int nodeFirst(const BvhNodeNaive& n)   { return nodeIsLeaf(n) ? n.triStart : n.left; }

RT_HD RT_INLINE int nodeTriCount(const BvhNode& n)      { return n.count; }
RT_HD RT_INLINE int nodeTriCount(const BvhNodeNaive& n) { return n.triCount; }

RT_HD RT_INLINE AABB nodeBox(const BvhNode& n)
{
    return AABB(Vec3(n.bmin[0], n.bmin[1], n.bmin[2]),
                Vec3(n.bmax[0], n.bmax[1], n.bmax[2]));
}

RT_HD RT_INLINE AABB nodeBox(const BvhNodeNaive& n)
{
    return AABB(n.bmin, n.bmax);
}

// ============================================================================
//  3. CPU 建树（自顶向下，显式栈；中点分割为主 + 可选 SAH）
// ============================================================================
struct BvhBuildStats
{
    int    triCount;
    int    nodeCount;        // 节点总数（两套布局同一个数）
    int    leafCount;
    int    maxDepth;
    int    maxLeafTris;
    int    sahFallbacks;     // SAH 切出空边 -> 回退到中点分割的次数
    double buildMs;
    float  avgTrisPerLeaf;
};

struct BvhBuildOptions
{
    int  leafMax;    // 叶子阈值
    int  maxDepth;   // 最大深度
    bool useSah;     // true = 12 候选 SAH，false = 质心中点分割

    BvhBuildOptions() : leafMax(BVH_MAX_LEAF_TRIANGLES), maxDepth(BVH_MAX_DEPTH), useSah(false) {}
};

struct BvhBuildResult
{
    std::vector<BvhNode>      aligned;    // 32B 对齐节点（优化后的布局）
    std::vector<BvhNodeNaive> naive;      // 40B 朴素节点（同一棵树，用于对比）
    std::vector<int>          triIndex;   // 叶子里的三角形下标（重排过，局部性更好）
    BvhBuildStats             stats;
};

// ---- 内部小工具 -----------------------------------------------------------
struct BvhBuildNode
{
    AABB box;
    int  left;        // 内节点：左孩子下标；叶子：-1
    int  triStart;    // 叶子：三角形起始下标
    int  triCount;    // 0 = 内节点
    int  depth;
};

struct BvhBuildWork
{
    int node;         // 节点数组下标
    int start;        // idx[] 上的区间
    int count;
    int depth;
};

// 质心比较器：nth_element 按「最大轴上的质心」划分。
// 捕获的是主机内存（std::vector），不要捕获设备指针。
struct CentroidLess
{
    const std::vector<Vec3>* centroids;
    int axis;

    bool operator()(int a, int b) const
    {
        return axisOf((*centroids)[a], axis) < axisOf((*centroids)[b], axis);
    }
};

// SAH：12 个候选划分（3 个轴 × 4 个位置），代价 = 表面积 × 三角形数。
// 只算「表面积 × 数量」这一项，不再乘父节点面积与遍历代价常数 —— 那是更完整的
// SAH 公式，这里保持可读、可解释，效果已经比中点分割好（build 时间换渲染时间）。
struct SahChoice
{
    bool  valid;
    float cost;
    int   axis;
    float pos;
};

inline SahChoice chooseSahSplit(const std::vector<int>& idx, int start, int count,
                                const std::vector<AABB>& boxes,
                                const std::vector<Vec3>& centroids,
                                const AABB& cbox)
{
    SahChoice best;
    best.valid = false;
    best.cost = 1.0e30f;
    best.axis = 0;
    best.pos = 0.0f;

    for (int axis = 0; axis < 3; ++axis)
    {
        const float lo = axisOf(cbox.bmin, axis);
        const float hi = axisOf(cbox.bmax, axis);
        if (hi - lo <= 0.0f) continue;                 // 这个轴上质心全重合，切了也是空边

        // 4 个位置取 1/5、2/5、3/5、4/5：贴着两端的位置几乎必然切出空边，跳过更省时间。
        for (int c = 1; c <= 4; ++c)
        {
            const float pos = lo + (hi - lo) * ((float)c / 5.0f);

            AABB leftBox, rightBox;
            int  leftCount = 0, rightCount = 0;
            for (int i = start; i < start + count; ++i)
            {
                const int t = idx[i];
                if (axisOf(centroids[t], axis) < pos) { leftBox.grow(boxes[t]);  ++leftCount; }
                else                                  { rightBox.grow(boxes[t]); ++rightCount; }
            }
            if (leftCount == 0 || rightCount == 0) continue;

            const float cost = leftBox.area() * (float)leftCount + rightBox.area() * (float)rightCount;
            if (cost < best.cost)
            {
                best.valid = true;
                best.cost = cost;
                best.axis = axis;
                best.pos = pos;
            }
        }
    }
    return best;
}

// 划分谓词：与 SAH 统计时用的判断完全一致，保证「算出来的代价」和「真的划分」一致。
struct SplitLess
{
    const std::vector<Vec3>* centroids;
    int   axis;
    float pos;

    bool operator()(int t) const { return axisOf((*centroids)[t], axis) < pos; }
};

// 把两个孩子的待处理项压栈。
// 先压右再压左：栈是后进先出，左孩子先被处理，于是「先左后右」的处理顺序
// 和节点数组里 左孩子(leftIdx) / 右孩子(leftIdx+1) 的排布一致，缓存更友好。
inline void pushBvhChildren(std::vector<BvhBuildWork>* stack, int leftIdx,
                            const BvhBuildWork& parent, int leftCount)
{
    BvhBuildWork rw;
    rw.node = leftIdx + 1;
    rw.start = parent.start + leftCount;
    rw.count = parent.count - leftCount;
    rw.depth = parent.depth + 1;
    stack->push_back(rw);

    BvhBuildWork lw;
    lw.node = leftIdx;
    lw.start = parent.start;
    lw.count = leftCount;
    lw.depth = parent.depth + 1;
    stack->push_back(lw);
}

inline void buildBvh(const std::vector<TriangleAoS>& tris, const BvhBuildOptions& opt,
                     BvhBuildResult* out)
{
    const std::chrono::steady_clock::time_point t0 = std::chrono::steady_clock::now();

    const int n = (int)tris.size();

    out->aligned.clear();
    out->naive.clear();
    out->triIndex.clear();

    BvhBuildStats& st = out->stats;
    memset(&st, 0, sizeof(st));
    st.triCount = n;

    if (n == 0)
    {
        // 空场景也要有一个根节点，遍历代码就不用写特例（nodeCount = 0 时直接返回）。
        BvhNode root;
        root.bmin[0] = root.bmin[1] = root.bmin[2] = 0.0f;
        root.bmax[0] = root.bmax[1] = root.bmax[2] = 0.0f;
        root.left = 0;
        root.count = 0;
        out->aligned.push_back(root);

        BvhNodeNaive q;
        q.bmin = q.bmax = Vec3(0.0f, 0.0f, 0.0f);
        q.left = -1; q.triCount = 0; q.triStart = 0; q.pad = 0;
        out->naive.push_back(q);

        st.nodeCount = 1;
        st.buildMs = 0.0;
        return;
    }

    // 每个三角形的包围盒与质心：建树过程中要反复用，预先算好。
    std::vector<AABB> boxes(n);
    std::vector<Vec3> centroids(n);
    for (int i = 0; i < n; ++i)
    {
        boxes[i] = triangleBounds(tris[i]);
        centroids[i] = boxes[i].centroid();
    }

    // 三角形下标数组：建树时被反复划分，建完就是叶子里的访问顺序。
    std::vector<int> idx(n);
    for (int i = 0; i < n; ++i) idx[i] = i;

    std::vector<BvhBuildNode> nodes;
    // 保守预留 2n+1 个槽位（二叉树节点数上界 2*叶子数-1，而叶子最多 n 个）：
    // 这样 push_back 永远不会触发重新分配，节点的下标也就一直稳定。
    nodes.reserve((size_t)(2 * n + 1));

    nodes.push_back(BvhBuildNode());     // 根（字段下面填）

    std::vector<BvhBuildWork> stack;
    stack.reserve(128);
    BvhBuildWork rootWork;
    rootWork.node = 0; rootWork.start = 0; rootWork.count = n; rootWork.depth = 0;
    stack.push_back(rootWork);

    int maxDepthSeen = 0;
    int maxLeafTris = 0;

    while (!stack.empty())
    {
        const BvhBuildWork w = stack.back();
        stack.pop_back();

        // ---- 1) 节点包围盒 + 质心包围盒 ----
        AABB box, cbox;
        for (int i = w.start; i < w.start + w.count; ++i)
        {
            box.grow(boxes[idx[i]]);
            cbox.grow(centroids[idx[i]]);
        }

        // 本节点的字段先攒在局部副本里，孩子槽位分配完再统一写回。
        // 原因：push_back 可能让 nodes 重新分配，先前取到的 BvhBuildNode& 会失效
        // （悬垂引用在 Debug 下不一定崩、Release 下才炸，属于最难查的一类 bug）。
        BvhBuildNode me;
        me.box = box;
        me.depth = w.depth;
        me.left = -1;
        me.triStart = 0;
        me.triCount = 0;

        if (w.depth > maxDepthSeen) maxDepthSeen = w.depth;

        // ---- 2) 叶子判定 ----
        const bool mustStop = (w.count <= opt.leafMax) || (w.depth >= opt.maxDepth);
        if (mustStop)
        {
            me.left = -1;
            me.triStart = w.start;
            me.triCount = w.count;
            if (w.count > maxLeafTris) maxLeafTris = w.count;
            nodes[w.node] = me;
            continue;
        }

        // ---- 3) 选划分 ----
        const int axis = cbox.longestAxis();
        bool partitioned = false;

        if (opt.useSah)
        {
            const SahChoice choice = chooseSahSplit(idx, w.start, w.count, boxes, centroids, cbox);
            if (choice.valid)
            {
                // ⚠️ 这里必须用**花括号**初始化，不能用圆括号。
                // SplitLess 是「聚合类型」（只有成员、没有用户声明的构造函数），
                // C++17 下 `SplitLess(a, b, c)`（圆括号）不合法 —— 那是 C++20 才加的
                // parenthesized aggregate initialization（P0960）。C++17 编译器会报
                //   error : too many initializer values
                //   error : no suitable constructor exists to convert from
                //           "std::vector<Vec3> *" to "SplitLess"
                // 用花括号就是标准写法（C++11 起就支持聚合初始化）。
                const SplitLess less{&centroids, choice.axis, choice.pos};
                std::vector<int>::iterator first = idx.begin() + w.start;
                const std::vector<int>::iterator midIt = std::partition(first, first + w.count, less);
                const int leftCount = (int)(midIt - first);
                if (leftCount > 0 && leftCount < w.count)
                {
                    me.left = (int)nodes.size();       // 内节点：存左孩子下标（右孩子 = +1）
                    nodes.push_back(BvhBuildNode());   // 左孩子
                    nodes.push_back(BvhBuildNode());   // 右孩子（成对分配，遍历时不用第二个下标字段）
                    pushBvhChildren(&stack, me.left, w, leftCount);
                    partitioned = true;
                }
                else
                {
                    ++st.sahFallbacks;   // 划分退化（一边为空）-> 下面用中点分割兜底
                }
            }
            else
            {
                ++st.sahFallbacks;       // 12 个候选全退化（质心全重合）-> 同上
            }
        }

        if (!partitioned)
        {
            // 中点分割：在质心包围盒最长轴的中位数处切开。
            // 用 nth_element 而不是 sort：只需要「左边都不大于右边」，平均 O(n)，
            // 而且 count >= 2 时一定能切出两个非空区间，建树不会卡住。
            const int mid = w.start + w.count / 2;
            std::nth_element(idx.begin() + w.start, idx.begin() + mid,
                             idx.begin() + w.start + w.count,
                             CentroidLess{&centroids, axis});   // 聚合初始化：花括号（同 Line 508 的说明）

            const int leftCount = mid - w.start;
            me.left = (int)nodes.size();
            nodes.push_back(BvhBuildNode());
            nodes.push_back(BvhBuildNode());
            pushBvhChildren(&stack, me.left, w, leftCount);
        }

        nodes[w.node] = me;
    }

    // ---- 4) 摊平成两套布局（同一棵树，字段含义一一对应）----
    const size_t nodeCount = nodes.size();
    out->aligned.resize(nodeCount);
    out->naive.resize(nodeCount);

    int leafCount = 0;
    for (size_t i = 0; i < nodeCount; ++i)
    {
        const BvhBuildNode& b = nodes[i];
        const bool leaf = (b.triCount > 0);
        if (leaf) ++leafCount;

        BvhNode& a = out->aligned[i];
        a.bmin[0] = b.box.bmin.x; a.bmin[1] = b.box.bmin.y; a.bmin[2] = b.box.bmin.z;
        a.bmax[0] = b.box.bmax.x; a.bmax[1] = b.box.bmax.y; a.bmax[2] = b.box.bmax.z;
        a.left = leaf ? b.triStart : b.left;
        a.count = b.triCount;

        BvhNodeNaive& q = out->naive[i];
        q.bmin = b.box.bmin;
        q.bmax = b.box.bmax;
        q.left = leaf ? -1 : b.left;
        q.triCount = b.triCount;
        q.triStart = leaf ? b.triStart : 0;
        q.pad = 0;
    }

    out->triIndex.swap(idx);

    st.nodeCount = (int)nodeCount;
    st.leafCount = leafCount;
    st.maxDepth = maxDepthSeen;
    st.maxLeafTris = maxLeafTris;
    st.avgTrisPerLeaf = (leafCount > 0) ? (float)n / (float)leafCount : 0.0f;

    const std::chrono::steady_clock::time_point t1 = std::chrono::steady_clock::now();
    st.buildMs = std::chrono::duration<double, std::milli>(t1 - t0).count();

    // 建树统计（节点数 / 叶子数 / 平均每叶三角形数 / 最大深度 / 建树耗时）
    printf("  建树: 划分=%s  三角形=%d  节点=%d  叶子=%d  平均每叶=%.2f  最大深度=%d  叶内最多=%d"
           "  回退=%d  耗时=%.1f ms\n",
           opt.useSah ? "SAH(12 候选)" : "质心中点(nth_element)", n, st.nodeCount, st.leafCount,
           st.avgTrisPerLeaf, st.maxDepth, st.maxLeafTris, st.sahFallbacks, st.buildMs);
    printf("        节点内存: 32B 对齐版 %.1f KB (每节点 %d 字节) / 40B 朴素版 %.1f KB\n",
           (double)nodeCount * sizeof(BvhNode) / 1024.0, (int)sizeof(BvhNode),
           (double)nodeCount * sizeof(BvhNodeNaive) / 1024.0);
}

// ============================================================================
//  4. 设备端遍历（显式栈 + 有序 + best 剪枝）
// ============================================================================
// 视图：只装设备指针，按值传进 kernel（不拥有任何内存）。
template <typename NodeT>
struct BvhView
{
    const NodeT* nodes;
    int          nodeCount;
    const int*   triIndex;    // 叶子三角形下标（重排数组）
    TriStore     tris;        // 三角形数据（AoS / SoA 两套都在里面）

    BvhView() : nodes(0), nodeCount(0), triIndex(0), tris() {}
};

// 取节点：UseSmem = true 时优先读共享内存里的根节点副本。
template <bool UseSmem, typename NodeT>
RT_HD RT_INLINE NodeT fetchNode(const BvhView<NodeT>& bvh, const NodeT* smemNodes, int index)
{
    if (UseSmem)
    {
        if (index < BVH_SMEM_NODE_COUNT) return smemNodes[index];
    }
    return bvh.nodes[index];
}

// ---------------------------------------------------------------------------
//  核心遍历：一个函数同时服务「最近命中」和「遮挡查询」两种用法
// ---------------------------------------------------------------------------
// AnyHit = false: 找最近交点（主光线、反射、折射）
// AnyHit = true : 命中一个就返回（阴影光线，省掉剩下的遍历）
//
// 遍历结构（核心部分）：
//     * 进入循环时，当前节点的盒子**已经**被判定为命中，所以每个节点只测一次盒子；
//       测试在孩子身上做：命中近的就下去、远的压栈，避免「父节点测一遍、
//       子节点再测一遍」的重复访存。
//     * 有序遍历：两个孩子的入口距离 t 谁小谁先走 —— 先命中的概率大，
//       best 会更早变小，另一个孩子的盒子就更可能被剪掉。
//     * 栈里同时存「入口距离 t」：退栈时如果这个 t 已经 >= best，
//       说明该子树不可能有更近的交点，直接丢掉，连盒子都不用测。
// ---------------------------------------------------------------------------
//  显式栈遍历（最近命中 / any-hit 共用一份，靠 AnyHit 模板参数分派）
// ---------------------------------------------------------------------------
// ⚠️ 这个函数**故意不加 RT_INLINE（__forceinline__）**，这是实际踩到的一个
//    真实崩溃里学来的，改之前请先看完再动：
//
//    函数体里有 int stack[BVH_STACK_SIZE] + float stackT[BVH_STACK_SIZE]
//    = 512 字节的局部数组。而调用链是：
//        renderKernel → traceRay（**递归**）→ intersectMesh/intersectScene（__forceinline__）
//                     → intersectBvh/occludedBvh（也是 __forceinline__）→ traverseBvh
//    只要 traverseBvh 也被强制内联，这 512 字节就会被复制进**每一层递归的栈帧**，
//    而且 traceRay 里求交调用出现多次（主光线一次 + 阴影一次），
//    -O0（Debug）下每层要 4~5 KB，depth=5 的 6 层就是 30 KB，直接顶爆线程栈，
//    设备端报 cudaErrorLaunchFailure(719)，且在 Nsight 里表现为「停在 traceRay
//    函数入口、局部变量全部 (Error: Value not available)」——很像无限递归，
//    实际是栈溢出。
//
//    早期版本就是这么崩的（cudaErrorLaunchFailure 719），当时同分辨率下暴力版能跑、
//    BVH 版必崩，depth=1 通过 / depth=5 崩。去掉强制内联后 6 层只需约 2 KB。
//    Release 构建下编译器仍会自己决定是否内联，性能不受影响。
//
// Ordered 模板参数（--bench 用）：
//    true  = **有序遍历**，先走入口距离近的孩子（近的先命中 -> best 立刻变小 ->
//            远处那棵子树整个被 tEntry >= best 剪掉）；
//    false = **固定先左后右**，完全不看距离。这是**对照组**，不是"劣化版" ——
//            两档一减，差出来的就是「有序遍历」这一项优化本身的收益。
//    因为它是编译期常量，false 时下面那个三元表达式会被直接折成 true，没有运行期开销。
template <bool UseSoA, bool UseSmem, bool AnyHit, bool Ordered, typename NodeT>
RT_HD bool traverseBvh(const BvhView<NodeT>& bvh, const NodeT* smemNodes,
                       const Ray& ray, float tMin, float tMax,
                       HitRecord* rec, int* nodeVisits,
                       int* peakOut, int* overflowOut)
{
    if (bvh.nodeCount <= 0 || bvh.triIndex == 0) return false;

    int   stack[BVH_STACK_SIZE];
    float stackT[BVH_STACK_SIZE];        // 每层入口距离，退栈时用来二次剪枝
    int   sp = 0;
    int   peak = 0;                      // 本函数内栈使用峰值
    int   overflow = 0;                  // 压栈溢出次数（不能静默丢三角形！）

    const Vec3 invD = safeInvDir(ray.direction);

    float best = tMax;
    bool  found = false;

    // 根节点单独判一次（之后所有节点都是「父节点已经确认命中」才进来的）。
    float tEntry;
    if (!hitAabb(nodeBox(fetchNode<UseSmem>(bvh, smemNodes, 0)), ray.origin, invD, tMin, best, &tEntry))
        return false;

    int nodeIndex = 0;
    for (;;)
    {
        if (nodeVisits != 0) ++(*nodeVisits);

        const NodeT node = fetchNode<UseSmem>(bvh, smemNodes, nodeIndex);

        if (nodeIsLeaf(node))
        {
            // ---- 叶子：逐个测三角形，best 只减不增 ----
            const int first = nodeFirst(node);
            const int count = nodeTriCount(node);
            for (int i = 0; i < count; ++i)
            {
                const TriangleAoS tri = triAt<UseSoA>(bvh.tris, bvh.triIndex[first + i]);
                if (AnyHit)
                {
                    if (hitTriangleAny(tri, ray, tMin, best)) { found = true; break; }
                }
                else
                {
                    HitRecord tmp;
                    if (hitTriangle(tri, ray, tMin, best, &tmp))
                    {
                        best = tmp.t;          // 收紧区间：后面的三角形自动享受剪枝
                        *rec = tmp;
                        found = true;
                    }
                }
            }
        }
        else
        {
            // ---- 内节点：两个孩子成对分配，右孩子 = 左孩子 + 1 ----
            const int leftIdx  = nodeFirst(node);
            const int rightIdx = leftIdx + 1;

            float tLeft = 0.0f, tRight = 0.0f;
            const bool hitLeft  = hitAabb(nodeBox(fetchNode<UseSmem>(bvh, smemNodes, leftIdx)),
                                          ray.origin, invD, tMin, best, &tLeft);
            const bool hitRight = hitAabb(nodeBox(fetchNode<UseSmem>(bvh, smemNodes, rightIdx)),
                                          ray.origin, invD, tMin, best, &tRight);

            if (hitLeft && hitRight)
            {
                // 有序遍历：近的先走、远的压栈。
                // Ordered=false 时固定「先左后右」（不看距离），作为对照组。
                const bool goLeftFirst = Ordered ? (tLeft <= tRight) : true;
                if (goLeftFirst)
                {
                    if (sp < BVH_STACK_SIZE) { stack[sp] = rightIdx; stackT[sp] = tRight; ++sp; }
                    else ++overflow;
                    nodeIndex = leftIdx;
                }
                else
                {
                    if (sp < BVH_STACK_SIZE) { stack[sp] = leftIdx; stackT[sp] = tLeft; ++sp; }
                    else ++overflow;
                    nodeIndex = rightIdx;
                }
                if (sp > peak) peak = sp;
                continue;                                   // 孩子盒子已经测过，下一轮不再测
            }
            if (hitLeft)  { nodeIndex = leftIdx;  continue; }
            if (hitRight) { nodeIndex = rightIdx; continue; }
            // 两个孩子都没中 -> 落到下面退栈
        }

        if (AnyHit && found) break;      // 阴影光线：命中一个就够，不用再看别的子树

        // ---- 退栈：跳过「入口距离已经不小于 best」的节点 ----
        int next = -1;
        while (sp > 0)
        {
            --sp;
            if (stackT[sp] < best) { next = stack[sp]; break; }
        }
        if (next < 0) break;
        nodeIndex = next;
    }

    // peak / overflow 是累加语义：一次渲染里主光线 + 所有阴影光线共用同一对计数器，
    // kernel 里只维护两个寄存器变量即可，不用给每条光线单独存一遍。
    if (peakOut != 0 && peak > *peakOut) *peakOut = peak;
    if (overflowOut != 0) *overflowOut += overflow;
    return found;
}

// ---- 两个薄封装，给 kernel 用 --------------------------------------------
// 最近命中（triangle + BVH）
template <typename NodeT, bool UseSoA, bool UseSmem, bool Ordered = true>
RT_HD RT_INLINE bool intersectBvh(const BvhView<NodeT>& bvh, const NodeT* smemNodes,
                                  const Ray& ray, float tMin, float tMax, HitRecord* rec,
                                  int* nodeVisits, int* peakOut, int* overflowOut)
{
    return traverseBvh<UseSoA, UseSmem, false, Ordered>(bvh, smemNodes, ray, tMin, tMax,
                                                        rec, nodeVisits, peakOut, overflowOut);
}

// 遮挡查询（阴影光线，命中即返回）
template <typename NodeT, bool UseSoA, bool UseSmem, bool Ordered = true>
RT_HD RT_INLINE bool occludedBvh(const BvhView<NodeT>& bvh, const NodeT* smemNodes,
                                 const Ray& ray, float tMin, float tMax,
                                 int* peakOut, int* overflowOut)
{
    HitRecord dummy;
    return traverseBvh<UseSoA, UseSmem, true, Ordered>(bvh, smemNodes, ray, tMin, tMax,
                                                       &dummy, (int*)0, peakOut, overflowOut);
}

// ============================================================================
//  5. 主机侧：设备内存管理（三角形 AoS + SoA、两套节点、triIndex）
// ============================================================================
// 这些函数是主机代码（只在 main 里调用），故意不加 RT_HD：cudaMalloc 不能在
// 设备端调用，加了标记反而会让 nvcc 去实例化设备版本。
struct GpuTriangles
{
    TriangleAoS* aos;
    float*       soaFloats;
    int*         soaMaterialIds;
    int          count;
    TriStore     store;

    GpuTriangles() : aos(0), soaFloats(0), soaMaterialIds(0), count(0), store() {}

    void upload(const std::vector<TriangleAoS>& tris)
    {
        count = (int)tris.size();
        if (count <= 0) return;

        const size_t aosBytes = sizeof(TriangleAoS) * (size_t)count;
        CUDA_CHECK_MSG(cudaMalloc((void**)&aos, aosBytes), "cudaMalloc(TriangleAoS)");
        CUDA_CHECK_MSG(cudaMemcpy(aos, tris.data(), aosBytes, cudaMemcpyHostToDevice),
                       "cudaMemcpy(TriangleAoS)");

        // SoA：9 条 float 数组 + 1 条 int 数组，各一块连续内存 -> 两次拷贝。
        std::vector<float> soaF;
        std::vector<int>   soaM;
        packTriangleSoA(tris, &soaF, &soaM);

        CUDA_CHECK_MSG(cudaMalloc((void**)&soaFloats, sizeof(float) * soaF.size()), "cudaMalloc(SoA floats)");
        CUDA_CHECK_MSG(cudaMemcpy(soaFloats, soaF.data(), sizeof(float) * soaF.size(),
                                  cudaMemcpyHostToDevice), "cudaMemcpy(SoA floats)");
        CUDA_CHECK_MSG(cudaMalloc((void**)&soaMaterialIds, sizeof(int) * soaM.size()), "cudaMalloc(SoA materialId)");
        CUDA_CHECK_MSG(cudaMemcpy(soaMaterialIds, soaM.data(), sizeof(int) * soaM.size(),
                                  cudaMemcpyHostToDevice), "cudaMemcpy(SoA materialId)");

        store.aos = aos;
        store.soa = bindTriangleSoA(soaFloats, soaMaterialIds, count);
        store.count = count;
    }

    void release()
    {
        if (aos)            { cudaFree(aos);            aos = 0; }
        if (soaFloats)      { cudaFree(soaFloats);      soaFloats = 0; }
        if (soaMaterialIds) { cudaFree(soaMaterialIds); soaMaterialIds = 0; }
        count = 0;
        store = TriStore();
    }
};

template <typename NodeT>
struct GpuBvh
{
    NodeT* nodes;
    int*   triIndex;
    int    nodeCount;
    int    triCount;

    GpuBvh() : nodes(0), triIndex(0), nodeCount(0), triCount(0) {}

    void upload(const std::vector<NodeT>& nodeList, const std::vector<int>& indexList)
    {
        nodeCount = (int)nodeList.size();
        triCount = (int)indexList.size();
        if (nodeCount <= 0) return;

        CUDA_CHECK_MSG(cudaMalloc((void**)&nodes, sizeof(NodeT) * (size_t)nodeCount), "cudaMalloc(BvhNode)");
        CUDA_CHECK_MSG(cudaMemcpy(nodes, nodeList.data(), sizeof(NodeT) * (size_t)nodeCount,
                                  cudaMemcpyHostToDevice), "cudaMemcpy(BvhNode)");

        if (triCount > 0)
        {
            CUDA_CHECK_MSG(cudaMalloc((void**)&triIndex, sizeof(int) * (size_t)triCount), "cudaMalloc(triIndex)");
            CUDA_CHECK_MSG(cudaMemcpy(triIndex, indexList.data(), sizeof(int) * (size_t)triCount,
                                      cudaMemcpyHostToDevice), "cudaMemcpy(triIndex)");
        }
    }

    BvhView<NodeT> view(const TriStore& tris) const
    {
        BvhView<NodeT> v;
        v.nodes = nodes;
        v.nodeCount = nodeCount;
        v.triIndex = triIndex;
        v.tris = tris;
        return v;
    }

    void release()
    {
        if (nodes)    { cudaFree(nodes);    nodes = 0; }
        if (triIndex) { cudaFree(triIndex); triIndex = 0; }
        nodeCount = 0;
        triCount = 0;
    }
};
