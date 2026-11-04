// ============================================================================
//  obj_assimp.cpp  --  assimp 模型加载器的实现（普通 C++ 编译单元）
// ----------------------------------------------------------------------------
//  为什么单独一个 .cpp，而不是写进 obj_loader.h：
//    <assimp/Importer.hpp> 会拖进 60+ 个内部头、几万行模板。如果把它 include
//    进 obj_loader.h，而后者又被 kernel.cu include，那么 **nvcc 的 device 编译
//    阶段也会去解析这些头** —— 既显著拖慢编译，也容易和 CUDA 的类型/宏冲突。
//    放到独立的 .cpp 里，它就只经过普通 C++ 编译器，nvcc 完全看不到。
//
//  为什么不用 empty.cpp：
//    empty.cpp 的职责是「让 VS 的 C/C++ 页属性有消费者」的**空占位单元**，
//    保持它不含任何代码更符合用途，也不会让人误判项目构成。
//
//  ⚠️ 本文件需要手动加入项目：右键项目 → 添加 → 现有项。
//     它和 kernel.cu 是两个独立的编译单元，前者走 nvcc，后者走 cl.exe。
// ============================================================================
// ⚠️ 必须在 obj_loader.h 之前 include cuda_runtime.h：
//    本文件是普通 C++ 编译单元（不经过 nvcc），而 obj_loader.h 会 include vector
//    相关的头，其中可能用到 CUDA 内建类型（如 uchar4）。没有 cuda_runtime.h 时
//    cl.exe 会报 error C4430「缺少类型说明符」。
#include <cuda_runtime.h>

#include <assimp/Importer.hpp>
#include <assimp/scene.h>
#include <assimp/postprocess.h>
#include <assimp/version.h>

#include <cstdio>
#include <cstring>
#include <string>

#include "obj_loader.h"

const char* assimpVersionString()
{
    static char buf[32] = { 0 };
    if (buf[0] == '\0')
    {
        std::snprintf(buf, sizeof(buf), "%d.%d.%d",
                      aiGetVersionMajor(), aiGetVersionMinor(), aiGetVersionRevision());
    }
    return buf;
}

// 用 assimp 读模型，直接产出 whitted 需要的 TriMesh（vector<TriangleAoS> + bounds + materialId）。
// 与手写 loadObj 的差异只有错误输出方式：这里把 assimp 的失败原因写进 errOut 缓冲。
bool loadObjWithAssimp(const char* path, TriMesh* mesh, int materialId,
                       char* errOut, size_t errSize)
{
    if (mesh == nullptr || path == nullptr) return false;
    mesh->triangles.clear();
    mesh->materialId = materialId;

    Assimp::Importer importer;
    const aiScene* scene = importer.ReadFile(
        path,
        aiProcess_Triangulate | aiProcess_JoinIdenticalVertices | aiProcess_SortByPType);

    if (scene == nullptr || scene->mRootNode == nullptr)
    {
        if (errOut != nullptr && errSize > 0)
            std::snprintf(errOut, errSize, "%s", importer.GetErrorString());
        return false;
    }

    for (unsigned m = 0; m < scene->mNumMeshes; ++m)
    {
        const aiMesh* am = scene->mMeshes[m];
        if (am == nullptr) continue;

        for (unsigned f = 0; f < am->mNumFaces; ++f)
        {
            const aiFace& face = am->mFaces[f];
            if (face.mNumIndices != 3) continue;

            const aiVector3D& p0 = am->mVertices[face.mIndices[0]];
            const aiVector3D& p1 = am->mVertices[face.mIndices[1]];
            const aiVector3D& p2 = am->mVertices[face.mIndices[2]];

            TriangleAoS tri;
            tri.v0 = Vec3(static_cast<float>(p0.x), static_cast<float>(p0.y), static_cast<float>(p0.z));
            tri.v1 = Vec3(static_cast<float>(p1.x), static_cast<float>(p1.y), static_cast<float>(p1.z));
            tri.v2 = Vec3(static_cast<float>(p2.x), static_cast<float>(p2.y), static_cast<float>(p2.z));
            tri.materialId = materialId;
            mesh->triangles.push_back(tri);
        }
    }

    if (mesh->triangles.empty())
    {
        if (errOut != nullptr && errSize > 0)
            std::snprintf(errOut, errSize, "assimp 解析出 0 个三角形");
        return false;
    }

    mesh->recomputeBounds();
    return true;
}