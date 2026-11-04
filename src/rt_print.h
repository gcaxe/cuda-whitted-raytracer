// ============================================================================
//  rt_print.h  --  输出小工具（**只用 iostream，不用 printf 家族**）
// ----------------------------------------------------------------------------
//  本项目的输出约定：stdout/stderr 一律走 std::cout / std::cerr。
//
//  但 printf 的格式化能力在流式输出里写起来很别扭，而且有两个真实的坑：
//    1. std::fixed / std::setprecision 是**粘性**的：设一次会影响之后所有浮点输出，
//       漏掉一次恢复就会让整份日志的小数位数全变（例如 1.384 变成 1.38）。
//    2. 宽度对齐要反复写 std::left / std::setw / std::right，很容易漏。
//  所以这里提供几个小的格式化工具：内部用 std::ostringstream 拼好再返回 string，
//  外面看起来就是普通的 << 拼接 —— **不产生任何全局状态副作用**。
// ============================================================================
#pragma once

#include <string>
#include <sstream>
#include <iomanip>

namespace rt {

// 固定 n 位小数（等价 printf 的 %.nf）
inline std::string fixed(double v, int n = 2)
{
    std::ostringstream os;
    os << std::fixed << std::setprecision(n) << v;
    return os.str();
}

// 左对齐、空格补到 w 宽（等价 printf 的 %-ws）
inline std::string ljust(const std::string& s, int w)
{
    if (static_cast<int>(s.size()) >= w) return s;
    return s + std::string(static_cast<size_t>(w) - s.size(), ' ');
}

// 右对齐、空格补到 w 宽（等价 printf 的 %wd）
inline std::string rjust(const std::string& s, int w)
{
    if (static_cast<int>(s.size()) >= w) return s;
    return std::string(static_cast<size_t>(w) - s.size(), ' ') + s;
}

// 右对齐的整数版本（等价 %wd），方便直接塞进 << 链
inline std::string rjust(int v, int w)
{
    std::ostringstream os;
    os << v;
    return rjust(os.str(), w);
}

// 左对齐的整数版本（等价 %-wd）
inline std::string ljust(int v, int w)
{
    std::ostringstream os;
    os << v;
    return ljust(os.str(), w);
}


// 科学计数法（等价 printf 的 %.ne），例如 RT_SHADOW_EPS = 1e-3 打印成 1.0e-03
inline std::string sci(double v, int n = 2)
{
    std::ostringstream os;
    os << std::scientific << std::setprecision(n) << v;
    return os.str();
}
}  // namespace rt