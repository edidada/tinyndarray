#ifndef TINYNDARRAY_OPS_IMPL_H
#define TINYNDARRAY_OPS_IMPL_H

#include "tinyndarray.h"
#include <functional>
#include <sstream>
#include <vector>

namespace tinyndarray {

enum class ReduceType  {
    SUM,
    MAX,
    MIN,
    UNKNOWN
};

int ResolveAxis(int axis, size_t ndim, const std::string& name);

Axis ResolveAxis(const Axis& axes, size_t ndim, const std::string& name,
                        bool sort = false, bool sort_order_normal=true);
// 定义元素级二元操作函数类型
using BinaryElementOp = std::function<void(void* out, const void* a, const void* b)>;
using UnaryElementOp = void(*)(void* out, const void* in);
using ScalarElementOp = void(*)(void* out, const void* in, const void* scalar);


static Shape PadShape(const Shape& shape, size_t size) {
    if (size < shape.size()) {
        throw std::runtime_error("Invalid shape to pad");
    }
    const size_t n_pad = size - shape.size();
    Shape ret_shape;
    ret_shape.reserve(size);
    ret_shape.resize(n_pad, 1);                                     // Fill by 1
    ret_shape.insert(ret_shape.end(), shape.begin(), shape.end());  // Concat
    return ret_shape;
}

static size_t ReduceShapesBroadcast(Shape& ret_shape, Shape& l_shape,
                                    Shape& r_shape, const size_t depth_offset) {
    // Require `ret_shape.size() == l_shape.size() == r_shape.size()`

    // Remove meaningless dimensions.
    Shape ret_shape_cleaned, l_shape_cleaned, r_shape_cleaned;
    int size_pool = 1;
    size_t depth = 0;
    for (; depth < ret_shape.size() - depth_offset; depth++) {
        if (l_shape[depth] == r_shape[depth]) {
            // Store
            size_pool *= l_shape[depth];
        } else {
            // Pop
            if (size_pool != 1) {
                ret_shape_cleaned.push_back(size_pool);
                l_shape_cleaned.push_back(size_pool);
                r_shape_cleaned.push_back(size_pool);
                size_pool = 1;
            }
            // Through current dimension
            ret_shape_cleaned.push_back(ret_shape[depth]);
            l_shape_cleaned.push_back(l_shape[depth]);
            r_shape_cleaned.push_back(r_shape[depth]);
        }
    }
    // Pop
    if (size_pool != 1 || ret_shape_cleaned.size() == 0) {
        ret_shape_cleaned.push_back(size_pool);
        l_shape_cleaned.push_back(size_pool);
        r_shape_cleaned.push_back(size_pool);
    }
    // Store actual depth count
    const size_t n_depth = ret_shape_cleaned.size();
    // Pass through included in `depth_offset`.
    for (; depth < ret_shape.size(); depth++) {
        ret_shape_cleaned.push_back(ret_shape[depth]);
        l_shape_cleaned.push_back(l_shape[depth]);
        r_shape_cleaned.push_back(r_shape[depth]);
    }
    // Return
    ret_shape = std::move(ret_shape_cleaned);
    l_shape = std::move(l_shape_cleaned);
    r_shape = std::move(r_shape_cleaned);
    return n_depth;
}

static Shape CheckBroadcastable(const Shape& l_shape, const Shape& r_shape) {
    // We assuming left array has deeper shape than right one.
    if (l_shape.size() < r_shape.size()) {
        return CheckBroadcastable(r_shape, l_shape);  // Swap
    }
    // `l_shape.size()` is maximum depth.

    // Check empty
    if (r_shape.size() == 0 || (r_shape.size() == 1 && r_shape[0] == 0)) {
        throw std::runtime_error("Broadcast of empty array");
    }

    // Compute broadcasted shape
    Shape shape(l_shape.size());
    size_t r_offset = l_shape.size() - r_shape.size();
    for (size_t i = 0; i < l_shape.size(); i++) {
        if (i < r_offset) {
            shape[i] = l_shape[i];
        } else {
            const int l = l_shape[i];
            const int r = r_shape[i - r_offset];
            if (l == r) {
                shape[i] = l;  // no broadcast
            } else if (l == 1) {
                shape[i] = r;  // left broadcast
            } else if (r == 1) {
                shape[i] = l;  // right broadcast
            } else {
                std::stringstream ss;
                ss << "Non operatable shape";
                ss << " (" << l_shape << " vs " << r_shape << ")";
                throw std::runtime_error(ss.str());
            }
        }
    }
    return shape;
}

// 广播操作的实现
template <typename F>
void ApplyOpBroadcastImpl(char* ret_data,
                         const char* l_data,
                         const char* r_data,
                         const Shape& ret_shape, const int ret_size,
                         const std::vector<int>& l_steps,
                         const std::vector<int>& r_steps,
                         const size_t start_depth, const size_t n_depth,
                         const int ret_step, F op, DataType dtype, size_t type_size) {
    // 创建栈和计数器
    std::vector<int> ret_cnts(n_depth);
    std::vector<int> l_idx_stack(n_depth), r_idx_stack(n_depth);
    size_t depth = start_depth;
    int l_idx = 0;
    int r_idx = 0;

    for (int ret_idx = 0; ret_idx < ret_size; ret_idx += ret_step) {
        // 向下遍历
        for (; depth < n_depth; depth++) {
            l_idx_stack[depth] = l_idx;  // 压栈
            r_idx_stack[depth] = r_idx;
        }

        // 执行操作
        op(ret_data + ret_idx * type_size, 
           l_data + l_idx * type_size, 
           r_data + r_idx * type_size);

        // 向上计数
        for (; start_depth < depth; depth--) {
            const size_t prev_d = depth - 1;
            ret_cnts[prev_d]++;        // 计数增加
            l_idx += l_steps[prev_d];  // 前进索引
            r_idx += r_steps[prev_d];
            if (ret_cnts[prev_d] < ret_shape[prev_d]) {
                break;  // 正常继续
            }
            // 返回上一层
            ret_cnts[prev_d] = 0;         // 重置计数
            l_idx = l_idx_stack[prev_d];  // 出栈
            r_idx = r_idx_stack[prev_d];
        }
    }
}

// 广播操作的主函数
template <typename F>
void ApplyOpBroadcast(NdArray& ret, const NdArray& lhs, const NdArray& rhs,
                      const size_t depth_offset, const int ret_step, F op) {
    Shape ret_shape = ret.shape();
    DataType dtype = ret.dtype();
    size_t type_size = NdArray::get_type_size(dtype);

    // 预处理形状
    Shape l_shape = PadShape(lhs.shape(), ret_shape.size());
    Shape r_shape = PadShape(rhs.shape(), ret_shape.size());

    // 计算广播后的形状
    const size_t n_depth = ReduceShapesBroadcast(ret_shape, l_shape, r_shape, depth_offset);

    // 计算子尺寸
    const std::vector<int>& ret_child_sizes = ComputeChildSizes(ret_shape);
    const std::vector<int>& l_child_sizes = ComputeChildSizes(l_shape);
    const std::vector<int>& r_child_sizes = ComputeChildSizes(r_shape);

    // 计算步长
    std::vector<int> l_steps, r_steps;
    l_steps.reserve(n_depth);
    r_steps.reserve(n_depth);
    for (size_t depth = 0; depth < n_depth; depth++) {
        const int& l_s = l_shape[depth];
        const int& r_s = r_shape[depth];
        const int l_step = (l_s == r_s || r_s == 1) ? l_child_sizes[depth] : 0;
        const int r_step = (l_s == r_s || l_s == 1) ? r_child_sizes[depth] : 0;
        l_steps.push_back(l_step);
        r_steps.push_back(r_step);
    }

    // 并行执行
    RunParallel(ret_shape[0], [&](int i) {
        const int ret_size = static_cast<int>(ret.size()) / ret_shape[0];
        ApplyOpBroadcastImpl(
            static_cast<char*>(ret.data_ptr()) + ret_child_sizes[0] * i * type_size,
            static_cast<const char*>(lhs.data_ptr()) + l_steps[0] * i * type_size,
            static_cast<const char*>(rhs.data_ptr()) + r_steps[0] * i * type_size,
            ret_shape, ret_size,
            l_steps, r_steps, 1, n_depth, ret_step, op, dtype, type_size
        );
    });
}

// 包装二元操作用于指针接口
// inline BinaryElementOp WrapBinaryOp(const BinaryElementOp& op) {
//     return op;
// }

template <typename F>
inline void ApplyOpSimple(NdArray& ret, F op_func) {
    DataType dtype = ret.dtype();
    size_t type_size = NdArray::get_type_size(dtype);
    char* ret_data = static_cast<char*>(ret.data_ptr());
    size_t size = ret.size();
    
    RunParallel(static_cast<int>(size), [&](int i) {
        op_func(ret_data + i * type_size, dtype);
    });
}

// 一元操作
template <typename F>
inline void ApplyOpSimple(NdArray& ret, const NdArray& src, F op_func) {
    if (ret.shape() != src.shape()) {
        throw std::runtime_error("Shapes must match for unary operation");
    }
    
    DataType dtype = ret.dtype();
    if (dtype != src.dtype()) {
        throw std::runtime_error("Data types must match for unary operation");
    }
    
    size_t type_size = NdArray::get_type_size(dtype);
    char* ret_data = static_cast<char*>(ret.data_ptr());
    const char* src_data = static_cast<const char*>(src.data_ptr());
    size_t size = ret.size();
    
    RunParallel(static_cast<int>(size), [&](int i) {
        op_func(ret_data + i * type_size, src_data + i * type_size);
    });
}

// 二元操作
template <typename F>
inline void ApplyOpSimple(NdArray& ret, const NdArray& lhs, const NdArray& rhs, F op_func) {
    if (ret.shape() != lhs.shape() || ret.shape() != rhs.shape()) {
        throw std::runtime_error("Shapes must match for binary operation");
    }
    
    DataType dtype = ret.dtype();
    if (dtype != lhs.dtype() || dtype != rhs.dtype()) {
        throw std::runtime_error("Data types must match for binary operation");
    }
    
    size_t type_size = NdArray::get_type_size(dtype);
    char* ret_data = static_cast<char*>(ret.data_ptr());
    const char* l_data = static_cast<const char*>(lhs.data_ptr());
    const char* r_data = static_cast<const char*>(rhs.data_ptr());
    size_t size = ret.size();
    
    RunParallel(static_cast<int>(size), [&](int i) {
        op_func(ret_data + i * type_size, l_data + i * type_size, r_data + i * type_size);
    });
}

// 标量操作（模板类型）
template <typename T, typename F>
inline void ApplyOpSimple(NdArray& ret, const NdArray& lhs, T scalar, F op_func) {
    if (ret.shape() != lhs.shape()) {
        throw std::runtime_error("Shapes must match for scalar operation");
    }
    
    DataType dtype = ret.dtype();
    if (dtype != lhs.dtype()) {
        throw std::runtime_error("Data types must match for scalar operation");
    }
    
    size_t type_size = NdArray::get_type_size(dtype);
    char* ret_data = static_cast<char*>(ret.data_ptr());
    const char* l_data = static_cast<const char*>(lhs.data_ptr());
    size_t size = ret.size();
    
    // 将标量转换为void指针
    T scalar_value = scalar;
    const void* scalar_ptr = &scalar_value;
    
    RunParallel(static_cast<int>(size), [&](int i) {
        op_func(ret_data + i * type_size, l_data + i * type_size, scalar_ptr);
    });
}

template <typename F>
NdArray ApplySingleOp(const NdArray& x, F op) {
    NdArray ret(x.dtype(), x.shape());
    ApplyOpSimple(ret, x, op);
    return ret;
}

template <typename F>
NdArray ApplySingleOpInplace(NdArray&& x, F op) {
    ApplyOpSimple(x, x, op);
    return std::move(x);
}

// 通用的二元操作函数
template <typename F>
NdArray ApplyDualOp(const NdArray& lhs, const NdArray& rhs, F op) {
    if (lhs.shape() == rhs.shape()) {
        NdArray ret(lhs.dtype(), lhs.shape());
        ApplyOpSimple(ret, lhs, rhs, op);
        return ret;
    } else {
        Shape ret_shape = CheckBroadcastable(lhs.shape(), rhs.shape());
        NdArray ret(lhs.dtype(), ret_shape);
        ApplyOpBroadcast(ret, lhs, rhs, 0, 1, op);
        return ret;
    }
}

} // namespace tinyndarray

#endif // TINYNDARRAY_OPS_IMPL_H