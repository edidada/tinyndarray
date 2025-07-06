#include "tinyndarray.h"
#include "tinyndarray_ops_impl.h"
#include <functional>
#include <cstring>

namespace tinyndarray {
// 加法操作
static float AddOp(float a, float b) { return a + b; }
static int AddOp(int a, int b) { return a + b; }
static double AddOp(double a, double b) { return a + b; }

// 减法操作
static float SubtractOp(float a, float b) { return a - b; }
static int SubtractOp(int a, int b) { return a - b; }
static double SubtractOp(double a, double b) { return a - b; }

// 乘法操作
static float MultiplyOp(float a, float b) { return a * b; }
static int MultiplyOp(int a, int b) { return a * b; }
static double MultiplyOp(double a, double b) { return a * b; }

// 除法操作
static float DivideOp(float a, float b) { 
    if (b == 0) throw std::runtime_error("Division by zero");
    return a / b; 
}
static int DivideOp(int a, int b) { 
    if (b == 0) throw std::runtime_error("Division by zero");
    return a / b; 
}
static double DivideOp(double a, double b) { 
    if (b == 0) throw std::runtime_error("Division by zero");
    return a / b; 
}

template <typename T>
void add_op(void* out, const void* a, const void* b) {
    T* out_typed = static_cast<T*>(out);
    const T* a_typed = static_cast<const T*>(a);
    const T* b_typed = static_cast<const T*>(b);
    *out_typed = *a_typed + *b_typed;
}
// 算术操作实现 - 数组与数组
// 加法操作的实现
NdArray Add(const NdArray& lhs, const NdArray& rhs) {
    BinaryElementOp typed_add_op;
    switch(lhs.dtype()) {
         case DataType::FLOAT: {
                typed_add_op = add_op<float>;
                break;
            }
            case DataType::INT: {
                typed_add_op = add_op<int>;
                break;                
            }
            case DataType::DOUBLE: {
                typed_add_op = add_op<double>;
                break;
            }
            default:
                throw std::runtime_error("Unsupported data type for Add");
    }
    
    return ApplyDualOp(lhs, rhs, typed_add_op);
}

template <typename T>
void sub_op(void* out, const void* a, const void* b) {
    T* out_typed = static_cast<T*>(out);
    const T* a_typed = static_cast<const T*>(a);
    const T* b_typed = static_cast<const T*>(b);
    *out_typed = *a_typed - *b_typed;
}

NdArray Subtract(const NdArray& lhs, const NdArray& rhs) {
    BinaryElementOp typed_add_op;
    switch(lhs.dtype()) {
         case DataType::FLOAT: {
                typed_add_op = add_op<float>;
                break;
            }
            case DataType::INT: {
                typed_add_op = add_op<int>;
                break;                
            }
            case DataType::DOUBLE: {
                typed_add_op = add_op<double>;
                break;
            }
            default:
                throw std::runtime_error("Unsupported data type for Add");
    }
    
    return ApplyDualOp(lhs, rhs, typed_add_op);
}

// 算术运算符实现
NdArray operator+(const NdArray& lhs, const NdArray& rhs) {
    return Add(lhs, rhs);
}

NdArray operator-(const NdArray& lhs, const NdArray& rhs) {
    // 类似实现...
}

NdArray operator*(const NdArray& lhs, const NdArray& rhs) {
    // 类似实现...
}

NdArray operator/(const NdArray& lhs, const NdArray& rhs) {
    // 类似实现...
}

template <typename T>
void neg_op(void* out, const void* a) {
    T* out_typed = static_cast<T*>(out);
    const T* a_typed = static_cast<const T*>(a);
    *out_typed = -*a_typed;
}

NdArray Neg(const NdArray& x) {
    UnaryElementOp typed_neg_op;
    switch(x.dtype()) {
         case DataType::FLOAT: {
                typed_neg_op = neg_op<float>;
                break;
            }
            case DataType::INT: {
                typed_neg_op = neg_op<int>;
                break;                
            }
            case DataType::DOUBLE: {
                typed_neg_op = neg_op<double>;
                break;
            }
            default:
                throw std::runtime_error("Unsupported data type for Add");
    }
    
    return ApplySingleOp(x, typed_neg_op);
}

NdArray operator-(const NdArray& x) {
    return Neg(x);
}

// 点积运算（优化版本）
NdArray NdArray::dot(const NdArray& other) const {
    if (dtype() != other.dtype()) {
        throw std::runtime_error("Data type mismatch for dot product");
    }
    
    if (ndim() != 1 || other.ndim() != 1 || size() != other.size()) {
        throw std::runtime_error("Only 1D arrays of same size supported for dot product");
    }
    
    NdArray result(dtype(), {1});
    const int n = static_cast<int>(size());
    const void* this_ptr = data_ptr();
    const void* other_ptr = other.data_ptr();
    void* result_ptr = result.data_ptr();
    
    switch (dtype()) {
        case DataType::FLOAT: {
            const float* this_data = static_cast<const float*>(this_ptr);
            const float* other_data = static_cast<const float*>(other_ptr);
            float sum = 0.0f;
            for (int i = 0; i < n; ++i) {
                sum += this_data[i] * other_data[i];
            }
            *static_cast<float*>(result_ptr) = sum;
            break;
        }
        case DataType::INT: {
            const int* this_data = static_cast<const int*>(this_ptr);
            const int* other_data = static_cast<const int*>(other_ptr);
            int sum = 0;
            for (int i = 0; i < n; ++i) {
                sum += this_data[i] * other_data[i];
            }
            *static_cast<int*>(result_ptr) = sum;
            break;
        }
        case DataType::DOUBLE: {
            const double* this_data = static_cast<const double*>(this_ptr);
            const double* other_data = static_cast<const double*>(other_ptr);
            double sum = 0.0;
            for (int i = 0; i < n; ++i) {
                sum += this_data[i] * other_data[i];
            }
            *static_cast<double*>(result_ptr) = sum;
            break;
        }
        default:
            throw std::runtime_error("Unsupported data type for dot product");
    }
    
    return result;
}

// 定义归约操作函数指针类型
using ReduceOpFunc = void(*)(void* accum, const void* value);

// 具体数据类型归约操作的实现
namespace reduce_ops {
    template <typename T, typename Op>
    void reduce_impl(void* accum, const void* value) {
        T* accum_ptr = static_cast<T*>(accum);
        const T* value_ptr = static_cast<const T*>(value);
        *accum_ptr = Op()(*accum_ptr, *value_ptr);
    }

    // 加法归约
    void add_float(void* accum, const void* value) {
        reduce_impl<float, std::plus<float>>(accum, value);
    }
    
    void add_int(void* accum, const void* value) {
        reduce_impl<int, std::plus<int>>(accum, value);
    }
    
    void add_double(void* accum, const void* value) {
        reduce_impl<double, std::plus<double>>(accum, value);
    }
    
    // 最小值归约
    void min_float(void* accum, const void* value) {
        float* accum_ptr = static_cast<float*>(accum);
        const float* value_ptr = static_cast<const float*>(value);
        *accum_ptr = std::min(*accum_ptr, *value_ptr);
    }
    
    void min_int(void* accum, const void* value) {
        int* accum_ptr = static_cast<int*>(accum);
        const int* value_ptr = static_cast<const int*>(value);
        *accum_ptr = std::min(*accum_ptr, *value_ptr);
    }
    
    void min_double(void* accum, const void* value) {
        double* accum_ptr = static_cast<double*>(accum);
        const double* value_ptr = static_cast<const double*>(value);
        *accum_ptr = std::min(*accum_ptr, *value_ptr);
    }
    
    // 最大值归约
    void max_float(void* accum, const void* value) {
        float* accum_ptr = static_cast<float*>(accum);
        const float* value_ptr = static_cast<const float*>(value);
        *accum_ptr = std::max(*accum_ptr, *value_ptr);
    }
    
    void max_int(void* accum, const void* value) {
        int* accum_ptr = static_cast<int*>(accum);
        const int* value_ptr = static_cast<const int*>(value);
        *accum_ptr = std::max(*accum_ptr, *value_ptr);
    }
    
    void max_double(void* accum, const void* value) {
        double* accum_ptr = static_cast<double*>(accum);
        const double* value_ptr = static_cast<const double*>(value);
        *accum_ptr = std::max(*accum_ptr, *value_ptr);
    }
}

static Shape CheckReductable(const Shape& shape, const Axis& axes,
                             bool keepdims) {
    // Mark reduction axes
    std::vector<char> mark(shape.size(), false);
    const int n_shape = static_cast<int>(shape.size());
    for (auto&& axis : axes) {
        if (0 <= axis && axis < n_shape) {
            mark[static_cast<size_t>(axis)] = true;
        } else {
            throw std::runtime_error("Invalid axes for reduction");
        }
    }

    if (keepdims) {
        // Pick up unmarked dimension
        Shape ret_shape_pad;
        for (size_t i = 0; i < mark.size(); i++) {
            if (mark[i]) {
                ret_shape_pad.push_back(1);
            } else {
                ret_shape_pad.push_back(shape[i]);
            }
        }
        return ret_shape_pad;
    } else {
        // No necessary
        return {};
    }
}

static auto ComputeReduceSizes(const Shape& src_shape, const size_t axis) {
    // Compute result shape
    Shape ret_shape;
    for (size_t dim = 0; dim < src_shape.size(); dim++) {
        if (dim != axis) {
            ret_shape.push_back(src_shape[dim]);
        }
    }
    if (ret_shape.empty()) {  // For all reduction
        ret_shape.push_back(1);
    }

    // Compute sizes
    int n_upper = 1, n_lower = 1, n_reduce = 0;
    for (size_t dim = 0; dim < src_shape.size(); dim++) {
        // Sizes
        if (dim < axis) {
            n_upper *= src_shape[dim];
        } else if (axis < dim) {
            n_lower *= src_shape[dim];
        } else {
            n_reduce = src_shape[dim];
        }
    }

    // Return
    return std::make_tuple(std::move(ret_shape), n_upper, n_lower, n_reduce);
}


// 获取归约操作函数
ReduceOpFunc get_reduce_op(DataType dtype, ReduceType reduce_type) {
    switch (dtype) {
        case DataType::FLOAT:
            switch (reduce_type) {
                case ReduceType::SUM: return reduce_ops::add_float;
                case ReduceType::MIN: return reduce_ops::min_float;
                case ReduceType::MAX: return reduce_ops::max_float;
                default: throw std::runtime_error("Unsupported reduce type");
            }
        case DataType::INT:
            switch (reduce_type) {
                case ReduceType::SUM: return reduce_ops::add_int;
                case ReduceType::MIN: return reduce_ops::min_int;
                case ReduceType::MAX: return reduce_ops::max_int;
                default: throw std::runtime_error("Unsupported reduce type");
            }
        case DataType::DOUBLE:
            switch (reduce_type) {
                case ReduceType::SUM: return reduce_ops::add_double;
                case ReduceType::MIN: return reduce_ops::min_double;
                case ReduceType::MAX: return reduce_ops::max_double;
                default: throw std::runtime_error("Unsupported reduce type");
            }
        default:
            throw std::runtime_error("Unsupported data type for reduction");
    }
}

// 归约操作的初始值
void* get_init_value(DataType dtype, ReduceType reduce_type) {
    static float f_init;
    static int i_init;
    static double d_init;
    
    switch (dtype) {
        case DataType::FLOAT:
            switch (reduce_type) {
                case ReduceType::SUM: f_init = 0.0f; return &f_init;
                case ReduceType::MIN: f_init = std::numeric_limits<float>::max(); return &f_init;
                case ReduceType::MAX: f_init = std::numeric_limits<float>::lowest(); return &f_init;
                default: throw std::runtime_error("Unsupported reduce type");
            }
        case DataType::INT:
            switch (reduce_type) {
                case ReduceType::SUM: i_init = 0; return &i_init;
                case ReduceType::MIN: i_init = std::numeric_limits<int>::max(); return &i_init;
                case ReduceType::MAX: i_init = std::numeric_limits<int>::lowest(); return &i_init;
                default: throw std::runtime_error("Unsupported reduce type");
            }
        case DataType::DOUBLE:
            switch (reduce_type) {
                case ReduceType::SUM: d_init = 0.0; return &d_init;
                case ReduceType::MIN: d_init = std::numeric_limits<double>::max(); return &d_init;
                case ReduceType::MAX: d_init = std::numeric_limits<double>::lowest(); return &d_init;
                default: throw std::runtime_error("Unsupported reduce type");
            }
        default:
            throw std::runtime_error("Unsupported data type for reduction");
    }
}

// 全轴归约实现
NdArray ReduceAxisAll(const NdArray& src, ReduceType reduce_type) {
    DataType dtype = src.dtype();
    size_t type_size = NdArray::get_type_size(dtype);
    const char* src_data = static_cast<const char*>(src.data_ptr());
    size_t size = src.size();
    
    // 获取归约操作函数和初始值
    ReduceOpFunc reduce_op = get_reduce_op(dtype, reduce_type);
    void* init_value = get_init_value(dtype, reduce_type);
    
    // 创建累加器并初始化
    char* accum = new char[type_size];
    memcpy(accum, init_value, type_size);
    
    // 执行归约
    for (size_t i = 0; i < size; i++) {
        reduce_op(accum, src_data + i * type_size);
    }
    
    // 创建结果数组
    NdArray result(dtype, {1});
    memcpy(result.data_ptr(), accum, type_size);
    delete[] accum;
    
    return result;
}

// 单轴归约实现
NdArray ReduceAxisOne(const NdArray& src, size_t axis, ReduceType reduce_type) {
    const Shape& src_shape = src.shape();
    DataType dtype = src.dtype();
    size_t type_size = NdArray::get_type_size(dtype);
    const char* src_data = static_cast<const char*>(src.data_ptr());
    
    // 计算归约尺寸
    auto reduce_sizes = ComputeReduceSizes(src_shape, axis);
    const Shape& ret_shape = std::get<0>(reduce_sizes);
    const int n_upper = std::get<1>(reduce_sizes);
    const int n_lower = std::get<2>(reduce_sizes);
    const int n_reduce = std::get<3>(reduce_sizes);

    // 创建结果数组
    NdArray ret(dtype, ret_shape);
    char* ret_data = static_cast<char*>(ret.data_ptr());
    
    // 获取归约操作函数和初始值
    ReduceOpFunc reduce_op = get_reduce_op(dtype, reduce_type);
    void* init_value = get_init_value(dtype, reduce_type);
    
    // 归约函数
    auto reduce = [=](int u_idx) {
        const int ret_idx_base = u_idx * n_lower;
        const int src_idx_base0 = u_idx * n_reduce * n_lower;
        
        for (int l_idx = 0; l_idx < n_lower; l_idx++) {
            // 初始化累加器
            char accum[type_size];
            memcpy(accum, init_value, type_size);
            
            for (int redu_idx = 0; redu_idx < n_reduce; redu_idx++) {
                const int src_idx = src_idx_base0 + redu_idx * n_lower + l_idx;
                const char* value_ptr = src_data + src_idx * type_size;
                
                // 应用归约操作
                reduce_op(accum, value_ptr);
            }
            
            // 存储结果
            char* result_ptr = ret_data + (ret_idx_base + l_idx) * type_size;
            memcpy(result_ptr, accum, type_size);
        }
    };

    // 并行执行
    RunParallel(n_upper, reduce);
    
    return ret;
}

// 通用归约函数
NdArray ReduceAxis(const NdArray& src, const Axis& axes_raw, bool keepdims, ReduceType reduce_type) {
    if (axes_raw.size() == 0) {
        // 全轴归约
        NdArray ret = ReduceAxisAll(src, reduce_type);
        
        // 保持维度
        if (keepdims) {
            Shape ret_shape(src.shape().size(), 1);
            ret = ret.reshape(ret_shape);
        }
        return ret;
    } else {
        // 解析轴
        const Axis& axes = ResolveAxis(axes_raw, src.ndim(), "Reduce", true);
        
        // 检查可归约性
        Shape src_shape = src.shape();
        const Shape& ret_shape_pad = CheckReductable(src_shape, axes, keepdims);
        
        // 逐个轴归约
        NdArray ret = src;
        for (size_t i = 0; i < axes.size(); i++) {
            size_t axis = static_cast<size_t>(axes[axes.size() - i - 1]);
            ret = ReduceAxisOne(ret, axis, reduce_type);
        }
        
        // 保持维度
        if (keepdims) {
            ret = ret.reshape(ret_shape_pad);
        }
        return ret;
    }
}

// 具体归约操作
NdArray Sum(const NdArray& x, const Axis& axes, bool keepdims) {
    return ReduceAxis(x, axes, keepdims, ReduceType::SUM);
}

NdArray Min(const NdArray& x, const Axis& axes, bool keepdims) {
    return ReduceAxis(x, axes, keepdims, ReduceType::MIN);
}

NdArray Max(const NdArray& x, const Axis& axes, bool keepdims) {
    return ReduceAxis(x, axes, keepdims, ReduceType::MAX);
}

NdArray NdArray::sum(const Axis& axes, bool keepdims) const {
    return Sum(*this, axes, keepdims);
}

// NdArray Mean(const NdArray& x, const Axis& axes, bool keepdims) {
//     // 计算和
//     NdArray sum_arr = Sum(x, axes, keepdims);
    
//     // 计算元素数量
//     size_t reduce_size = 1;
//     if (axes.size() == 0) {
//         reduce_size = x.size();
//     } else {
//         Shape shape = x.shape();
//         for (int axis : axes) {
//             int resolved_axis = axis < 0 ? shape.size() + axis : axis;
//             reduce_size *= shape[resolved_axis];
//         }
//     }
    
//     // 转换为浮点数计算平均值
//     NdArray mean_arr = sum_arr / static_cast<float>(reduce_size);
    
//     return mean_arr;
// }

} // namespace tinyndarray