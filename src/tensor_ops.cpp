#include "tinyndarray.h"
#include <algorithm>
#include <cstring>
#include <sstream>
namespace tinyndarray {

// 堆叠
NdArray Stack(const std::vector<NdArray>& xs, int axis) {
    if (xs.empty()) {
        throw std::invalid_argument("Cannot stack empty list of arrays");
    }
    
    // 验证所有数组形状相同
    const Shape& ref_shape = xs[0].shape();
    DataType ref_dtype = xs[0].dtype();
    for (const auto& x : xs) {
        if (x.shape() != ref_shape || x.dtype() != ref_dtype) {
            throw std::invalid_argument("All arrays must have same shape and dtype for stacking");
        }
    }
    
    // 解析轴
    if (axis < 0) {
        axis = ref_shape.size() + 1 + axis;
    }
    if (axis < 0 || axis > ref_shape.size()) {
        throw std::invalid_argument("Invalid axis for stacking");
    }
    
    // 创建新形状
    Shape new_shape = ref_shape;
    new_shape.insert(new_shape.begin() + axis, xs.size());
    
    // 创建新数组
    NdArray result(ref_dtype, new_shape);
    const size_t element_size = NdArray::get_type_size(ref_dtype);
    const size_t n_elements_per_array = xs[0].size();
    const size_t total_bytes_per_array = n_elements_per_array * element_size;
    
    // 计算目标偏移
    const std::vector<size_t> strides = compute_strides(new_shape, element_size);
    size_t stack_stride = strides[axis];
    
    // 并行复制数据
    RunParallel(static_cast<int>(xs.size()), [&](int i) {
        const void* src = xs[i].data_ptr();
        void* dst = static_cast<char*>(result.data_ptr()) + i * stack_stride;
        std::memcpy(dst, src, total_bytes_per_array);
    });
    
    return result;
}
// 连接
int ResolveAxis(int axis, size_t ndim, const std::string& name) {
    // Resolve negative
    const int ndim_i = static_cast<int>(ndim);
    if (axis < 0) {
        axis = ndim_i + axis;
    }
    // Check range
    if (axis < 0 || ndim_i <= axis) {
        std::stringstream ss;
        ss << "Invalid axes for " << name;
        ss << " (" << ndim << "vs" << axis << ")";
        throw std::runtime_error(ss.str());
    }
    return axis;
}

Axis ResolveAxis(const Axis& axes, size_t ndim, const std::string& name,
                        bool sort, bool sort_order_normal) {
    // Resolve for each
    Axis ret_axes;
    ret_axes.reserve(axes.size());
    for (auto&& axis : axes) {
        ret_axes.push_back(ResolveAxis(axis, ndim, name));
    }
    // Sort axes
    if (sort) {
        if (sort_order_normal) {
            // Normal order
            std::sort(ret_axes.begin(), ret_axes.end());
        } else {
            // Inverse order
            std::sort(ret_axes.begin(), ret_axes.end(), std::greater<int>());
        }
    }
    return ret_axes;
}
static void CheckConcatenatable(const std::vector<NdArray>& xs, int axis) {
    // Check empty
    if (xs.empty()) {
        throw std::runtime_error("Need at least one array to concatenate");
    }
    // Check same shape except axis dimension
    const Shape& fst_shape = xs[0].shape();
    const size_t axis_l = static_cast<size_t>(axis);
    const std::string error_msg = "all the input array dimensions except for "
                                  "the concatenation axis must match exactly";
    for (size_t i = 1; i < xs.size(); i++) {
        const Shape& cur_shape = xs[i].shape();
        // Check the size of shapes
        if (fst_shape.size() != cur_shape.size()) {
            throw std::runtime_error(error_msg);
        }
        // Check dimensions except axis
        for (size_t j = 0; j < fst_shape.size(); j++) {
            if (j == axis_l) {
                continue;
            }
            if (fst_shape[j] != cur_shape[j]) {
                throw std::runtime_error(error_msg);
            }
        }
    }
}

static auto ComputeConcatSizes(const std::vector<NdArray>& xs, int axis) {
    const Shape& fst_shape = xs[0].shape();
    const auto& src_s_iter0 = fst_shape.begin();
    const auto& src_s_iter1 = fst_shape.begin() + axis;
    const auto& src_s_iter2 = fst_shape.begin() + axis + 1;
    const auto& src_s_iter3 = fst_shape.end();
    const auto& mul = std::multiplies<int>();

    // Upper common size
    const int n_upper = std::accumulate(src_s_iter0, src_s_iter1, 1, mul);
    // Lower size depends on each sources
    std::vector<int> n_lowers;
    for (auto&& x : xs) {
        n_lowers.push_back(static_cast<int>(x.size()) / n_upper);
    }
    // Result indices of concatenation
    std::vector<int> concat_offsets;
    int n_lower_accum = 0;
    for (auto&& n_lower : n_lowers) {
        concat_offsets.push_back(n_lower_accum);
        n_lower_accum += n_lower;
    }
    // Step of concatenating dimension
    const int concat_step = n_lower_accum;

    // Concatenating dimensions
    int concat_dim = 0;
    const size_t axis_l = static_cast<size_t>(axis);
    for (auto&& x : xs) {
        concat_dim += x.shape()[axis_l];
    }

    // Create result shape
    Shape ret_shape;
    ret_shape.insert(ret_shape.end(), src_s_iter0, src_s_iter1);  // Upper
    ret_shape.push_back(concat_dim);  // Concatenating dimension
    ret_shape.insert(ret_shape.end(), src_s_iter2, src_s_iter3);  // Lower

    return std::make_tuple(std::move(ret_shape), n_upper, std::move(n_lowers),
                           xs.size(), std::move(concat_offsets), concat_step);
}

NdArray Concatenate(const std::vector<NdArray>& xs, int axis) {
    // 解析轴
    axis = ResolveAxis(axis, xs[0].ndim(), "Concatenate");
    
    // 验证可连接性
    CheckConcatenatable(xs, axis);
    
    // 计算连接参数
    auto concat_sizes = ComputeConcatSizes(xs, axis);
    const Shape& ret_shape = std::get<0>(concat_sizes);
    const int n_upper = std::get<1>(concat_sizes);
    const std::vector<int>& n_lowers = std::get<2>(concat_sizes);
    const size_t n_concat = std::get<3>(concat_sizes);
    const std::vector<int>& concat_offsets = std::get<4>(concat_sizes);
    const int concat_step = std::get<5>(concat_sizes);
    
    // 创建结果数组
    NdArray ret = NdArray::Empty(xs[0].dtype(), ret_shape);
    const size_t element_size = NdArray::get_type_size(ret.dtype());
    
    // 获取目标指针
    char* ret_data = static_cast<char*>(ret.data_ptr());
    
    // 并行复制数据
    if (n_upper == 1) {
        // 情况1：连接轴是最高维（最外层）
        RunParallel(static_cast<int>(n_concat), [&](int concat_idx) {
            const NdArray& src_arr = xs[concat_idx];
            const size_t bytes_to_copy = static_cast<size_t>(n_lowers[concat_idx]) * element_size;
            const char* src_data = static_cast<const char*>(src_arr.data_ptr());
            char* dst_data = ret_data + concat_offsets[concat_idx] * element_size;
            
            std::memcpy(dst_data, src_data, bytes_to_copy);
        });
    } else {
        // 情况2：连接轴在中间维度
        RunParallel(n_upper, [&](int u_idx) {
            const size_t base_offset = static_cast<size_t>(u_idx) * concat_step * element_size;
            
            for (size_t concat_idx = 0; concat_idx < n_concat; concat_idx++) {
                const NdArray& src_arr = xs[concat_idx];
                const size_t bytes_to_copy = static_cast<size_t>(n_lowers[concat_idx]) * element_size;
                const size_t src_offset = static_cast<size_t>(u_idx) * n_lowers[concat_idx] * element_size;
                const size_t dst_offset = base_offset + concat_offsets[concat_idx] * element_size;
                
                const char* src_data = static_cast<const char*>(src_arr.data_ptr()) + src_offset;
                char* dst_data = ret_data + dst_offset;
                
                std::memcpy(dst_data, src_data, bytes_to_copy);
            }
        });
    }
    
    return ret;
}
// 分割
// ======================= 分割操作函数 =======================

static std::vector<NdArray> SplitImpl(const NdArray& x, const Index& idxs, int axis) {
    const size_t axis_l = static_cast<size_t>(axis);
    const Shape& x_shape = x.shape();
    const int idx_end = x_shape[axis_l];
    const size_t element_size = NdArray::get_type_size(x.dtype());
    
    // 计算高层维度大小
    int n_upper = 1;
    for (size_t i = 0; i < axis_l; i++) {
        n_upper *= x_shape[i];
    }
    
    // 计算低层维度大小
    int n_lower = 1;
    for (size_t i = axis_l + 1; i < x_shape.size(); i++) {
        n_lower *= x_shape[i];
    }
    
    // 创建结果数组
    std::vector<NdArray> results;
    int prev_idx = 0;
    const char* src_base = static_cast<const char*>(x.data_ptr());
    
    // 计算分割点（包含最后一段）
    std::vector<int> split_points = idxs;
    split_points.push_back(idx_end);
    
    for (int curr_idx : split_points) {
        // 计算当前段的长度
        const int seg_length = curr_idx - prev_idx;
        
        // 创建子数组形状
        Shape sub_shape = x_shape;
        sub_shape[axis_l] = seg_length;
        
        // 创建空结果数组
        NdArray sub_arr = NdArray::Empty(x.dtype(), sub_shape);
        results.push_back(sub_arr);
        
        // 计算每段需要复制的字节数
        const size_t seg_bytes = static_cast<size_t>(seg_length) * n_lower * element_size;
        
        // 计算源和目标偏移
        const size_t src_offset = static_cast<size_t>(prev_idx) * n_lower * element_size;
        char* dst_base = static_cast<char*>(sub_arr.data_ptr());
        
        // 并行复制每个高层维度的数据
        RunParallel(n_upper, [&](int u_idx) {
            const size_t base_offset = static_cast<size_t>(u_idx) * x_shape[axis_l] * n_lower * element_size;
            const char* src_ptr = src_base + base_offset + src_offset;
            char* dst_ptr = dst_base + static_cast<size_t>(u_idx) * seg_bytes;
            
            std::memcpy(dst_ptr, src_ptr, seg_bytes);
        });
        
        prev_idx = curr_idx;
    }
    
    return results;
}

std::vector<NdArray> Split(const NdArray& x, int n_section, int axis) {
    // 解析轴
    axis = ResolveAxis(axis, x.ndim(), "Split");
    
    // 检查分割数量
    const int dim_size = x.shape()[static_cast<size_t>(axis)];
    if (dim_size % n_section != 0) {
        throw std::runtime_error("Dimension size must be divisible by number of sections");
    }
    
    // 计算分割点
    const int section_size = dim_size / n_section;
    Index idxs;
    for (int sec_i = 1; sec_i < n_section; sec_i++) {
        idxs.push_back(section_size * sec_i);
    }
    
    return SplitImpl(x, idxs, axis);
}

std::vector<NdArray> Split(const NdArray& x, const Index& idxs, int axis) {
    // 解析轴
    axis = ResolveAxis(axis, x.ndim(), "Split");
    return SplitImpl(x, idxs, axis);
}

// 压缩维度
NdArray Squeeze(const NdArray& x, const Axis& axes) {
    // 实现与原始SqueezeNdArray类似，但使用直接内存操作
    Shape new_shape;
    const Shape& old_shape = x.shape();
    
    if (axes.empty()) {
        // 移除所有长度为1的维度
        for (int dim : old_shape) {
            if (dim != 1) {
                new_shape.push_back(dim);
            }
        }
        if (new_shape.empty()) {
            new_shape.push_back(1);
        }
    } else {
        // 移除指定维度
        new_shape = old_shape;
        // 从后往前移除，避免索引变化
        std::vector<int> sorted_axes = axes;
        std::sort(sorted_axes.rbegin(), sorted_axes.rend());
        for (int axis : sorted_axes) {
            if (axis < 0 || axis >= new_shape.size()) {
                throw std::invalid_argument("Invalid axis in Squeeze");
            }
            if (new_shape[axis] != 1) {
                throw std::runtime_error("Can only squeeze dimensions of size 1");
            }
            new_shape.erase(new_shape.begin() + axis);
        }
    }
    
    // 创建新数组（共享数据）
    return NdArray(x.dtype(), new_shape, x.data_ptr());
}

NdArray ExpandDims(const NdArray& x, int axis) {
    // 解析轴
    if (axis < 0) {
        axis = x.ndim() + 1 + axis;
    }
    if (axis < 0 || axis > x.ndim()) {
        throw std::invalid_argument("Invalid axis in ExpandDims");
    }
    
    // 创建新形状
    Shape new_shape = x.shape();
    new_shape.insert(new_shape.begin() + axis, 1);
    
    // 创建新数组（共享数据）
    return NdArray(x.dtype(), new_shape, x.data_ptr());
}
} // namespace tinyndarray