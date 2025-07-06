#include "tinyndarray.h"
#include <algorithm>
#include <numeric>
#include <cstring>

namespace tinyndarray {
// 辅助函数：计算元素偏移量
size_t calculate_offset(const Index& index, const Shape& shape, const std::vector<size_t>& strides) {
    size_t offset = 0;
    for (size_t i = 0; i < index.size(); i++) {
        if (index[i] < 0 || index[i] >= shape[i]) {
            throw std::out_of_range("Index out of range");
        }
        offset += index[i] * strides[i];
    }
    return offset;
}

// 辅助函数：计算步长
std::vector<size_t> compute_strides(const Shape& shape, size_t element_size) {
    std::vector<size_t> strides(shape.size());
    size_t stride = element_size;
    for (int i = shape.size() - 1; i >= 0; i--) {
        strides[i] = stride;
        stride *= shape[i];
    }
    return strides;
}

// 裁剪操作
static int ClipOp(int value, int min_val, int max_val) {
    return std::min(std::max(value, min_val), max_val - 1);
}

// 切片操作
// 计算切片偏移量
template <bool IsPrev>
static std::vector<int> ComputeSliceOffset(const std::vector<int>& child_sizes,
                                           const SliceIndex& slice_index,
                                           const Shape& src_shape) {
    std::vector<int> offsets;
    offsets.reserve(child_sizes.size());
    
    for (size_t depth = 0; depth < child_sizes.size(); depth++) {
        const auto& si = slice_index[depth];
        const int len = IsPrev ? si.first : (src_shape[depth] - si.second);
        offsets.push_back(child_sizes[depth] * len);
    }
    
    return offsets;
}
// 切片复制实现
template <typename T>
static void CopySliceImpl(const T* src_data, T* ret_data, 
                          const Shape& ret_shape,
                          const std::vector<int>& prev_offsets,
                          const std::vector<int>& post_offsets,
                          const int src_step_top, const int ret_step_top) {
    const size_t n_depth = ret_shape.size();
    const int type_size = sizeof(T);

    // 并行处理顶层维度
    RunParallel(ret_shape[0], [&](int ret_top) {
        const int ret_idx_base = ret_top * ret_step_top;
        const int src_idx_base = ret_top * src_step_top + prev_offsets[0];
        
        // 用于跟踪当前深度的栈
        std::vector<int> ret_cnts(n_depth, 0);
        size_t depth = 1;  // 从第二维度开始
        int src_idx = 0;

        for (int ret_idx = 0; ret_idx < ret_step_top; ret_idx++) {
            // 向下遍历维度
            while (depth < n_depth) {
                src_idx += prev_offsets[depth];
                depth++;
            }

            // 复制元素
            ret_data[ret_idx_base + ret_idx] = src_data[src_idx_base + src_idx];
            src_idx++;

            // 向上计数
            while (depth > 1) {
                depth--;
                const size_t prev_d = depth - 1;
                ret_cnts[prev_d]++;
                
                if (ret_cnts[prev_d] < ret_shape[prev_d]) {
                    break;  // 继续当前维度
                }
                
                // 重置计数器并应用后偏移
                ret_cnts[prev_d] = 0;
                src_idx += post_offsets[prev_d];
            }
        }
    });
}

// 切片复制主函数
static NdArray CopySlice(const NdArray& src, const Shape& ret_shape,
                         const SliceIndex& slice_index) {
    const Shape& src_shape = src.shape();
    const DataType dtype = src.dtype();
    const size_t type_size = NdArray::get_type_size(dtype);

    // 预计算子大小
    const std::vector<int> child_sizes = ComputeChildSizes(src_shape);
    
    // 预计算偏移量
    const std::vector<int> prev_offsets = 
        ComputeSliceOffset<true>(child_sizes, slice_index, src_shape);
    
    const std::vector<int> post_offsets = 
        ComputeSliceOffset<false>(child_sizes, slice_index, src_shape);

    // 计算顶层步长
    const int ret_step_top = std::accumulate(
        ret_shape.begin() + 1, ret_shape.end(), 1, std::multiplies<int>());
    
    const int src_step_top = child_sizes[0];

    // 创建切片数组
    NdArray result(dtype, ret_shape);
    const void* src_ptr = src.data_ptr();
    void* ret_ptr = result.data_ptr();

    // 根据数据类型分派
    switch (dtype) {
        case DataType::FLOAT:
            CopySliceImpl(
                static_cast<const float*>(src_ptr),
                static_cast<float*>(ret_ptr),
                ret_shape, prev_offsets, post_offsets,
                src_step_top, ret_step_top
            );
            break;
            
        case DataType::INT:
            CopySliceImpl(
                static_cast<const int*>(src_ptr),
                static_cast<int*>(ret_ptr),
                ret_shape, prev_offsets, post_offsets,
                src_step_top, ret_step_top
            );
            break;
            
        case DataType::DOUBLE:
            CopySliceImpl(
                static_cast<const double*>(src_ptr),
                static_cast<double*>(ret_ptr),
                ret_shape, prev_offsets, post_offsets,
                src_step_top, ret_step_top
            );
            break;
            
        default:
            throw std::runtime_error("Unsupported data type for slicing");
    }

    return result;
}

// 切片操作主函数
NdArray NdArray::slice(const SliceIndex& slice_index) const {
    const Shape& shape = m_sub->shape;
    const size_t ndim = shape.size();
    
    if (slice_index.size() != ndim) {
        throw std::runtime_error("Slice index dimension mismatch");
    }

    // 计算切片形状和新的正索引
    Shape slice_shape;
    SliceIndex new_index;
    
    for (size_t i = 0; i < ndim; i++) {
        const auto& si = slice_index[i];
        int s = (si.first >= 0) ? si.first : shape[i] + si.first;
        int e = (si.second >= 0) ? si.second : shape[i] + si.second;
        
        // 裁剪到有效范围
        s = ClipOp(s, 0, shape[i]);
        e = ClipOp(e, s + 1, shape[i] + 1);  // 确保e > s
        
        // 计算维度大小
        const int dim_size = std::max(e - s, 0);
        slice_shape.push_back(dim_size);
        new_index.push_back({s, e});
    }

    // 复制切片
    return CopySlice(*this, slice_shape, new_index);
}

// 将初始化列表转换为切片索引项
inline std::pair<int, int> CvtToSliceIndexItem(std::initializer_list<int> l) {
    if (l.size() != 2) {
        throw std::runtime_error("Invalid slice index format");
    }
    return {*l.begin(), *(l.begin() + 1)};
}

// 便捷切片接口
template <typename... I>
NdArray NdArray::slice(std::initializer_list<I>... slice_index) const {
    return slice(SliceIndex{CvtToSliceIndexItem(slice_index)...});
}

template NdArray NdArray::slice(std::initializer_list<int>, std::initializer_list<int>) const;

NdArray Transpose(const NdArray& x, const Axis& perm) {
    const Shape& shape = x.shape();
    const int ndim = shape.size();
    
    // 验证排列
    if (perm.size() != ndim) {
        throw std::invalid_argument("Permutation size must match array dimension");
    }
    
    // 创建新形状
    Shape new_shape(ndim);
    for (int i = 0; i < ndim; i++) {
        if (perm[i] < 0 || perm[i] >= ndim) {
            throw std::out_of_range("Invalid axis in permutation");
        }
        new_shape[i] = shape[perm[i]];
    }
    
    // 创建结果数组
    NdArray result(x.dtype(), new_shape);
    const size_t element_size = NdArray::get_type_size(x.dtype());
    const size_t total_elements = x.size();
    
    // 计算源步长和目标步长
    const std::vector<size_t> src_strides = compute_strides(shape, element_size);
    const std::vector<size_t> dst_strides = compute_strides(new_shape, element_size);
    
    // 创建逆排列
    std::vector<int> inv_perm(ndim);
    for (int i = 0; i < ndim; i++) {
        inv_perm[perm[i]] = i;
    }
    
    // 并行复制数据
    RunParallel(static_cast<int>(total_elements), [&](int linear_idx) {
        // 将线性索引转换为多维索引
        Index src_index(ndim);
        size_t remainder = linear_idx;
        for (int d = ndim - 1; d >= 0; d--) {
            src_index[d] = remainder % shape[d];
            remainder /= shape[d];
        }
        
        // 应用排列
        Index dst_index(ndim);
        for (int d = 0; d < ndim; d++) {
            dst_index[d] = src_index[perm[d]];
        }
        
        // 计算偏移量
        size_t src_offset = 0;
        size_t dst_offset = 0;
        for (int d = 0; d < ndim; d++) {
            src_offset += src_index[d] * src_strides[d];
            dst_offset += dst_index[d] * dst_strides[d];
        }
        
        // 复制元素
        const void* src_ptr = static_cast<const char*>(x.data_ptr()) + src_offset;
        void* dst_ptr = static_cast<char*>(result.data_ptr()) + dst_offset;
        std::memcpy(dst_ptr, src_ptr, element_size);
    });
    
    return result;
}

NdArray SwapAxes(const NdArray& x, int axis1, int axis2) {
    const Shape& shape = x.shape();
    const int ndim = shape.size();
    
    // 解析轴
    if (axis1 < 0) axis1 = ndim + axis1;
    if (axis2 < 0) axis2 = ndim + axis2;
    if (axis1 < 0 || axis1 >= ndim || axis2 < 0 || axis2 >= ndim) {
        throw std::invalid_argument("Invalid axis for swapping");
    }
    
    // 创建排列
    Axis perm(ndim);
    for (int i = 0; i < ndim; i++) {
        perm[i] = i;
    }
    perm[axis1] = axis2;
    perm[axis2] = axis1;
    
    return Transpose(x, perm);
}

NdArray BroadcastTo(const NdArray& x, const Shape& new_shape) {
    const Shape& old_shape = x.shape();
    const int old_ndim = old_shape.size();
    const int new_ndim = new_shape.size();
    
    // 验证新形状兼容
    if (old_ndim > new_ndim) {
        throw std::invalid_argument("Cannot broadcast to lower dimension");
    }
    
    // 检查维度兼容性
    for (int i = 0; i < old_ndim; i++) {
        const int old_dim = old_shape[old_ndim - 1 - i];
        const int new_dim = new_shape[new_ndim - 1 - i];
        
        if (old_dim != new_dim && old_dim != 1) {
            throw std::invalid_argument("Incompatible shapes for broadcasting");
        }
    }
    
    // 创建结果数组
    NdArray result(x.dtype(), new_shape);
    const size_t element_size = NdArray::get_type_size(x.dtype());
    const size_t total_elements = result.size();
    
    // 计算源步长和目标步长
    const std::vector<size_t> src_strides = compute_strides(old_shape, element_size);
    const std::vector<size_t> dst_strides = compute_strides(new_shape, element_size);
    
    // 填充源步长以匹配目标维度
    std::vector<size_t> padded_src_strides(new_ndim, 0);
    for (int i = 0; i < old_ndim; i++) {
        padded_src_strides[new_ndim - old_ndim + i] = src_strides[i];
    }
    
    // 并行复制数据
    RunParallel(static_cast<int>(total_elements), [&](int linear_idx) {
        // 将线性索引转换为多维索引
        Index dst_index(new_ndim);
        size_t remainder = linear_idx;
        for (int d = new_ndim - 1; d >= 0; d--) {
            dst_index[d] = remainder % new_shape[d];
            remainder /= new_shape[d];
        }
        
        // 计算源索引
        Index src_index(old_ndim);
        for (int i = 0; i < old_ndim; i++) {
            const int src_dim = old_shape[i];
            const int dst_dim_idx = new_ndim - old_ndim + i;
            src_index[i] = (src_dim == 1) ? 0 : dst_index[dst_dim_idx];
        }
        
        // 计算偏移量
        size_t src_offset = calculate_offset(src_index, old_shape, src_strides);
        size_t dst_offset = linear_idx * element_size;
        
        // 复制元素
        const void* src_ptr = static_cast<const char*>(x.data_ptr()) + src_offset;
        void* dst_ptr = static_cast<char*>(result.data_ptr()) + dst_offset;
        std::memcpy(dst_ptr, src_ptr, element_size);
    });
    
    return result;
}

} // namespace tinyndarray