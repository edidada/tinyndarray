#include "tinyndarray.h"
#include <thread>
#include <vector>
#include <atomic>
#include <algorithm>

namespace tinyndarray {


// 获取并行参数
void GetParallelParams(int size, int& n_workers, int& n_batch, int& batch_size) {
    n_workers = std::max(1, static_cast<int>(std::thread::hardware_concurrency()));
    batch_size = std::max(1, size / (n_workers * 4));  // 每个批次至少1个元素
    n_batch = (size + batch_size - 1) / batch_size;
}

std::vector<int> ComputeChildSizes(const Shape& shape) {
    const size_t n_shape = shape.size();
    if (n_shape == 0) {
        return {};
    }
    // Compute child sizes from back (the number of children for each dimension)
    std::vector<int> child_sizes(n_shape, 1);
    int size = 1;
    for (size_t depth = n_shape - 1; 0 < depth; depth--) {
        child_sizes[depth] = size;
        size *= shape[depth];
    }
    child_sizes[0] = size;
    return child_sizes;
}

// ----------------------- Utilities for NdArray (Print) -----------------------
static void OutputArrayLine(std::ostream& os, NdArray::ConstIter data,
                            const int size) {
    os << "[";  // Begin of a line
    for (int i = 0; i < size; i++) {
        // 使用解引用操作符 * 获取元素值
        os << *data;
        
        if (i == size - 1) {
            os << "]";  // End of a line
        } else {
            os << ", ";  // Splitter of an element
        }
        
        // 移动到下一个元素
        ++data;
    }
}

static void OutputArrayMultiDim(std::ostream& os,
                                NdArray::ConstIter data,
                                const Shape& shape,
                                const std::vector<int>& child_sizes,
                                size_t depth) {
    for (int i = 0; i < shape[depth]; i++) {
        // Heading
        if (i == 0) {
            os << "[";  // begin of array
        } else {
            for (size_t d = 0; d < depth + 1; d++) {  // array indent
                os << " ";
            }
        }

        // Output internal array
        const int& child_size = child_sizes[depth];
        if (depth == shape.size() - 2) {
            // 使用迭代器偏移访问子数组
            OutputArrayLine(os, data, shape[depth + 1]);
        } else {
            OutputArrayMultiDim(os, data, shape, child_sizes, depth + 1);
        }

        // Tailing
        if (i == shape[depth] - 1) {
            os << "]";  // End of array
        } else {
            os << "," << std::endl;  // Splitter of array
        }
        
        // 移动到下一个子数组的起始位置
        for (int j = 0; j < child_size; j++) {
            ++data;
        }
    }
}

static void OutputNdArray(std::ostream& os, const NdArray& x) {
    const int size = static_cast<int>(x.size());
    const Shape& shape = x.shape();
    const std::vector<int>& child_sizes = ComputeChildSizes(shape);

    if (size == 0 || shape.size() == 0) {
        // Empty
        os << "[]";
    } else if (shape.size() == 1) {
        // 1-dim - 使用 begin() 获取迭代器
        OutputArrayLine(os, x.begin(), size);
    } else {
        // Multi-dim - 使用 begin() 获取迭代器
        OutputArrayMultiDim(os, x.begin(), shape, child_sizes, 0);
    }
}

static void OutputShape(std::ostream& os, const Shape& shape) {
    os << "[";
    for (size_t i = 0; i < shape.size(); i++) {
        os << shape[i];
        if (i < shape.size() - 1) {
            os << ", ";
        }
    }
    os << "]";
}

// 输出运算符
std::ostream& operator<<(std::ostream& os, const NdArray& x) {
    OutputNdArray(os, x);
    return os;
}

std::ostream& operator<<(std::ostream& os, const Shape& shape) {
    OutputShape(os, shape);
    return os;
}

} // namespace tinyndarray