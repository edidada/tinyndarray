#include "cnpy.h"  // 包含 cnpy 头文件
#include "tinyndarray.h"
#include <cstring>

namespace tinyndarray {

// ====================== NPY 文件支持 ======================

// 获取 cnpy 类型表示
const char* NdArray::get_cnpy_type() const {
    switch (m_sub->dtype) {
        case DataType::FLOAT: return "float32";
        case DataType::DOUBLE: return "float64";
        case DataType::INT: return "int32";
        default: throw std::runtime_error("Unsupported data type for NPY export");
    }
}

// 获取 cnpy 类型大小
size_t NdArray::get_cnpy_type_size() const {
    switch (m_sub->dtype) {
        case DataType::FLOAT: return sizeof(float);
        case DataType::DOUBLE: return sizeof(double);
        case DataType::INT: return sizeof(int);
        default: throw std::runtime_error("Unsupported data type for NPY export");
    }
}

// 获取 cnpy 类型字符
char NdArray::get_cnpy_type_char() const {
    switch (m_sub->dtype) {
        case DataType::FLOAT: return 'f';
        case DataType::DOUBLE: return 'f';
        case DataType::INT: return 'i';
        default: throw std::runtime_error("Unsupported data type for NPY export");
    }
}

// 从文件加载 NPY 数组
NdArray Load(const std::string& file_path) {
    // 使用 cnpy 加载文件
    cnpy::NpyArray npy_data = cnpy::npy_load(file_path);
    
    // 转换数据类型
    DataType dtype;
    if (npy_data.type_char == 'f') {
        dtype = (npy_data.word_size == 4) ? DataType::FLOAT : DataType::DOUBLE;
    } else if (npy_data.type_char == 'i') {
        dtype = (npy_data.word_size == 4) ? DataType::INT : DataType::UNKNOWN;
    } else {
        throw std::runtime_error("Unsupported NPY data type");
    }
    
    if (dtype == DataType::UNKNOWN) {
        throw std::runtime_error("Unsupported integer size in NPY file");
    }
    
    // 创建形状向量
    Shape shape;
    for (size_t i = 0; i < npy_data.shape.size(); ++i) {
        shape.push_back(static_cast<int>(npy_data.shape[i]));
    }
    
    // 创建 NdArray
    NdArray array(dtype, shape);
    
    // 复制数据
    size_t total_bytes = npy_data.num_bytes();
    void* dest = array.data_ptr();
    const void* src = npy_data.data<void>();
    std::memcpy(dest, src, total_bytes);
    
    return array;
}

// 保存为 NPY 文件
void NdArray::Save(const std::string& file_path) const {
    // 获取形状作为 size_t 向量
    std::vector<size_t> shape;
    for (int dim : m_sub->shape) {
        shape.push_back(static_cast<size_t>(dim));
    }
    
    // 获取数据指针
    const void* data = m_sub->data.get();
    
    // 根据数据类型调用 cnpy
    switch (m_sub->dtype) {
        case DataType::FLOAT:
            cnpy::npy_save(file_path, static_cast<const float*>(data), shape, "w");
            break;
        case DataType::DOUBLE:
            cnpy::npy_save(file_path, static_cast<const double*>(data), shape, "w");
            break;
        case DataType::INT:
            cnpy::npy_save(file_path, static_cast<const int*>(data), shape, "w");
            break;
        default:
            throw std::runtime_error("Unsupported data type for NPY export");
    }
}

// ====================== 从 cnpy 数组加载 ======================

NdArray NdArray::FromCnpyArray(const void* cnpy_array_ptr) {
    const cnpy::NpyArray* npy_array = static_cast<const cnpy::NpyArray*>(cnpy_array_ptr);
    
    // 转换数据类型
    DataType dtype;
    if (npy_array->type_char == 'f') {
        dtype = (npy_array->word_size == 4) ? DataType::FLOAT : DataType::DOUBLE;
    } else if (npy_array->type_char == 'i') {
        dtype = (npy_array->word_size == 4) ? DataType::INT : DataType::UNKNOWN;
    } else {
        throw std::runtime_error("Unsupported NPY data type");
    }
    
    if (dtype == DataType::UNKNOWN) {
        throw std::runtime_error("Unsupported integer size in NPY array");
    }
    
    // 创建形状向量
    Shape shape;
    for (size_t i = 0; i < npy_array->shape.size(); ++i) {
        shape.push_back(static_cast<int>(npy_array->shape[i]));
    }
    
    // 创建 NdArray
    NdArray array(dtype, shape);
    
    // 复制数据
    size_t total_bytes = npy_array->num_bytes();
    void* dest = array.data_ptr();
    const void* src = npy_array->data<void>();
    std::memcpy(dest, src, total_bytes);
    
    return array;
}

} // namespace tinyndarray