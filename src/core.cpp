#include "tinyndarray.h"
#include <cstring>
#include <numeric>
#include <stdexcept>
#include <thread>

namespace tinyndarray {

NdArray::Iter NdArray::begin() { 
    return Iter(this, 0); 
}

NdArray::Iter NdArray::end() { 
    return Iter(this, m_sub->size); 
}

NdArray::ConstIter NdArray::begin() const { 
    return ConstIter(const_cast<NdArray*>(this), 0); 
}

NdArray::ConstIter NdArray::end() const { 
    return ConstIter(const_cast<NdArray*>(this), m_sub->size); 
}

size_t NdArray::get_type_size(DataType dtype) {
    switch (dtype) {
        case DataType::FLOAT: return sizeof(float);
        case DataType::INT: return sizeof(int);
        case DataType::DOUBLE: return sizeof(double);
        default: throw std::runtime_error("Unknown data type");
    }
}

size_t NdArray::calculate_size(const Shape& shape) {
    if (shape.empty()) return 0;
    return std::accumulate(shape.begin(), shape.end(), 1, std::multiplies<size_t>());
}

size_t NdArray::calculate_offset(const Index& index) const {
    if (index.size() != m_sub->shape.size()) {
        throw std::runtime_error("Index dimension mismatch");
    }
    
    size_t offset = 0;
    size_t stride = 1;
    const size_t ndim = m_sub->shape.size();
    
    for (size_t i = 0; i < ndim; ++i) {
        const size_t dim = ndim - 1 - i;
        const int idx = (index[dim] >= 0) ? 
                        index[dim] : 
                        m_sub->shape[dim] + index[dim];
                        
        if (idx < 0 || idx >= m_sub->shape[dim]) {
            throw std::out_of_range("Index out of range");
        }
        
        offset += stride * static_cast<size_t>(idx);
        stride *= m_sub->shape[dim];
    }
    
    return offset;
}

// 构造函数实现
NdArray::NdArray() 
    : m_sub(std::make_shared<Substance>(DataType::FLOAT, 0, Shape{})) {}

NdArray::NdArray(DataType dtype, const Shape& shape)
    : m_sub(std::make_shared<Substance>(dtype, calculate_size(shape), shape)) {}

NdArray::NdArray(DataType dtype, const Shape& shape, float fill_value)
    : NdArray(dtype, shape) {
    fill(fill_value);
}

NdArray::NdArray(DataType dtype, const Shape& shape, const void* external_data)
    : m_sub(std::make_shared<Substance>(dtype, calculate_size(shape), shape)) {
    const size_t type_size = get_type_size(dtype);
    const size_t total_bytes = m_sub->size * type_size;
    std::memcpy(m_sub->data.get(), external_data, total_bytes);
}

NdArray::NdArray(DataType dtype, const Shape& shape, void* external_data, bool own_data)
    : m_sub(std::make_shared<Substance>(dtype, calculate_size(shape), shape, external_data)) 
{
    // 如果需要拥有数据，则创建拷贝
    if (own_data) {
        size_t total_size = calculate_size(shape) * get_type_size(dtype);
        void* new_data = new char[total_size];
        std::memcpy(new_data, external_data, total_size);
        m_sub = std::make_shared<Substance>(dtype, calculate_size(shape), shape, new_data);
        m_sub->owns_data = true;
    }
}

// 静态工厂方法
NdArray NdArray::Empty(DataType dtype, const Shape& shape) {
    return NdArray(dtype, shape);
}

NdArray NdArray::Zeros(DataType dtype, const Shape& shape) {
    NdArray arr(dtype, shape);
    arr.fill(0);
    return arr;
}

NdArray NdArray::Ones(DataType dtype, const Shape& shape) {
    NdArray arr(dtype, shape);
    arr.fill(1);
    return arr;
}

// 基本属性
uintptr_t NdArray::id() const { 
    return reinterpret_cast<uintptr_t>(m_sub->data.get()); 
}

bool NdArray::empty() const { 
    return m_sub->size == 0; 
}

size_t NdArray::size() const { 
    return m_sub->size; 
}

const Shape& NdArray::shape() const { 
    return m_sub->shape; 
}

size_t NdArray::ndim() const { 
    return m_sub->shape.size(); 
}

DataType NdArray::dtype() const { 
    return m_sub->dtype; 
}

void* NdArray::data_ptr() {
    return m_sub->data.get();
}

const void* NdArray::data_ptr() const {
    return m_sub->data.get();
}

// 类型转换
template <typename T>
T NdArray::as_scalar() const {
    if (size() != 1)
        throw std::runtime_error("Only size-1 arrays can be converted to scalar");
    return static_cast<T>(ElementProxy(m_sub->data.get(), m_sub->dtype));
}

// 显式实例化
template float NdArray::as_scalar<float>() const;
template int NdArray::as_scalar<int>() const;
template double NdArray::as_scalar<double>() const;

// 填充数据
void NdArray::fill(float value) {
    const size_t type_size = get_type_size(m_sub->dtype);
    const size_t total_bytes = m_sub->size * type_size;
    
    switch (m_sub->dtype) {
        case DataType::FLOAT:
            std::fill_n(reinterpret_cast<float*>(m_sub->data.get()), m_sub->size, value);
            break;
        case DataType::INT:
            std::fill_n(reinterpret_cast<int*>(m_sub->data.get()), m_sub->size, static_cast<int>(value));
            break;
        case DataType::DOUBLE:
            std::fill_n(reinterpret_cast<double*>(m_sub->data.get()), m_sub->size, static_cast<double>(value));
            break;
        default:
            throw std::runtime_error("Unsupported data type for fill");
    }
}

// 复制数组
NdArray NdArray::copy() const {
    NdArray copy(m_sub->dtype, m_sub->shape);
    const size_t type_size = get_type_size(m_sub->dtype);
    const size_t total_bytes = m_sub->size * type_size;
    std::memcpy(copy.m_sub->data.get(), m_sub->data.get(), total_bytes);
    return copy;
}

// 调整大小
void NdArray::resize(const Shape& new_shape) {
    const size_t new_size = calculate_size(new_shape);
    if (new_size == m_sub->size) {
        m_sub->shape = new_shape;
        return;
    }
    
    // 创建新存储
    auto new_sub = std::make_shared<Substance>(m_sub->dtype, new_size, new_shape);
    const size_t type_size = get_type_size(m_sub->dtype);
    const size_t copy_size = std::min(m_sub->size, new_size) * type_size;
    
    // 复制数据
    std::memcpy(new_sub->data.get(), m_sub->data.get(), copy_size);
    
    // 填充剩余部分
    if (new_size > m_sub->size) {
        const size_t fill_start = m_sub->size * type_size;
        const size_t fill_size = (new_size - m_sub->size) * type_size;
        std::memset(reinterpret_cast<char*>(new_sub->data.get()) + fill_start, 0, fill_size);
    }
    
    m_sub = new_sub;
}

// 元素访问
ElementProxy NdArray::operator[](size_t index) {
    if (index >= m_sub->size) {
        throw std::out_of_range("Index out of range");
    }
    size_t type_size = get_type_size(m_sub->dtype);
    void* elem_ptr = reinterpret_cast<char*>(m_sub->data.get()) + index * type_size;
    return ElementProxy(elem_ptr, m_sub->dtype);
}

ConstElementProxy NdArray::operator[](size_t index) const {
    if (index >= m_sub->size) {
        throw std::out_of_range("Index out of range");
    }
    size_t type_size = get_type_size(m_sub->dtype);
    const void* elem_ptr = reinterpret_cast<char*>(m_sub->data.get()) + index * type_size;
    return ConstElementProxy(elem_ptr, m_sub->dtype);
}

ElementProxy NdArray::operator[](const Index& index) {
    return (*this)[calculate_offset(index)];
}

ConstElementProxy NdArray::operator[](const Index& index) const {
    return (*this)[calculate_offset(index)];
}

template <typename... I>
ElementProxy NdArray::operator()(I... indices) {
    return (*this)[{static_cast<int>(indices)...}];
}

// 形状操作
NdArray NdArray::reshape(const Shape& new_shape) const {
    const size_t new_size = calculate_size(new_shape);
    if (new_size != m_sub->size) {
        throw std::runtime_error("Total size must remain the same for reshape");
    }
    
    NdArray result = *this;
    result.m_sub->shape = new_shape;
    return result;
}

NdArray NdArray::flatten() const {
    return reshape({static_cast<int>(m_sub->size)});
}

NdArray NdArray::ravel() const {
    return reshape({static_cast<int>(m_sub->size)});
}

// 类型转换操作符
NdArray::operator float() const { return as_scalar<float>(); }
NdArray::operator int() const { return as_scalar<int>(); }
NdArray::operator double() const { return as_scalar<double>(); }

// 类型转换
NdArray NdArray::astype(DataType new_dtype) const {
    if (m_sub->dtype == new_dtype) {
        return *this;  // 无需转换
    }
    
    NdArray result(new_dtype, shape());
    size_t size = this->size();
    
    // 获取数据指针
    const char* src_data = static_cast<const char*>(data_ptr());
    char* dst_data = static_cast<char*>(result.data_ptr());
    
    // 类型转换
    for (size_t i = 0; i < size; i++) {
        switch (m_sub->dtype) {
            case DataType::FLOAT: {
                float value = *reinterpret_cast<const float*>(src_data);
                switch (new_dtype) {
                    case DataType::INT: 
                        *reinterpret_cast<int*>(dst_data) = static_cast<int>(value);
                        break;
                    case DataType::DOUBLE: 
                        *reinterpret_cast<double*>(dst_data) = static_cast<double>(value);
                        break;
                    default: break;
                }
                break;
            }
            case DataType::INT: {
                int value = *reinterpret_cast<const int*>(src_data);
                switch (new_dtype) {
                    case DataType::FLOAT: 
                        *reinterpret_cast<float*>(dst_data) = static_cast<float>(value);
                        break;
                    case DataType::DOUBLE: 
                        *reinterpret_cast<double*>(dst_data) = static_cast<double>(value);
                        break;
                    default: break;
                }
                break;
            }
            case DataType::DOUBLE: {
                double value = *reinterpret_cast<const double*>(src_data);
                switch (new_dtype) {
                    case DataType::FLOAT: 
                        *reinterpret_cast<float*>(dst_data) = static_cast<float>(value);
                        break;
                    case DataType::INT: 
                        *reinterpret_cast<int*>(dst_data) = static_cast<int>(value);
                        break;
                    default: break;
                }
                break;
            }
            default:
                throw std::runtime_error("Unsupported data type for conversion");
        }
        
        src_data += NdArray::get_type_size(m_sub->dtype);
        dst_data += NdArray::get_type_size(new_dtype);
    }
    
    return result;
}
} // namespace tinyndarray