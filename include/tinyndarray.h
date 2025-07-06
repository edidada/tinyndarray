#ifndef TINYNDARRAY_H
#define TINYNDARRAY_H

#include <vector>
#include <memory>
#include <cstddef>
#include <stdexcept>
#include <initializer_list>
#include <utility>
#include <atomic>
#include <mutex>
#include <random>
#include <iostream>
#include <thread>
namespace tinyndarray {

// 数据类型枚举
enum class DataType {
    FLOAT,
    INT,
    DOUBLE,
    UNKNOWN
};

// 辅助类型定义
using InitShape = std::initializer_list<int>;
using Shape = std::vector<int>;
using Index = std::vector<int>;
using SliceIndex = std::vector<std::pair<int, int>>;
using Axis = std::vector<int>;

class ConstElementProxy;
class ElementProxy;
class NdArray;

// 代理类
class ConstElementProxy {
private:
    const void* ptr_;     // 指向元素的指针
    DataType dtype_;      // 元素的数据类型

public:
    ConstElementProxy(const void* ptr, DataType dtype) 
        : ptr_(ptr), dtype_(dtype) {}

    // 类型转换运算符
    operator float() const {
        if (dtype_ != DataType::FLOAT) 
            throw std::runtime_error("Type mismatch: not a float");
        return *static_cast<const float*>(ptr_);
    }

    operator int() const {
        if (dtype_ != DataType::INT) 
            throw std::runtime_error("Type mismatch: not an int");
        return *static_cast<const int*>(ptr_);
    }

    operator double() const {
        if (dtype_ != DataType::DOUBLE) 
            throw std::runtime_error("Type mismatch: not a double");
        return *static_cast<const double*>(ptr_);
    }

    friend std::ostream& operator<<(std::ostream& os, const ConstElementProxy& proxy) {
        switch (proxy.dtype_) {
            case DataType::FLOAT:
                os << *static_cast<const float*>(proxy.ptr_);
                break;
            case DataType::INT:
                os << *static_cast<const int*>(proxy.ptr_);
                break;
            case DataType::DOUBLE:
                os << *static_cast<const double*>(proxy.ptr_);
                break;
            default:
                os << "?";
        }
        return os;
    }
};

// ================================ ElementProxy ===============================
class ElementProxy {
private:
    void* ptr_;     // 指向元素的指针
    DataType dtype_; // 元素的数据类型
    bool is_temporary_; // 标记是否为临时值

public:
    // 构造函数（用于原始数组元素）
    ElementProxy(void* ptr, DataType dtype) 
        : ptr_(ptr), dtype_(dtype), is_temporary_(false) {}
    
    // 构造函数（用于临时值）
    ElementProxy(void* ptr, DataType dtype, bool is_temporary) 
        : ptr_(ptr), dtype_(dtype), is_temporary_(is_temporary) {}
    
    // // 移动构造函数
    // ElementProxy(ElementProxy&& other) noexcept
    //     : ptr_(other.ptr_), dtype_(other.dtype_), is_temporary_(other.is_temporary_) {
    //     other.ptr_ = nullptr;
    //     other.is_temporary_ = false;
    // }
    // 
    // 拷贝赋值运算符
    ElementProxy& operator=(const ElementProxy& other) {
        if (this != &other) {
            // 复制值而不是指针
            switch (dtype_) {
                case DataType::FLOAT:
                    *static_cast<float*>(ptr_) = *static_cast<float*>(other.ptr_);
                    break;
                case DataType::INT:
                    *static_cast<int*>(ptr_) = *static_cast<int*>(other.ptr_);
                    break;
                case DataType::DOUBLE:
                    *static_cast<double*>(ptr_) = *static_cast<double*>(other.ptr_);
                    break;
                default:
                    throw std::runtime_error("Unsupported data type for copy assignment");
            }
        }
        return *this;
    }
    
    // // 移动赋值运算符
    // ElementProxy& operator=(ElementProxy&& other) {
    //     if (this != &other) {
    //         // 如果当前是临时值，先释放资源
    //         if (is_temporary_ && ptr_) {
    //             release();
    //         }
            
    //         // 移动资源
    //         ptr_ = other.ptr_;
    //         dtype_ = other.dtype_;
    //         is_temporary_ = other.is_temporary_;
            
    //         // 置空other
    //         other.ptr_ = nullptr;
    //         other.is_temporary_ = false;
    //     }
    //     return *this;
    // }
    
    ~ElementProxy() {
        release();
    }

    void release() {
        if (is_temporary_ && ptr_) {
            switch (dtype_) {
                case DataType::FLOAT: delete static_cast<float*>(ptr_); break;
                case DataType::INT: delete static_cast<int*>(ptr_); break;
                case DataType::DOUBLE: delete static_cast<double*>(ptr_); break;
                default: break;
            }
            ptr_ = nullptr;
        }
    }

    // 类型转换运算符
    operator float() const {
        if (dtype_ != DataType::FLOAT) 
            throw std::runtime_error("Type mismatch: not a float");
        return *static_cast<float*>(ptr_);
    }

    operator int() const {
        if (dtype_ != DataType::INT) 
            throw std::runtime_error("Type mismatch: not an int");
        return *static_cast<int*>(ptr_);
    }

    operator double() const {
        if (dtype_ != DataType::DOUBLE) 
            throw std::runtime_error("Type mismatch: not a double");
        return *static_cast<double*>(ptr_);
    }

    // 赋值运算符
    ElementProxy& operator=(float value) {
        if (dtype_ != DataType::FLOAT) 
            throw std::runtime_error("Cannot assign float to non-float element");
        *static_cast<float*>(ptr_) = value;
        return *this;
    }

    ElementProxy& operator=(int value) {
        if (dtype_ != DataType::INT) 
            throw std::runtime_error("Cannot assign int to non-int element");
        *static_cast<int*>(ptr_) = value;
        return *this;
    }

    ElementProxy& operator=(double value) {
        if (dtype_ != DataType::DOUBLE) 
            throw std::runtime_error("Cannot assign double to non-double element");
        *static_cast<double*>(ptr_) = value;
        return *this;
    }

    // 通用赋值模板（自动类型转换）
    template <typename T>
    ElementProxy& operator=(T value) {
        switch (dtype_) {
            case DataType::FLOAT:
                *static_cast<float*>(ptr_) = static_cast<float>(value);
                break;
            case DataType::INT:
                *static_cast<int*>(ptr_) = static_cast<int>(value);
                break;
            case DataType::DOUBLE:
                *static_cast<double*>(ptr_) = static_cast<double>(value);
                break;
            default:
                throw std::runtime_error("Unknown data type for assignment");
        }
        return *this;
    }
    
    // 重置指针和类型
    void reset(void* ptr, DataType dtype) {
        ptr_ = ptr;
        dtype_ = dtype;
    }
    
    // 算术运算符重载
    ElementProxy operator*(int factor) const {
        return apply_unary_operation([factor](auto a) { return a * factor; });
    }
    
    friend std::ostream& operator<<(std::ostream& os, const ElementProxy& proxy) {
        switch (proxy.dtype_) {
            case DataType::FLOAT:
                os << *static_cast<const float*>(proxy.ptr_);
                break;
            case DataType::INT:
                os << *static_cast<const int*>(proxy.ptr_);
                break;
            case DataType::DOUBLE:
                os << *static_cast<const double*>(proxy.ptr_);
                break;
            default:
                os << "?";
        }
        return os;
    }
private:
    // 应用一元操作
    template <typename Op>
    ElementProxy apply_unary_operation(Op op) const {
        switch (dtype_) {
            case DataType::FLOAT: {
                float val = *static_cast<float*>(ptr_);
                return ElementProxy(new float(op(val)), DataType::FLOAT, true);
            }
            case DataType::INT: {
                int val = *static_cast<int*>(ptr_);
                return ElementProxy(new int(op(val)), DataType::INT, true);
            }
            case DataType::DOUBLE: {
                double val = *static_cast<double*>(ptr_);
                return ElementProxy(new double(op(val)), DataType::DOUBLE, true);
            }
            default:
                throw std::runtime_error("Unsupported data type for operation");
        }
    }
};


// 主类声明
class NdArray {
public:
    template <bool Const>
    class IterBase;
    using Iter = IterBase<false>;
    using ConstIter = IterBase<true>;

    static std::mt19937 s_rand_engine;
    static std::atomic<uint32_t> s_rand_seed;
    static std::mutex s_rand_mutex;
    
    // 构造和析构
    NdArray();
    NdArray(DataType dtype, const Shape& shape);
    NdArray(DataType dtype, const Shape& shape, float fill_value);
    NdArray(DataType dtype, const Shape& shape, const void* external_data);
    // 新增构造函数：使用外部数据（无所有权）
    NdArray(DataType dtype, const Shape& shape, void* external_data, bool own_data = false);
    NdArray(const NdArray&) = default;
    NdArray(NdArray&&) noexcept = default;
    ~NdArray() = default;
    NdArray astype(DataType new_dtype) const;
    // 赋值运算符
    NdArray& operator=(const NdArray&) = default;
    NdArray& operator=(NdArray&&) = default;

    // 静态创建方法
    static NdArray Empty(DataType dtype, const Shape& shape);
    static NdArray Zeros(DataType dtype, const Shape& shape);
    static NdArray Ones(DataType dtype, const Shape& shape);
    
    // 类型转换
    template <typename T>
    T as_scalar() const;
    
    void* data_ptr();
    const void* data_ptr() const;
    // 检查是否拥有数据
    bool owns_data() const;

    template <typename T>
    T* typed_data();
    
    template <typename T>
    const T* typed_data() const;
    
    // 基本属性
    uintptr_t id() const;
    bool empty() const;
    size_t size() const;
    const Shape& shape() const;
    size_t ndim() const;
    DataType dtype() const;
    
    // 数据操作
    void fill(float value);
    NdArray copy() const;
    void resize(const Shape& new_shape);
    
    // 迭代器
    Iter begin();
    Iter end();
    ConstIter begin() const;
    ConstIter end() const;
    
    // 元素访问
    ElementProxy operator[](size_t index);
    ElementProxy operator[](const Index& index);
    ConstElementProxy operator[](size_t index) const;
    ConstElementProxy operator[](const Index& index) const;
    
    template <typename... I>
    ElementProxy operator()(I... indices);
    
    // 形状操作
    NdArray reshape(const Shape& new_shape) const;
    NdArray flatten() const;
    NdArray ravel() const;
    
    // 切片操作
    NdArray slice(const SliceIndex& slice_index) const;
    template <typename... I>
    NdArray slice(std::initializer_list<I>... slice_index) const;
    
    // 点积和叉积
    NdArray dot(const NdArray& other) const;
    NdArray cross(const NdArray& other) const;
    
    // 归约操作
    NdArray sum(const Axis& axes = {}, bool keepdims = false) const;
    NdArray mean(const Axis& axes = {}, bool keepdims = false) const;
    NdArray min(const Axis& axes = {}, bool keepdims = false) const;
    NdArray max(const Axis& axes = {}, bool keepdims = false) const;
    
    // 文件 I/O 方法
    void Save(const std::string& file_path) const;
    
    // 从 cnpy 数组加载
    static NdArray FromCnpyArray(const void* cnpy_array);
    
    // 获取 cnpy 类型
    const char* get_cnpy_type() const;
    
    // 获取 cnpy 类型大小
    size_t get_cnpy_type_size() const;
    
    // 获取 cnpy 类型字符
    char get_cnpy_type_char() const;

    // 类型转换操作符
    operator float() const;
    operator int() const;
    operator double() const;
    
    // 随机数
    static void Seed();
    static void Seed(uint32_t seed);
    static NdArray Uniform(float low, float high, const Shape& shape);
    static NdArray Uniform(double low, double high, const Shape& shape);
    static NdArray Uniform(int low, int high, const Shape& shape);
    static NdArray Uniform(const Shape& shape);
    static NdArray Normal(float loc, float scale, const Shape& shape);
    static NdArray Normal(double loc, double scale, const Shape& shape);
    static NdArray Normal(const Shape& shape);
    
    static size_t get_type_size(DataType dtype);

    // 辅助方法
    size_t calculate_offset(const Index& index) const;
    static size_t calculate_size(const Shape& shape);
    static std::mt19937 GetThreadSafeRandomEngine();
private:
    class Substance;
    std::shared_ptr<Substance> m_sub;
};

NdArray Load(const std::string& file_path);

class NdArray::Substance {
public:
    DataType dtype;
    size_t size;
    Shape shape;
    std::shared_ptr<void> data;
    bool owns_data;  // 新增：标识是否拥有数据所有权

    // 拥有数据的构造函数
    Substance(DataType dtype_, size_t size_, const Shape& shape_)
        : dtype(dtype_), size(size_), shape(shape_),
          data(new char[size_ * NdArray::get_type_size(dtype_)], std::default_delete<char[]>()),
          owns_data(true) {}
    
    // 使用外部数据的构造函数
    Substance(DataType dtype_, size_t size_, const Shape& shape_, void* external_data)
        : dtype(dtype_), size(size_), shape(shape_),
          data(external_data, [](void*){}),  // 空删除器
          owns_data(false) {}
};

size_t calculate_offset(const Index& index, const Shape& shape, const std::vector<size_t>& strides);
std::vector<size_t> compute_strides(const Shape& shape, size_t element_size);

// 迭代器实现
// ================================ 迭代器实现 ================================
template <bool Const>
class NdArray::IterBase {
public:
    using iterator_category = std::random_access_iterator_tag;
    using value_type = std::conditional_t<Const, const ElementProxy, ElementProxy>;
    using difference_type = std::ptrdiff_t;
    using pointer = value_type*;
    using reference = value_type&;

    IterBase(NdArray* array, size_t index) 
        : array_(array), index_(index) {
        // 只在有效索引时更新代理
        if (index < array->size()) {
            update_proxy();
        }
    }

    // 解引用
    reference operator*() {
        if (index_ >= array_->size()) {
            throw std::out_of_range("Cannot dereference end iterator");
        }
        update_proxy();
        return proxy_;
    }

    // 成员访问
    pointer operator->() {
        if (index_ >= array_->size()) {
            throw std::out_of_range("Cannot dereference end iterator");
        }
        update_proxy();
        return &proxy_;
    }

    // 自增/自减
    IterBase& operator++() {
        ++index_;
        if (index_ < array_->size()) {
            update_proxy();
        }
        return *this;
    }

    IterBase operator++(int) {
        IterBase tmp = *this;
        ++index_;
        if (index_ < array_->size()) {
            update_proxy();
        }
        return tmp;
    }

    IterBase& operator--() {
        --index_;
        update_proxy();
        return *this;
    }

    IterBase operator--(int) {
        IterBase tmp = *this;
        --index_;
        update_proxy();
        return tmp;
    }

    // 算术运算
    IterBase operator+(difference_type n) const {
        return IterBase(array_, index_ + n);
    }

    IterBase operator-(difference_type n) const {
        return IterBase(array_, index_ - n);
    }

    IterBase& operator+=(difference_type n) {
        index_ += n;
        if (index_ < array_->size()) {
            update_proxy();
        }
        return *this;
    }

    IterBase& operator-=(difference_type n) {
        index_ -= n;
        update_proxy();
        return *this;
    }

    // 关系运算
    bool operator==(const IterBase& other) const {
        return array_ == other.array_ && index_ == other.index_;
    }

    bool operator!=(const IterBase& other) const {
        return !(*this == other);
    }

    bool operator<(const IterBase& other) const {
        return array_ == other.array_ && index_ < other.index_;
    }

    bool operator>(const IterBase& other) const {
        return array_ == other.array_ && index_ > other.index_;
    }

    bool operator<=(const IterBase& other) const {
        return array_ == other.array_ && index_ <= other.index_;
    }

    bool operator>=(const IterBase& other) const {
        return array_ == other.array_ && index_ >= other.index_;
    }

    // 转换操作符（用于const转换）
    operator IterBase<true>() const {
        return IterBase<true>(array_, index_);
    }

private:
    void update_proxy() {
        if (index_ >= array_->size()) {
            return; // end() 位置不更新代理
        }
        size_t type_size = NdArray::get_type_size(array_->dtype());
        void* ptr = reinterpret_cast<char*>(array_->m_sub->data.get()) + index_ * type_size;
        proxy_.reset(ptr, array_->dtype());
    }

    NdArray* array_;
    size_t index_;
    ElementProxy proxy_{nullptr, DataType::FLOAT};
};

// 非成员函数声明
std::ostream& operator<<(std::ostream& os, const NdArray& arr);
std::ostream& operator<<(std::ostream& os, const Shape& shape);
NdArray operator+(const NdArray& lhs, const NdArray& rhs);
NdArray operator-(const NdArray& lhs, const NdArray& rhs);
NdArray operator*(const NdArray& lhs, const NdArray& rhs);
NdArray operator/(const NdArray& lhs, const NdArray& rhs);
NdArray operator-(const NdArray& x);

// 形状操作函数
NdArray Reshape(const NdArray& x, const Shape& shape);
NdArray Squeeze(const NdArray& x, const Axis& axes = {});
NdArray ExpandDims(const NdArray& x, int axis);
NdArray Transpose(const NdArray& x, const Axis& perm);
NdArray SwapAxes(const NdArray& x, int axis1, int axis2);
NdArray BroadcastTo(const NdArray& x, const Shape& new_shape);

// 组合操作
NdArray Stack(const std::vector<NdArray>& xs, int axis);
NdArray Concatenate(const std::vector<NdArray>& xs, int axis);
std::vector<NdArray> Split(const NdArray& x, int n_section, int axis);
std::vector<NdArray> Split(const NdArray& x, const Index& idxs, int axis);
std::vector<NdArray> Separate(const NdArray& x, int axis);

void GetParallelParams(int size, int& n_workers, int& n_batch, int& batch_size);
std::vector<int> ComputeChildSizes(const Shape& shape);
template <typename F>
void RunParallel(int size, F op) {
    // 决定并行参数
    int n_workers = -1, n_batch = -1, batch_size = -1;
    GetParallelParams(size, n_workers, n_batch, batch_size);

    if (n_workers <= 1) {
        // 单线程执行
        for (int i = 0; i < size; i++) {
            op(i);
        }
    } else {
        // 并行执行
        std::atomic<int> next_batch(0);
        std::vector<std::thread> workers;
        workers.reserve(n_workers);
        
        for (int w = 0; w < n_workers; w++) {
            workers.emplace_back([=, &next_batch, &op]() {
                while (true) {
                    int batch = next_batch++;
                    if (batch >= n_batch) break;
                    
                    int start = batch * batch_size;
                    int end = std::min(start + batch_size, size);
                    
                    for (int i = start; i < end; i++) {
                        op(i);
                    }
                }
            });
        }
        
        for (auto& worker : workers) {
            worker.join();
        }
    }
}

// 带归约的并行执行
template <typename F, typename R>
float RunParallelWithReduce(int size, F op, R reduce, float init_v) {
    // 决定并行参数
    int n_workers = -1, n_batch = -1, batch_size = -1;
    GetParallelParams(size, n_workers, n_batch, batch_size);

    if (n_workers <= 1) {
        // 单线程执行
        float v = init_v;
        for (int i = 0; i < size; i++) {
            v = reduce(v, op(i));
        }
        return v;
    } else {
        // 并行执行
        std::atomic<int> next_batch(0);
        std::vector<std::thread> workers;
        std::vector<float> results(n_workers);
        workers.reserve(n_workers);
        
        for (int w = 0; w < n_workers; w++) {
            workers.emplace_back([=, &next_batch, &results, &op, &reduce]() {
                float v = init_v;
                while (true) {
                    int batch = next_batch++;
                    if (batch >= n_batch) break;
                    
                    int start = batch * batch_size;
                    int end = std::min(start + batch_size, size);
                    
                    for (int i = start; i < end; i++) {
                        v = reduce(v, op(i));
                    }
                }
                results[w] = v;
            });
        }
        
        for (auto& worker : workers) {
            worker.join();
        }
        
        // 归约结果
        float result = init_v;
        for (float v : results) {
            result = reduce(result, v);
        }
        return result;
    }
}

} // namespace tinyndarray

#endif // TINYNDARRAY_H