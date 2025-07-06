#include "tinyndarray.h"
#include <random>

namespace tinyndarray {

std::mt19937 NdArray::s_rand_engine(std::random_device{}());
std::atomic<uint32_t> NdArray::s_rand_seed(0);
std::mutex NdArray::s_rand_mutex;

// 随机种子
void NdArray::Seed() {
    s_rand_engine = std::mt19937(s_rand_seed);
}

void NdArray::Seed(uint32_t seed) {
    s_rand_engine = std::mt19937(seed);
}

// 类型转换辅助函数
template <typename T>
DataType GetDataType() {
    if constexpr (std::is_same<T, float>::value) {
        return DataType::FLOAT;
    } else if constexpr (std::is_same<T, double>::value) {
        return DataType::DOUBLE;
    } else if constexpr (std::is_same<T, int>::value) {
        return DataType::INT;
    } else {
        static_assert(sizeof(T) == 0, "Unsupported data type");
    }
}

// 线程安全的随机引擎获取
std::mt19937 NdArray::GetThreadSafeRandomEngine() {
    static thread_local std::mt19937 engine = []{
        std::lock_guard<std::mutex> lock(s_rand_mutex);
        // std::seed_seq seq{NdArray::s_rand_seed.load(), NdArray::s_rand_engine()};
        std::seed_seq seq;
        return std::mt19937(seq);
    }();
    
    return engine;
}

// 通用随机数组创建函数
template <typename T, typename Distribution>
NdArray CreateRandomArrayImpl(const Shape& shape, Distribution&& dist) {
    NdArray result(GetDataType<T>(), shape);
    const int size = static_cast<int>(result.size());
    T* data = static_cast<T*>(result.data_ptr());
    
    // 获取随机引擎的线程安全副本
    std::mt19937 local_engine = NdArray::GetThreadSafeRandomEngine();
    
    RunParallel(size, [&](int i) {
        data[i] = static_cast<T>(dist(local_engine));
    });
    
    return result;
}

// 内部实现
// 均匀分布（支持整数和浮点）
template <typename T>
NdArray UniformImpl(T low, T high, const Shape& shape) {
    std::uniform_real_distribution<double> dist(static_cast<double>(low), 
                                            static_cast<double>(high));
    return CreateRandomArrayImpl<T>(shape, dist);
}

template <typename T>
NdArray UniformImpl(const Shape& shape) {
    return UniformImpl<T>(static_cast<T>(0), static_cast<T>(1), shape);
}

// 正态分布（只支持浮点类型）
template <typename T>
NdArray NormalImpl(T loc, T scale, const Shape& shape) {
    static_assert(std::is_floating_point<T>::value, 
                 "Normal distribution only supports floating point types");
    
    std::normal_distribution<double> dist(static_cast<double>(loc), 
                                        static_cast<double>(scale));
    return CreateRandomArrayImpl<T>(shape, dist);
}

template <typename T>
NdArray NormalImpl(const Shape& shape) {
    return NormalImpl<T>(static_cast<T>(0), static_cast<T>(1), shape);
}


// 均匀分布
NdArray NdArray::Uniform(float low, float high, const Shape& shape) {
    return UniformImpl<float>(low, high, shape);
}

// 正态分布
NdArray NdArray::Normal(float loc, float scale, const Shape& shape) {
    return NormalImpl<float>(loc, scale, shape);
}


} // namespace tinyndarray