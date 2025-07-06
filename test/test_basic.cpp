#include <iostream>
#include <vector>
#include <stdexcept>
#include <iterator>
#include <type_traits>
#include <algorithm>
#include "tinyndarray.h"
#undef __EXCEPTIONS
#include "utest.h"

using namespace tinyndarray;

UTEST(ndarray, basic) {
    // 创建float类型数组
    auto float_arr = NdArray::Zeros(DataType::FLOAT, {3, 3});
    float_arr[0] = 1.5f;       // 正确赋值
    float_arr[{1, 1}] = 2.5f;  // 多维索引
    
    try {
        float_arr[0] = 10;    // 自动转换为float (1.5f -> 10.0f)
        int val = float_arr[0]; // 抛出异常（类型不匹配）
    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << std::endl;
    }
    
    // 创建int类型数组
    NdArray int_arr(DataType::INT, {4});
    for (size_t i = 0; i < int_arr.size(); ++i) {
        int_arr[i] = static_cast<int>(i * i);
    }
    
    // 点积运算
    auto vec1 = NdArray::Ones(DataType::FLOAT, {5});
    auto vec2 = NdArray::Ones(DataType::FLOAT, {5});
    auto dot_result = vec1.dot(vec2);
    float dot_value = dot_result; // 标量转换
    
    // 迭代器使用
    for (auto& elem : int_arr) {
        elem = elem * 2; // 通过代理修改元素
    }
    
    for (auto& elem : int_arr) {
        std::cout << static_cast<int>(elem) << std::endl;
    }
    
    std::cout << int_arr << std::endl;
    // 类型转换
    NdArray scalar_arr(DataType::INT, {1}, 42);
    int value = scalar_arr; // 转换为int
}

UTEST(ndarray, slice) {
    // 创建float类型数组
    auto m = NdArray::Uniform(0.f, 10.f, {6, 6});
    auto sliced_m_1 = m.slice({0, 3}, {1, 3});
    auto sliced_m_2 = m.slice({1, 4}, {2, 4});
    auto m2 = sliced_m_1 + sliced_m_2; 
    std::cout << m2 << std::endl;
}
UTEST(ndarray, concat) {
    auto m1 = NdArray::Uniform(0.f, 10.f, {6, 6});
    auto m2 = NdArray::Uniform(0.f, 10.f, {6, 6});
    auto m = Concatenate({m1, m2}, 1);
    std::cout << m1 << std::endl;
    std::cout << m2 << std::endl;
    std::cout << m << std::endl;
}

UTEST(ndarray, transpose) {
    auto m1 = NdArray::Uniform(0.f, 10.f, {2, 2, 2});
    auto m2 = Transpose(m1, {0, 2, 1});
    std::cout << m1 << std::endl;
    std::cout << m2 << std::endl;
}

UTEST(ndarray, split) {
    auto m = NdArray::Uniform(0.f, 10.f, {6, 6});
    auto mv = Split(m, 2, 1);
    std::cout << m << std::endl;
    std::cout << mv.size() << std::endl;
    std::cout << mv[0] << std::endl;
    std::cout << mv[1] << std::endl;
}

UTEST(ndarray, add) {
    auto m1 = NdArray::Uniform(0.f, 10.f, {3, 3});
    auto m2 = NdArray::Uniform(0.f, 10.f, {1, 3});
    auto m = m1 + m2;
    std::cout << m1 << std::endl;
    std::cout << m2 << std::endl;
    std::cout << m << std::endl;
}

UTEST(ndarray, cvt) {
    auto m1 = NdArray::Uniform(0.f, 10.f, {3, 3});
    auto m2 = m1.astype(DataType::INT);
    std::cout << m1 << std::endl;
    std::cout << m2 << std::endl;
}


UTEST(ndarray, reduce) {
    auto m1 = NdArray::Uniform(0.f, 10.f, {3, 3});
    auto m2 = m1.sum({1});
    std::cout << m1 << std::endl;
    std::cout << m2 << std::endl;
}

UTEST(ndarray, neg) {
    auto m1 = NdArray::Uniform(0.f, 10.f, {3, 3});
    auto m2 = -m1;
    std::cout << m1 << std::endl;
    std::cout << m2 << std::endl;
}

UTEST(ndarray, io) {
    auto m1 = NdArray::Uniform(0.f, 10.f, {3, 3});
    m1.Save("m1.npy");
    auto m2 = NdArray::Load("m1.npy");
    std::cout << m1 << std::endl;
    std::cout << m2 << std::endl;
}

UTEST_STATE();
int main(int argc, const char *const argv[]) {
  // do your own thing
    return utest_main(argc, argv);
}