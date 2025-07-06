# TinyNdArray

C++ 实现的 NumPy NdArray。

期待您的 Pull Request。

## 要求

* C++14 编译器

## 示例代码
```cpp
#include "tinyndarray.h"

using tinyndarray::NdArray;
using tinyndarray::DataType;

int main() {
    // 创建整型数组
    auto int_arr = NdArray(DataType::INT, {2, 3}, {1, 2, 3, 4, 5, 6});
    std::cout << "整型数组:\n" << int_arr << std::endl;
    
    // 创建双精度浮点数组
    auto double_arr = NdArray(DataType::DOUBLE, {3}, {1.5, 2.5, 3.5});
    std::cout << "双精度数组:\n" << double_arr << std::endl;
    
    // 混合类型运算
    auto result = int_arr + double_arr;
    std::cout << "混合类型运算结果:\n" << result << std::endl;
    
    return 0;
}
```
## 使用说明

在CMakeLists中作为子项目引入：

```bash
add_subdirectory(tinyndarray)
target_link_libraries(demo PRIVATE tinyndarray)
```

## 快速指南

TinyNdArray 支持多种数据类型：`FLOAT`, `INT`, `DOUBLE`。

以下示例中，C++ 代码假设已声明 `using namespace tinyndarray;`。

### 数据类型支持

| 数据类型 | C++ 类型 | 枚举值        |
|----------|----------|---------------|
| 浮点数   | float    | DataType::FLOAT |
| 整数     | int      | DataType::INT   |
| 双精度   | double   | DataType::DOUBLE|

### 复制行为

NdArray 的复制行为是浅拷贝，与 NumPy 相同。

<table>
<tr align="center">
    <td> <b>Numpy (Python)</b> </td>
    <td> <b>TinyNdArray (C++)</b> </td>
</tr>
<td><pre lang="python">
    a = np.ones((2, 3))
    b = a
    b[0, 0] = -1
    print(a[0, 0])  # -1
</pre></td>
<td><pre lang="cpp">
    auto a = NdArray::Ones(DataType::FLOAT, {2, 3});
    auto b = a;
    b[{0, 0}] = -1;
    std::cout << a[{0, 0}] << std::endl;  // -1
</pre></td>
</tr>
</table>

### 基础构造

| **Numpy (Python)**                     | **TinyNdArray (C++)**                                |
|----------------------------------------|-----------------------------------------------------|
| `a = np.array([2, 3])`                 | `NdArray a(DataType::INT, {2}, {2, 3});`            |
|                                        | `NdArray a = {2, 3}; // 默认为FLOAT类型`            |

### 初始化器（支持多种类型）

| **Numpy (Python)**                     | **TinyNdArray (C++)**                                |
|----------------------------------------|-----------------------------------------------------|
| `a = np.array([1.0, 2.0])`             | `NdArray a = {1.0f, 2.0f}; // FLOAT类型`            |
| `a = np.array([[1.0, 2.0]])`           | `NdArray a = {{1.0f, 2.0f}};`                       |
| `a = np.array([[1,2],[3,4]])`          | `NdArray a(DataType::INT, {{1,2}, {3,4}});`         |
| `a = np.array([[[[1.5,2.5]]]])`        | `NdArray a = {{{{1.5, 2.5}}}}; // 4维数组`          |

### 静态初始化

| **Numpy (Python)**                     | **TinyNdArray (C++)**                                |
|----------------------------------------|-----------------------------------------------------|
| `a = np.empty((2,3))`                  | `auto a = NdArray::Empty(DataType::FLOAT, {2,3});`  |
| `a = np.zeros((2,3))`                  | `auto a = NdArray::Zeros(DataType::INT, {2,3});`    |
| `a = np.ones((2,3))`                   | `auto a = NdArray::Ones(DataType::DOUBLE, {2,3});`  |
| `a = np.arange(10)`                    | `auto a = NdArray::Arange(DataType::INT, 10);`      |
| `a = np.arange(0,100,10)`              | `auto a = NdArray::Arange(DataType::INT, 0,100,10);`|

### 随机初始化

| **Numpy (Python)**                     | **TinyNdArray (C++)**                                |
|----------------------------------------|-----------------------------------------------------|
| `a = np.random.uniform(size=10)`       | `auto a = NdArray::Uniform(DataType::FLOAT, {10});` |
| `a = np.random.uniform(size=(2,10))`   | `auto a = NdArray::Uniform(DataType::FLOAT, {2,10});`|
| `a = np.random.uniform(0.0,1.0,10)`    | `auto a = NdArray::Uniform(0.0f,1.0f, {10});`      |
| `a = np.random.normal(size=10)`        | `auto a = NdArray::Normal(DataType::FLOAT, {10});`  |
| `a = np.random.normal(0.0,1.0,10)`     | `auto a = NdArray::Normal(0.0f,1.0f, {10});`       |

### 随机种子

| **Numpy (Python)**                     | **TinyNdArray (C++)**                                |
|----------------------------------------|-----------------------------------------------------|
| `np.random.seed()`                     | `NdArray::Seed();`                                  |
| `np.random.seed(0)`                    | `NdArray::Seed(0);`                                 |

### 基本内置方法

| **Numpy (Python)** | **TinyNdArray (C++)**      |
|--------------------|----------------------------|
| `id(a)`            | `a.id()`                   |
| `a.size`           | `a.size()`                 |
| `a.shape`          | `a.shape()`                |
| `a.ndim`           | `a.ndim()`                 |
| `a.fill(2.0)`      | `a.fill(2.0f); // FLOAT`   |
|                    | `a.fill(2); // INT`        |
| `a.copy()`         | `a.copy()`                 |

### 原始内置方法

| **Numpy (Python)** | **TinyNdArray (C++)**      |
|--------------------|----------------------------|
| ---                | `a.empty()`                |
| ---                | `a.data_ptr()`             |
| ---                | `a.begin()`                |
| ---                | `a.end()`                  |

### 单元素类型转换

| **Numpy (Python)** | **TinyNdArray (C++)**      |
|--------------------|----------------------------|
| `float(a[0])`      | `static_cast<float>(a(0));`|
| `int(a[0])`        | `static_cast<int>(a(0));`  |

### 索引访问

| **Numpy (Python)** | **TinyNdArray (C++)**      |
|--------------------|----------------------------|
| `a[2, -3]`         | `a({2, -3})` 或 `a(2, -3)` |

### 形状变换方法

| **Numpy (Python)**         | **TinyNdArray (C++)**            |
|----------------------------|----------------------------------|
| `a.reshape(-1, 2, 1)`      | `a.reshape({-1, 2, 1});`         |
| `a.flatten()`              | `a.flatten();`                   |
| `a.ravel()`                | `a.ravel();`                     |

### 形状变换函数

| **Numpy (Python)**         | **TinyNdArray (C++)**            |
|----------------------------|----------------------------------|
| `np.reshape(a, (-1,2,1))`  | `Reshape(a, {-1,2,1});`          |
| `np.squeeze(a)`            | `Squeeze(a);`                    |
| `np.expand_dims(a, 1)`     | `ExpandDims(a, 1);`              |

### 切片

| **Numpy (Python)**     | **TinyNdArray (C++)**            |
|------------------------|----------------------------------|
| `a[1:5, -4:-1]`        | `a.slice({{1,5}, {-4,-1}});`     |

### 打印

| **Numpy (Python)**     | **TinyNdArray (C++)**            |
|------------------------|----------------------------------|
| `print(a)`             | `std::cout << a << std::endl;`   |

### 单目运算符

| **Numpy (Python)**         | **TinyNdArray (C++)**            |
|----------------------------|----------------------------------|
| `+a`                       | `+a`                             |
| `-a`                       | `-a`                             |

### 算术运算符

| **Numpy (Python)**                     | **TinyNdArray (C++)**                                |
|----------------------------------------|-----------------------------------------------------|
| `a + b`                                | `a + b` // 支持混合类型                             |
| `a - b`                                | `a - b`                                             |
| `a * b`                                | `a * b`                                             |
| `a / b`                                | `a / b`                                             |
| `a + 2.0`                              | `a + 2.0f`                                          |
| `2.0 * a`                              | `2.0f * a`                                          |

### 比较运算符

| **Numpy (Python)**                     | **TinyNdArray (C++)**                                |
|----------------------------------------|-----------------------------------------------------|
| `a == b`                               | `a == b` // 返回布尔数组                           |
| `a > b`                                | `a > b`                                             |
| `a == 1.0`                             | `a == 1.0f`                                         |
| `1.0 < a`                              | `1.0f < a`                                          |

### 复合赋值运算符

| **Numpy (Python)**                     | **TinyNdArray (C++)**                                |
|----------------------------------------|-----------------------------------------------------|
| `a += b`                               | `a += b`                                            |
| `a *= 2.0`                             | `a *= 2.0f`                                         |

### 数学函数

| **Numpy (Python)**                     | **TinyNdArray (C++)**                                |
|----------------------------------------|-----------------------------------------------------|
| `np.abs(a)`                            | `Abs(a)`                                            |
| `np.sin(a)`                            | `Sin(a)`                                            |
| `np.power(a, 2.0)`                     | `Power(a, 2.0f)`                                    |
| `np.power(2.0, a)`                     | `Power(2.0f, a)`                                    |

### 轴函数（归约操作）

| **Numpy (Python)**                     | **TinyNdArray (C++)**                                |
|----------------------------------------|-----------------------------------------------------|
| `np.sum(a)`                            | `Sum(a)`                                            |
| `np.sum(a, axis=0)`                    | `Sum(a, {0})`                                       |
| `np.mean(a, axis=0)`                   | `Mean(a, {0})`                                      |

### 逻辑函数

| **Numpy (Python)**                     | **TinyNdArray (C++)**                                |
|----------------------------------------|-----------------------------------------------------|
| `np.all(a)`                            | `All(a)`                                            |
| `np.any(a)`                            | `Any(a)`                                            |
| `np.where(cond, x, y)`                 | `Where(cond, x, y)`                                 |

### 轴方法（归约操作）

| **Numpy (Python)**                     | **TinyNdArray (C++)**                                |
|----------------------------------------|-----------------------------------------------------|
| `a.sum()`                              | `a.sum()`                                           |
| `a.sum(axis=0)`                        | `a.sum({0})`                                        |
| `a.min(axis=0)`                        | `a.min({0})`                                        |

### 组合函数

| **Numpy (Python)**                     | **TinyNdArray (C++)**                                |
|----------------------------------------|-----------------------------------------------------|
| `np.stack([a,b], axis=0)`              | `Stack({a, b}, 0);`                                 |
| `np.concatenate([a,b], axis=0)`        | `Concatenate({a, b}, 0);`                           |

### 视图变换函数

| **Numpy (Python)**                     | **TinyNdArray (C++)**                                |
|----------------------------------------|-----------------------------------------------------|
| `np.transpose(a)`                      | `Transpose(a);`                                     |
| `np.broadcast_to(a, (3,2))`            | `BroadcastTo(a, {3,2});`                            |

### 文件I/O

| **Numpy (Python)**                     | **TinyNdArray (C++)**                                |
|----------------------------------------|-----------------------------------------------------|
| `np.load('m.npy')`                     | `Load("m.npy");`                                    |
| `a.save('m.npy')`                     | `a.Save("m.npy");`                                    |

### 矩阵乘积

| **Numpy (Python)**                     | **TinyNdArray (C++)**                                |
|----------------------------------------|-----------------------------------------------------|
| `np.dot(a, b)`                         | `Dot(a, b);`                                        |
| `a.dot(b)`                             | `a.dot(b);`                                         |

### 逆矩阵

| **Numpy (Python)**                     | **TinyNdArray (C++)**                                |
|----------------------------------------|-----------------------------------------------------|
| `np.linalg.inv(a)`                     | `Inv(a);`                                           |

### 算术运算

支持广播和混合类型：

```cpp
auto int_arr = NdArray::Ones(DataType::INT, {2, 3});
auto float_arr = NdArray::Ones(DataType::FLOAT, {3});
auto result = int_arr * float_arr; // 结果为FLOAT类型
```

### 矩阵运算

```cpp
auto a = NdArray::Ones(DataType::FLOAT, {3, 4});
auto b = NdArray::Ones(DataType::FLOAT, {4, 5});
auto c = a.dot(b); // 矩阵乘法
```

### 归约操作

```cpp
auto arr = NdArray::Arange(DataType::INT, 12).reshape(3, 4);
auto sum_all = arr.sum(); // 所有元素求和
auto sum_axis0 = arr.sum({0}); // 沿0轴求和
```

### 并行执行

默认情况下，大多数操作使用多线程并行执行。

```cpp
// 使用所有核心（默认）
NdArray::SetNumWorkers(-1);

// 使用4个核心
NdArray::SetNumWorkers(4);

// 禁用并行
NdArray::SetNumWorkers(1);
```


## 待实现功能

* [ ] 优化轴归约算法
* [ ] 实现引用切片（无拷贝）
* [ ] 引入SIMD指令优化
* [ ] 改进矩阵求逆算法（LU分解）

## 贡献指南

欢迎提交 Pull Request！请确保：
1. 代码符合C++14标准
2. 添加相应的单元测试
3. 更新文档和示例

## 许可证

MIT 许可证