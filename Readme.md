## 深入理解 Candle ML framework

按照Candle的源代码，进行理解分解, 在新版的（0.2.1）中原来的crate （candle-core）合并进入到统一的crate（candle）中了

### rust的机器学习库的特点

1. 模型的基本定义是通过结构体来定义
2. 定义的结构体主使用impl进行方法的扩展
   a. 方法中主要由``new``方法和``forward``方法实现具体的算法
3. 代码中``my-candle-kernels``模块是扩展``my-candle-core``的并行运算库

### candle-examples
采用candle的0.2.0版本，目前的candle功能还有很多不完善的地方
