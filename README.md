# HRNet_TensorRT
HRNet的TensorRT加速，编译器是VS2019，具体可以看博客[人体姿态估计 c++版 HRNet tensorrt加速](https://jiahui.blog.csdn.net/article/details/120437862)

关于环境的配置可以自行百度。主要是opencv的cuda版、onnx推理库以及TensorRT Windows版本的配置。该项目是Release x64版本

使用方法：

```
HRNet-C++.exe -v E:\ytxs111.mp4 -m 2
```
参数的含义如下，-h可以查看

```
Usage: example [options...]
Options:
    -v, --video            Video path
    -c, --camera           camera index
    -m, --model            model type,0-w48_256x192,1-w48_384x288,2-w32_256x192,3-w32_128x96
    -d, --display          point display mode, 0-左右,1-左,2-右
    -w, --write_video      write video path
    -h, --help             Shows this page
```
