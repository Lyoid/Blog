# caffe的C++接口应用

caffe模型应用到其他应用程序中的几个步骤：

1. 利用caffe训练生成 .caffemodel 文件，同时保留网络结构的 .prototxt 文件。
2. 在c++程序中调用 caffe/caffe.hpp 头文件 来导入接口函数。

在 caffe/examples/cpp_classification 中有 classification.cpp 文件，它是官方提供的应用实例。



## 编译caffe

新建cbuild文件，在里面执行：

```
cmake ..
sudo make -j8
sudo make install
```

编译完成之后，出现 caffe.pb.h 丢失的问题：

```
/home/ubuntu/caffe/include/caffe/blob.hpp:9:34: fatal error: caffe/proto/caffe.pb.h: No such file or directory
```

**解决方法：** 用protoc从caffe/src/caffe/proto/caffe.proto生成caffe.pb.h和caffe.pb.cc

```
protoc ./src/caffe/proto/caffe.proto  --cpp_out=./include/caffe/
```

然后在 /caffe/include/caffe/ 目录下新建 proto 目录，将 caffe.pb.h 拷贝到里面。



## 运行实例

首先下载缺省的文件：

​	在caffe\data\ilsvrc12目录下：

```
./get_ilsvrc_aux.sh
```

​	在~/caffe/models/bvlc_reference_caffenet目录下，下载权值文件：

​	http://dl.caffe.berkeleyvision.org/bvlc_reference_caffenet.caffemodel

这里使用网上的一个实例来实现调用：​	

​	classify.cpp 文件：

```c++
#include "caffe/caffe.hpp"
#include <string>
#include <opencv2/opencv.hpp>

using namespace std;
using namespace cv;
using namespace caffe;

//用于表存输出结果的，string保存的预测结果对应的字符，如cat；float表示概率
typedef pair<string, float> Prediction;

// 函数Argmax（）需要用到的子函数
static bool PairCompare(const std::pair<float, int>& lhs,
                        const std::pair<float, int>& rhs) {
    return lhs.first > rhs.first;
}

// 返回预测结果中概率从大到小的前N个预测结果的索引
static std::vector<int> Argmax(const std::vector<float>& v, int N) {
    std::vector<std::pair<float, int> > pairs;
    for (size_t i = 0; i < v.size(); ++i)
        pairs.push_back(std::make_pair(v[i], i));
    std::partial_sort(pairs.begin(), pairs.begin() + N, pairs.end(), PairCompare);

    std::vector<int> result;
    for (int i = 0; i < N; ++i)
        result.push_back(pairs[i].second);
    return result;
}

int main(int argc, char** argv)
{
    // 定义模型配置文件，模型文件，均值文件，标签文件以及带分类的图像
    string model_file   = "/home/lyoid/caffe/models/bvlc_reference_caffenet/deploy.prototxt";
    string trained_file = "/home/lyoid/caffe/models/bvlc_reference_caffenet/bvlc_reference_caffenet.caffemodel";
    string label_file   = "/home/lyoid/caffe/data/ilsvrc12/synset_words.txt";
    string img_file     = "/home/lyoid/caffe/examples/images/cat.jpg";
    string mean_file    = "/home/lyoid/caffe/data/ilsvrc12/imagenet_mean.binaryproto";
    Mat img = imread(img_file);

    // 定义变量
    shared_ptr<Net<float> > net_;// 保存模型
    Size input_geometry_; // 模型输入图像的尺寸
    int num_channels_; // 图像的通道数
    Mat mean_; // 根据均值文件计算得到的均值图像
    vector<string> labels_; // 标签向量

    Caffe::set_mode(Caffe::GPU); // 是否使用GPU
    net_.reset(new Net<float>(model_file, TEST)); // 加载配置文件，设定模式为分类
    net_->CopyTrainedLayersFrom(trained_file); // 根据训练好的模型修改模型参数

    Blob<float>* input_layer = net_->input_blobs()[0]; // 定义输入层变量
    num_channels_ = input_layer->channels(); // 得到输入层的通道数
    LOG(INFO) << "num_channels_:" << num_channels_; // 输出通道数
    input_geometry_ = Size(input_layer->width(), input_layer->height()); // 得到输入层的图像大小

    // 处理均值文件，得到均值图像
    BlobProto blob_proto;
    ReadProtoFromBinaryFileOrDie(mean_file.c_str(), &blob_proto); // mean_file.c_str()将string类型转化为字符型
    Blob<float> mean_blob;
    mean_blob.FromProto(blob_proto);
    vector<Mat> channels;
    float* data = mean_blob.mutable_cpu_data();// data指针
    for (int i = 0; i < num_channels_; i++)
    {
        Mat channel(mean_blob.height(), mean_blob.width(), CV_32FC1, data);//将一副单通道图像的数据记录再channel中
        channels.push_back(channel);
        data += mean_blob.height() * mean_blob.width();// data指向下一个通道的开始
    }
    Mat mean;
    merge(channels, mean); //分离的通道融合，查看cv：：merge的作用
    Scalar channel_mean = cv::mean(mean);
    mean_ = Mat(input_geometry_, mean.type(), channel_mean);//得到均值图像

    // 得到标签
    ifstream labels(label_file.c_str());
    string line;
    while (getline(labels, line))
        labels_.push_back(string(line));
    //判断标签的类数和模型输出的类数是否相同
    Blob<float>* output_layer = net_->output_blobs()[0];
    LOG(INFO) << "output_layer dimension: " << output_layer->channels()
              << "; labels number: " << labels_.size();


    // 预测图像
    input_layer->Reshape(1, num_channels_, input_geometry_.height, input_geometry_.width);
    net_->Reshape();

    //将input_channels指向模型的输入层相关位置（大概是这样的）
    vector<Mat> input_channels;
    int width = input_layer->width();
    int height = input_layer->height();
    float* input_data = input_layer->mutable_cpu_data();
    for (int i = 0; i < input_layer->channels(); i++)
    {
        Mat channel(height, width, CV_32FC1, input_data);
        input_channels.push_back(channel);
        input_data += width * height;
    }

    //改变图像的大小等
    Mat sample;
    if (img.channels() == 3 && num_channels_ == 1)
        cv::cvtColor(img, sample, cv::COLOR_BGR2GRAY);
    else if (img.channels() == 4 && num_channels_ == 1)
        cv::cvtColor(img, sample, cv::COLOR_BGRA2GRAY);
    else if (img.channels() == 4 && num_channels_ == 3)
        cv::cvtColor(img, sample, cv::COLOR_BGRA2BGR);
    else if (img.channels() == 1 && num_channels_ == 3)
        cv::cvtColor(img, sample, cv::COLOR_GRAY2BGR);
    else
        sample = img;
    // change img size
    cv::Mat sample_resized;
    if (sample.size() != input_geometry_)
        cv::resize(sample, sample_resized, input_geometry_);
    else
        sample_resized = sample;
    // change img to float
    cv::Mat sample_float;
    if (num_channels_ == 3)
        sample_resized.convertTo(sample_float, CV_32FC3);
    else
        sample_resized.convertTo(sample_float, CV_32FC1);
    // img normalize
    cv::Mat sample_normalized;
    cv::subtract(sample_float, mean_, sample_normalized);

    //将图像通过input_channels变量传递给模型
    /* This operation will write the separate BGR planes directly to the
     * input layer of the network because it is wrapped by the cv::Mat
     * objects in input_channels. */
    cv::split(sample_normalized, input_channels);
    // 调用模型进行预测
    net_->Forward();

    // 得到输出
    const float* begin = output_layer->cpu_data();
    const float* end = begin + output_layer->channels();
    //将输出给vector容器
    vector<float> output = vector<float>(begin, end);
    //显示概率前N大的结果
    int N = 10;
    N = std::min<int>(labels_.size(), N);
    std::vector<int> maxN = Argmax(output, N);
    std::vector<Prediction> predictions;
    for (int i = 0; i < N; ++i) {
        int idx = maxN[i];
        predictions.push_back(std::make_pair(labels_[idx], output[idx]));
    }
    for (size_t i = 0; i < predictions.size(); ++i) {
        Prediction p = predictions[i];
        std::cout << std::fixed << std::setprecision(4) << p.second << " - \""
                  << p.first << "\"" << std::endl;
    }

    return 0;
}// end for main

```

对应CMakeLists.txt 文件：

```
cmake_minimum_required(VERSION 2.8.3)
project(classify)
set(CUDA_USE_STATIC_CUDA_RUNTIME OFF)
#设置caffe的根目录，需改成你的路径
set(Caffe_DIR /home/lyoid/caffe/cbuild)
#设置OpenCV的根目录，需改成你的路径
set(OpenCV_DIR /home/lyoid/下载/opencv-3.4.0/build)

find_package(Caffe)
include_directories(${Caffe_INCLUDE_DIRS})
find_package(OpenCV REQUIRED)
include_directories(${OpenCV_INCLUDE_DIRS})

include_directories(include
        ${OpenCV_INCLUDE_DIRS}
        ${Caffe_INCLUDE_DIRS}
        )

set(CPP_SOURCES src/classify.cpp)

add_executable(classify ${CPP_SOURCES})
target_link_libraries(classify
        ${OpenCV_LIBS}
        ${Caffe_LIBRARIES}
        )

```

注意：这里使用opencv3.4的api，由于之前尝试3.1的api会报错：

```
/usr/bin/ld: 找不到 -lopencv_cudabgsegm
```

改成3.4问题解决。

调试，遭遇问题：

```
cudnn_conv_layer.cpp:53] Check failed: status == CUDNN_STATUS_SUCCESS (4 vs. 0)  CUDNN_STATUS_INTERNAL_ERROR
```

解决：获取管理员权限运行。