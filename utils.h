#include <vector>
#include <opencv2/highgui.hpp>

const float mean1[3] = { 103.53, 116.28, 123.675 };
const float std1[3] = { 57.375, 57.12, 58.395 };

cv::Mat Array2Mat(float* array, std::vector<int64_t> t);
void _normalize(cv::Mat& img);
void convertMat2pointer(cv::Mat& img, float* x);
void convertMat2pointer(cv::Mat& img, float* x);
int sign(float x);