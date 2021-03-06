#include "utils.h"
/*
将数组转换为Mat，即指针转换为Mat
*/
cv::Mat Array2Mat(float* array, std::vector<int64_t> t){
	int row = t[0];
	int col = t[1];
	if (t.size() == 2) {
		cv::Mat img(row, col, CV_32F);
		for (int i = 0; i < img.rows; ++i) {
			float* cur_row = img.ptr<float>(i);
			for (int j = 0; j < img.cols; ++j) {
				*cur_row++ = array[i * col + j];
			}
		}
		return img;
	}
	else {
		int sizes[] = { t[0],t[1],t[2],t[3] };
		cv::Mat img = cv::Mat(4, sizes, CV_32F, array);
		return img;
	}
}

/*
图像的预处理，减去mean除以std
*/
void _normalize(cv::Mat& img){
	img.convertTo(img, CV_32F);
	for (int i = 0; i < img.rows; ++i) {
		float* cur_row = img.ptr<float>(i);
		for (int j = 0; j < img.cols; ++j) {
			*cur_row++ = (*cur_row - mean1[0]) / std1[0];
			*cur_row++ = (*cur_row - mean1[1]) / std1[1];
			*cur_row++ = (*cur_row - mean1[2]) / std1[2];
		}
	}
}

/*
mat转换为数组，即指针
*/
void convertMat2pointer(cv::Mat& img, float* x){
	for (int i = 0; i < img.rows; ++i) {

		float* cur_row = img.ptr<float>(i);
		for (int j = 0; j < img.cols; ++j) {

			x[i * img.cols + j] = *cur_row++;
			x[img.rows * img.cols + i * img.cols + j] = *cur_row++;
			x[img.rows * img.cols * 2 + i * img.cols + j] = *cur_row++;
		}
	}
}

/*
得到数字的符号
*/
int sign(float x) {
	int w = 0;
	if (x > 0) {
		w = 1;
	}
	else if (x == 0) {
		w = 0;
	}
	else {
		w = -1;
	}
	return w;
}