#include <fstream>
#include <sstream>
#include <opencv2/videoio/videoio_c.h>
#include <opencv2/highgui.hpp>
#include "NvInfer.h"
#include "NvOnnxParser.h"
#include <cuda.h>
#include "cuda_runtime.h"  
#include "device_launch_parameters.h" 

#define _CRT_SECURE_NO_WARNINGS
#include <iostream>
#include <assert.h>
#include <ctime>
#include <vector>
#include <map>
#include "argparse.h"
#include "NanoDet.h"

using namespace nvinfer1;
using namespace nvonnxparser;

class Logger : public ILogger
{
    void log(Severity severity, const char* msg) override
    {
        // suppress info-level messages
        if (severity != Severity::kINFO)
            std::cout << msg << std::endl;
    }
} gLogger;

std::vector<float> get_3rd_point(std::vector<float>& a, std::vector<float>& b) {
	std::vector<float> direct{ a[0] - b[0],a[1] - b[1] };
	return std::vector<float>{b[0] - direct[1], b[1] + direct[0]};
}

std::vector<float> get_dir(float src_point_x, float src_point_y, float rot_rad) {
	float sn = sin(rot_rad);
	float cs = cos(rot_rad);
	std::vector<float> src_result{ 0.0,0.0 };
	src_result[0] = src_point_x * cs - src_point_y * sn;
	src_result[1] = src_point_x * sn + src_point_y * cs;
	return src_result;
}

void affine_tranform(float pt_x, float pt_y, cv::Mat& t, float* x, int p, int num) {
	float new1[3] = { pt_x, pt_y, 1.0 };
	cv::Mat new_pt(3, 1, t.type(), new1);
	cv::Mat w = t * new_pt;
	x[p] = w.at<float>(0, 0);
	/*std::cout << w.size() << std::endl;
	std::cout << t.size() << std::endl;
	std::cout << w.at<float>(0, 0) << " " << w.at<float>(1,0) << std::endl;*/
	x[p + num] = w.at<float>(1, 0);

}

cv::Mat get_affine_transform(std::vector<float>& center, std::vector<float>& scale, float rot, std::vector<int>& output_size, int inv) {
	std::vector<float> scale_tmp;
	scale_tmp.push_back(scale[0] * 200);
	scale_tmp.push_back(scale[1] * 200);
	float src_w = scale_tmp[0];
	int dst_w = output_size[0];
	int dst_h = output_size[1];
	float rot_rad = rot * 3.1415926535 / 180;
	std::vector<float> src_dir = get_dir(0, -0.5 * src_w, rot_rad);
	std::vector<float> dst_dir{ 0.0, float(-0.5) * dst_w };
	std::vector<float> src1{ center[0] + src_dir[0],center[1] + src_dir[1] };
	std::vector<float> dst0{ float(dst_w * 0.5),float(dst_h * 0.5) };
	std::vector<float> dst1{ float(dst_w * 0.5) + dst_dir[0],float(dst_h * 0.5) + dst_dir[1] };
	std::vector<float> src2 = get_3rd_point(center, src1);
	std::vector<float> dst2 = get_3rd_point(dst0, dst1);
	if (inv == 0) {
		float a[6][6] = { {center[0],center[1],1,0,0,0},
						  {0,0,0,center[0],center[1],1},
						  {src1[0],src1[1],1,0,0,0},
						  {0,0,0,src1[0],src1[1],1},
						  {src2[0],src2[1],1,0,0,0},
						  {0,0,0,src2[0],src2[1],1} };
		float b[6] = { dst0[0],dst0[1],dst1[0],dst1[1],dst2[0],dst2[1] };
		cv::Mat a_1 = cv::Mat(6, 6, CV_32F, a);
		cv::Mat b_1 = cv::Mat(6, 1, CV_32F, b);
		cv::Mat result;
		solve(a_1, b_1, result, 0);
		cv::Mat dst = result.reshape(0, 2);
		return dst;
	}
	else {
		float a[6][6] = { {dst0[0],dst0[1],1,0,0,0},
						  {0,0,0,dst0[0],dst0[1],1},
						  {dst1[0],dst1[1],1,0,0,0},
						  {0,0,0,dst1[0],dst1[1],1},
						  {dst2[0],dst2[1],1,0,0,0},
						  {0,0,0,dst2[0],dst2[1],1} };
		float b[6] = { center[0],center[1],src1[0],src1[1],src2[0],src2[1] };
		cv::Mat a_1 = cv::Mat(6, 6, CV_32F, a);
		cv::Mat b_1 = cv::Mat(6, 1, CV_32F, b);
		cv::Mat result;
		solve(a_1, b_1, result, 0);
		cv::Mat dst = result.reshape(0, 2);
		return dst;
	}
}


void transform_preds(float* coords, std::vector<float>& center, std::vector<float>& scale, std::vector<int>& output_size, std::vector<int64_t>& t, float* target_coords) {
	cv::Mat tran = get_affine_transform(center, scale, 0, output_size, 1);
	for (int p = 0; p < t[1]; ++p) {
		affine_tranform(coords[p], coords[p + t[1]], tran, target_coords, p, t[1]);
	}
}

void box_to_center_scale(std::vector<int>& box, int width, int height, std::vector<float>& center, std::vector<float>& scale) {
	int box_width = box[2] - box[0];
	int box_height = box[3] - box[1];
	center[0] = box[0] + box_width * 0.5;
	center[1] = box[1] + box_height * 0.5;
	float aspect_ratio = width * 1.0 / height;
	int pixel_std = 200;
	if (box_width > aspect_ratio * box_height) {
		box_height = box_width * 1.0 / aspect_ratio;
	}
	else if (box_width < aspect_ratio * box_height) {
		box_width = box_height * aspect_ratio;
	}
	scale[0] = box_width * 1.0 / pixel_std;
	scale[1] = box_height * 1.0 / pixel_std;
	if (center[0] != -1) {
		scale[0] = scale[0] * 1.25;
		scale[1] = scale[1] * 1.25;
	}
}

/*
* 该函数暂时只实现了batch为1的情况
*/
void get_max_preds(float* heatmap, std::vector<int64_t>& t, float* preds, float* maxvals) {
	int batch_size = t[0];
	int num_joints = t[1];
	int width = t[3];
	float* pred_mask = new float[num_joints * 2];
	int* idx = new int[num_joints * 2];
	for (int i = 0; i < batch_size; ++i) {
		for (int j = 0; j < num_joints; ++j) {
			float max = heatmap[i * num_joints * t[2] * t[3] + j * t[2] * t[3]];
			int max_id = 0;
			for (int k = 1; k < t[2] * t[3]; ++k) {
				int index = i * num_joints * t[2] * t[3] + j * t[2] * t[3] + k;
				if (heatmap[index] > max) {
					max = heatmap[index];
					max_id = k;
				}
			}
			maxvals[j] = max;
			idx[j] = max_id;
			idx[j + num_joints] = max_id;
		}
	}
	for (int i = 0; i < num_joints; ++i) {
		idx[i] = idx[i] % width;
		idx[i + num_joints] = idx[i + num_joints] / width;
		if (maxvals[i] > 0) {
			pred_mask[i] = 1.0;
			pred_mask[i + num_joints] = 1.0;
		}
		else {
			pred_mask[i] = 0.0;
			pred_mask[i + num_joints] = 0.0;
		}
		preds[i] = idx[i] * pred_mask[i];
		preds[i + num_joints] = idx[i + num_joints] * pred_mask[i + num_joints];
	}

}

void get_final_preds(float* heatmap, std::vector<int64_t>& t, std::vector<float>& center, std::vector<float> scale, float* preds) {
	float* coords = new float[t[1] * 2];
	float* maxvals = new float[t[1]];
	int heatmap_height = t[2];
	int heatmap_width = t[3];
	get_max_preds(heatmap, t, coords, maxvals);
	for (int i = 0; i < t[0]; ++i) {
		for (int j = 0; j < t[1]; ++j) {
			int px = int(coords[i * t[1] + j] + 0.5);
			int py = int(coords[i * t[1] + j + t[1]] + 0.5);
			int index = (i * t[1] + j) * t[2] * t[3];
			if (px > 1 && px < heatmap_width - 1 && py>1 && py < heatmap_height - 1) {
				float diff_x = heatmap[index + py * t[3] + px + 1] - heatmap[index + py * t[3] + px - 1];
				float diff_y = heatmap[index + (py + 1) * t[3] + px] - heatmap[index + (py - 1) * t[3] + px];
				coords[i * t[1] + j] += sign(diff_x) * 0.25;
				coords[i * t[1] + j + t[1]] += sign(diff_y) * 0.25;
			}
		}
	}
	std::vector<int> img_size{ heatmap_width,heatmap_height };
	transform_preds(coords, center, scale, img_size, t, preds);
}

int pair_line[] = {
	//0,2,
	//2,4,
	//4,6,
	6,8,
	8,10,
	6,12,
	12,14,
	14,16,

	//0,1,
	//1,3,
	//3,5,
	5,7,
	7,9,
	5,11,
	11,13,
	13,15,
};


int main(int argc, char** argv) {
	clock_t startTime_nano, endTime_nano, startTime_hrnet, endTime_hrnet;
	std::string videopath;
	cv::VideoCapture capture;
	cv::Mat frame, frame1;
	int t = 0;
	int model = 0;
	int k = 0;
	std::string write_videopath = "D:/write.mp4";
	argparse::ArgumentParser arg_parser("example", "Argument parser example");
	arg_parser.add_argument("-v",
		"--video",
		"Video path",
		false
	);
	arg_parser.add_argument("-c",
		"--camera",
		"camera index",
		false
	);
	arg_parser.add_argument("-m",
		"--model",
		"model type,0-w48_256x192,1-w48_384x288,2-w32_256x192,3-w32_128x96",
		false
	);
	arg_parser.add_argument("-d",
		"--display",
		"point display mode, 0-左右,1-左,2-右",
		false
	);
	arg_parser.add_argument("-w",
		"--write_video",
		"write video path",
		false
	);
	arg_parser.enable_help();
	arg_parser.parse(argc, (const char**)argv);
	if (arg_parser.exists("help")) {
		arg_parser.print_help();
		return 0;
	}
	if (arg_parser.exists("video")) {
		videopath = arg_parser.get<std::string>("video");
		frame = capture.open(videopath);
		k = 1;
	}
	else if (arg_parser.exists("camera")) {
		frame = capture.open(arg_parser.get<int>("camera"));
		k = 1;
	}
	if (arg_parser.exists("display")) {
		t = arg_parser.get<int>("display");
	}
	if (arg_parser.exists("model")) {
		model = arg_parser.get<int>("model");
	}
	if (arg_parser.exists("write_video")) {
		write_videopath = arg_parser.get<std::string>("write_video");
	}
	if (k == 0) {
		frame = capture.open(0);
	}
	if (!capture.isOpened()) {
		std::cout << "can't open" << std::endl;
		return -1;
	}
	NanoDet nanonet(320, 0.4, 0.5);

	size_t input_tensor_size_hrnet;

	std::map<const char*, std::vector<int64_t>> output_dim_hrnet;
	std::vector<const char*> input_node_names_hrnet;
	std::vector<const char*> output_node_names_hrnet;
	std::vector<int64_t> input_node_dims_hrnet;
	std::vector<int64_t> output_node_dims_hrnet;

	Ort::Env env(ORT_LOGGING_LEVEL_WARNING, "test");
	std::string model_hrnet_path = "s" + std::to_string(model) + ".onnx";

	input_tensor_size_hrnet = 1;
	std::string onnx_filename = model_hrnet_path;
	IBuilder* builder = createInferBuilder(gLogger);
	nvinfer1::INetworkDefinition* network = builder->createNetworkV2(1U << static_cast<uint32_t>(NetworkDefinitionCreationFlag::kEXPLICIT_BATCH));
	auto parser = nvonnxparser::createParser(*network, gLogger);
	parser->parseFromFile(onnx_filename.c_str(), 2);
	for (int i = 0; i < parser->getNbErrors(); ++i)
	{
		std::cout << parser->getError(i)->desc() << std::endl;
	}
	std::cout << "tensorRT load onnx model!" << std::endl;

	std::fstream _file;
	_file.open("s" + std::to_string(model) + ".trt", std::ios::in);
	if (!_file) //不存在trt文件则需要构建推理引擎并生成trt文件
	{
		std::cout << "trt模型文件不存在，开始生成！" << std::endl;
		// 构建推理引擎
		unsigned int maxBatchSize = 1;
		builder->setMaxBatchSize(maxBatchSize);
		IBuilderConfig* config = builder->createBuilderConfig();
		config->setMaxWorkspaceSize(1 << 20);
		ICudaEngine* engine = builder->buildEngineWithConfig(*network, *config);

		// 序列化模型
		IHostMemory* gieModelStream = engine->serialize();
		std::string serialize_str;
		std::ofstream serialize_output_stream;
		serialize_str.resize(gieModelStream->size());
		memcpy((void*)serialize_str.data(), gieModelStream->data(), gieModelStream->size());
		serialize_output_stream.open("s" + std::to_string(model) + ".trt", std::ios_base::out | std::ios_base::binary);
		serialize_output_stream << serialize_str;
		serialize_output_stream.close();
	}
	// 反序列化模型
	IRuntime* runtime = createInferRuntime(gLogger);
	std::string cached_path = "s" + std::to_string(model) + ".trt";
	std::ifstream fin(cached_path, std::ios_base::in | std::ios_base::binary);
	std::string cached_engine = "";
	while (fin.peek() != EOF) {
		std::stringstream buffer;
		buffer << fin.rdbuf();
		cached_engine.append(buffer.str());
	}
	fin.close();
	ICudaEngine* re_engine = runtime->deserializeCudaEngine(cached_engine.data(), cached_engine.size(), nullptr);
	IExecutionContext* context = re_engine->createExecutionContext();

	// 获取输入与输出名称，格式
	const char* input_blob_name = network->getInput(0)->getName();
	const char* output_blob_name = network->getOutput(0)->getName();
	printf("input_blob_name : %s \n", input_blob_name);
	printf("output_blob_name : %s \n", output_blob_name);

	const int inputH = network->getInput(0)->getDimensions().d[2];
	const int inputW = network->getInput(0)->getDimensions().d[3];
	printf("inputH : %d, inputW: %d \n", inputH, inputW);
	// 创建GPU显存输入/输出缓冲区
	void* buffers[2] = { NULL, NULL };
	int nBatchSize = 1;
	int nOutputSize = 10;
	// iterate over all input nodes
	for (int i = 0; i < network->getNbInputs(); i++) {
		// print input node names
		const char* input_name = network->getInput(i)->getName();
		printf("Input %d : name=%s\n", i, input_name);

		input_node_names_hrnet.push_back(input_name);
		// print input shapes/dims
		printf("Input %d : num_dims=%d\n", i, 4);
		for (int j = 0; j < 4; j++) {
			printf("Input %d : dim %d=%d\n", i, j, network->getInput(0)->getDimensions().d[j]);
			input_tensor_size_hrnet *= network->getInput(0)->getDimensions().d[j];
		}
	}
	for (int i = 0; i < network->getNbOutputs(); i++) {
		// print input node names
		const char* output_name = network->getOutput(i)->getName();
		printf("Output %d : name=%s\n", i, output_name);
		output_node_names_hrnet.push_back(output_name);
		auto x = network->getOutput(0)->getDimensions().d;
		output_node_dims_hrnet = { x[0] ,x[1], x[2], x[3] };
		nOutputSize = x[0] * x[1] * x[2] * x[3];
		output_dim_hrnet[output_name] = output_node_dims_hrnet;
		printf("Output %d : num_dims=%zu\n", i, output_node_dims_hrnet.size());
		for (int j = 0; j < output_node_dims_hrnet.size(); j++) {
			printf("Output %d : dim %d=%jd\n", i, j, output_node_dims_hrnet[j]);
		}
	}
	static const std::string kWinName = "HRNet";
	cv::namedWindow(kWinName, cv::WINDOW_KEEPRATIO || cv::WINDOW_NORMAL);
	float* x_hrnet = new float[input_tensor_size_hrnet];
	std::vector<int> last_box_max = { 0,0,0,0 };
	cv::Size videoSize(capture.get(CV_CAP_PROP_FRAME_WIDTH), capture.get(CV_CAP_PROP_FRAME_HEIGHT));
	cv::VideoWriter writer(write_videopath, CV_FOURCC('M', 'J', 'P', 'G'), 24, videoSize);
	while (capture.read(frame)) {
		startTime_nano = clock();//计时开始
		std::vector<int> box_max{ 0,0,0,0 };
		nanonet.detect(frame, box_max);
		/*
		int area_box_max = (box_max[2] - box_max[0]) * (box_max[3] - box_max[1]);
		int area_last_box_max = (last_box_max[2] - last_box_max[0]) * (last_box_max[3] - last_box_max[1]);
		if (box_max[2] == 0 || area_box_max < 0.5 * area_last_box_max) {
			box_max = last_box_max;
		}
		else {
			last_box_max=box_max;
		}
		*/
		endTime_nano = clock();//计时结束
		std::cout << "The nanodet run time is: " << (double)(endTime_nano - startTime_nano) / CLOCKS_PER_SEC << "s" << std::endl;
		startTime_hrnet = clock();//计时开始

		std::vector<float> center{ 0,0 }, scale{ 0,0 };
		std::vector<int> img_size;
		if (model == 1) {
			img_size = { 288,384 };
		}
		else if (model == 4) {
			img_size = { 128,96 };
		}
		else {
			img_size = { 192,256 };
		}

		box_to_center_scale(box_max, img_size[0], img_size[1], center, scale);
		cv::Mat input;
		cv::Mat tran = get_affine_transform(center, scale, 0, img_size, 0);
		cv::warpAffine(frame, input, tran, cv::Size(img_size[0], img_size[1]), cv::INTER_LINEAR);
		_normalize(input);


		convertMat2pointer(input, x_hrnet);
		endTime_nano = clock();//计时结束

		std::cout << "The pre time is: " << (double)(endTime_nano - startTime_hrnet) / CLOCKS_PER_SEC << "s" << std::endl;
		
		cudaMalloc(&buffers[0], input_tensor_size_hrnet * sizeof(float));
		cudaMalloc(&buffers[1], nOutputSize * sizeof(float));

		// 创建cuda流
		cudaStream_t stream;
		cudaStreamCreate(&stream);
		void* data = malloc(input_tensor_size_hrnet * sizeof(float));
		memcpy(data, x_hrnet, input_tensor_size_hrnet * sizeof(float));

		// 内存到GPU显存
		cudaMemcpyAsync(buffers[0], data, input_tensor_size_hrnet * sizeof(float), cudaMemcpyHostToDevice, stream);

		// 推理
		context->enqueueV2(buffers, stream, nullptr);

		// 显存到内存
		float *output_tensors_hrnet = new float[nOutputSize];
		cudaMemcpyAsync(output_tensors_hrnet, buffers[1], nOutputSize * sizeof(float), cudaMemcpyDeviceToHost, stream);
		// output_tensors_hrnet是模型得到的结果

		// 同步结束，释放资源
		cudaStreamSynchronize(stream);
		cudaStreamDestroy(stream);
		endTime_nano = clock();//计时结束

		std::cout << "The trt time is: " << (double)(endTime_nano - startTime_hrnet) / CLOCKS_PER_SEC << "s" << std::endl;
		float* preds = new float[output_dim_hrnet[output_node_names_hrnet[0]][1] * 2 + 2];
		// 后处理
		get_final_preds(output_tensors_hrnet, output_dim_hrnet[output_node_names_hrnet[0]], center, scale, preds);
		//std::cout << "后处理结束！" << std::endl;
		preds[34] = (preds[5] + preds[6]) / 2;
		preds[35] = (preds[5 + 17] + preds[+17]) / 2;
		int line_begin, line_end, iter, point_begin;
		if (t == 0) {//所有关节点
			line_begin = 0;
			line_end = 20;
			iter = 1;
			point_begin = 0;
		}
		//左侧
		else if (t == 1) {
			line_begin = 0;
			line_end = 10;
			iter = 2;
			point_begin = 0;
		}
		//右侧
		else if (t == 2) {
			line_begin = 10;
			line_end = 20;
			iter = 2;
			point_begin = 1;
		}
		
		rectangle(frame, cv::Point(box_max[0], box_max[1]), cv::Point(box_max[2], box_max[3]), cv::Scalar(0, 0, 255), 3);
		for (int i = line_begin; i < line_end; i = i + 2) {
			cv::line(frame, cv::Point2d(int(preds[pair_line[i]]), int(preds[pair_line[i] + 17])),
				cv::Point2d(int(preds[pair_line[i + 1]]), int(preds[pair_line[i + 1] + 17])), (0, 0, 255), 4);
		}
		for (int i = point_begin; i < 17; i = i + iter) {
			int x_coord = int(preds[i]);
			int y_coord = int(preds[i + 17]);
			cv::circle(frame, cv::Point2d(x_coord, y_coord), 1, (0, 255, 0), 2);
		}
		endTime_hrnet = clock();//计时结束
		std::cout << "The hrnet run time is: " << (double)(endTime_hrnet - startTime_hrnet) / CLOCKS_PER_SEC << "s" << std::endl;
		std::cout << "The run time is: " << (double)(endTime_hrnet - startTime_nano) / CLOCKS_PER_SEC << "s" << std::endl;
		std::string label = cv::format("%.2f", 1.0 / (double)(endTime_hrnet - startTime_nano) * CLOCKS_PER_SEC);
		putText(frame, label + "FPS", cv::Point(50, 50), cv::FONT_HERSHEY_SIMPLEX, 0.75, cv::Scalar(0, 255, 0), 1);
		writer.write(frame);
		imshow(kWinName, frame);
		if (cv::waitKey(1) == 27)
			break;
	}
	writer.release();
	cv::destroyAllWindows();
    return 0;
}