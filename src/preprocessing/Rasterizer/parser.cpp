#include <iostream>
#include <cstring>
#include <opencv2/opencv.hpp>
extern "C"
{
struct Matrix
{
	float data[16];
};
std::vector<cv::Mat> depths;
std::vector<cv::Mat> colors;
std::vector<Matrix> cam2worlds;

Matrix intrinsic_matrix, extrinsic_matrix;

int depth_width, depth_height, color_width, color_height;

void Parse(const char* sens_file) {
	FILE* fp = fopen(sens_file, "rb");

	depths.clear();
	colors.clear();
	cam2worlds.clear();

	fread(&depth_width, sizeof(int), 1, fp);
	fread(&depth_height, sizeof(int), 1, fp);
	fread(&color_width, sizeof(int), 1, fp);
	fread(&color_height, sizeof(int), 1, fp);
	fread(&intrinsic_matrix, sizeof(float), 16, fp);
	fread(&extrinsic_matrix, sizeof(float), 16, fp);

	int depth_len, color_len;
	Matrix cam2world;

	while(!feof(fp)) {
		cv::Mat depth(480, 640, CV_32F);
		fread(&depth_len, sizeof(int), 1, fp);
        fread(depth.data, sizeof(uint8_t), depth_len, fp);

        fread(&color_len, sizeof(int), 1, fp);
    	cv::Mat jpg(1, color_len, CV_8UC1);
        fread(jpg.data, sizeof(uint8_t), color_len, fp);
    	cv::Mat color = cv::imdecode(jpg, 1);
        
        fread(cam2world.data, sizeof(float), 16, fp);

        if (color.rows == color_height && color.cols == color_width) {
	        depths.push_back(depth);
	        colors.push_back(color);
	        cam2worlds.push_back(cam2world);
	    }
	}

}

int CH() {
	return color_height;
}
int CW() {
	return color_width;
}
int DH() {
	return depth_height;
}
int DW() {
	return depth_width;
}

int Frames()
{
	return depths.size();
}

void GetData(float* depth_data, unsigned char* color_data, float* cam2world_data, Matrix* intrinsic_data) {
	memcpy(intrinsic_data, &intrinsic_matrix, sizeof(Matrix));
	memcpy(cam2world_data, cam2worlds.data(), sizeof(Matrix) * cam2worlds.size());
	for (int i = 0; i < depths.size(); ++i) {
		memcpy(depth_data, depths[i].data, sizeof(float) * depth_width * depth_height);
		depth_data += depth_width * depth_height;
		memcpy(color_data, colors[i].data, sizeof(unsigned char) * 3 * color_width * color_height);
		color_data += color_width * color_height * 3;
	}
}

void Clear()
{
	depths.clear();
	colors.clear();
	cam2worlds.clear();
}

std::vector<std::vector<int> > pose_pairs;
void GetPosePair(float* poses, int num_pose) {
	pose_pairs.resize(num_pose);
	for (int i = 0; i < num_pose; ++i) {
		for (int j = i; j < num_pose; ++j) {
			float* pose_i = poses + i * 16;
			float* pose_j = poses + j * 16;
			double angle = pose_i[2] * pose_j[2] + pose_i[6] * pose_j[6] + pose_i[10] * pose_j[10];
			double offset_x = pose_i[3] - pose_j[3];
			double offset_y = pose_i[7] - pose_j[7];
			double offset_z = pose_i[11] - pose_j[11];
			double distance = sqrt(offset_x * offset_x + offset_y * offset_y + offset_z * offset_z);
			if (angle > cos(15.0 / 180.0 * 3.141592654) && distance < 1) {
				pose_pairs[i].push_back(j);
				pose_pairs[j].push_back(i);
			}
		}
	}
}

int GetPosePairNumForID(int id) {
	return pose_pairs[id].size();
}

void GetPosePairForID(int* pair, int id) {
	memcpy(pair, pose_pairs[id].data(), sizeof(int) * pose_pairs[id].size());
}
}
