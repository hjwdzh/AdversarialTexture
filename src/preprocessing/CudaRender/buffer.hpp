#ifndef BUFFER_H_
#define BUFFER_H_

#include <cuda_runtime.h>
#include <glm/glm.hpp>
//#include <opencv2/opencv.hpp>
#include <vector>

class VertexBuffer
{
public:
	VertexBuffer();
	~VertexBuffer();

	void Reset();

	void SetPositions(const std::vector<glm::vec3>& positions);
	void SetNormals(const std::vector<glm::vec3>& normals);
	void SetColors(const std::vector<glm::vec3>& colors);
	void SetTexcoords(const std::vector<glm::vec2>& texcoords);
	void SetIndices(const std::vector<glm::ivec3>& indices);

	glm::vec3* d_positions;
	glm::vec3* d_colors;
	glm::vec3* d_normals;
	glm::vec2* d_texcoords;

	glm::ivec3* d_indices;

	glm::mat3 rotation;
	glm::vec3 translation;
	int num_positions;
	int num_colors;
	int num_normals;
	int num_texcoords;
	int num_indices;
};

class FrameBuffer
{
public:
	FrameBuffer();
	FrameBuffer(int rows, int cols, float cx, float cy, float fx, float fy);
	~FrameBuffer();

	void Create(int rows, int cols, float cx, float cy, float fx, float fy);
	void Initialize(int rows, int cols);
	void ClearBuffer();
	void Reset();

	//cv::Mat GetImage();
	void GetDepth(float* depth);
	void GetVMap(glm::ivec3* vindices, glm::vec3* vweights, int* findices);
	int* d_z;
	int* d_colors;
	int* d_findices;
	glm::vec3* d_vweights;
	glm::ivec3* d_vindices;

	float* d_depth;
	int row, col;
	float cx, cy, fx, fy;
};

template<class T>
void FreeCudaArray(T* &array, int& num) {
	if (array) {
		cudaFree(array);
		num = 0;
		array = 0;
	}
}

template<class T>
void FreeCudaImage(T* &array, int& row, int& col) {
	if (array) {
		cudaFree(array);
		row = 0;
		col = 0;
		array = 0;
	}
}

#endif