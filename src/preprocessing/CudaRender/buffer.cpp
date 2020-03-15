#include "buffer.hpp"
#include "render.hpp"
#include <iostream>

VertexBuffer::VertexBuffer()
: d_positions(0), d_colors(0), d_normals(0), d_texcoords(0), d_indices(0),
  num_positions(0), num_colors(0), num_normals(0), num_texcoords(0), num_indices(0)
{
	translation = glm::vec3(0, 0, 0);
	rotation = glm::mat3(1.0f);
}

VertexBuffer::~VertexBuffer() {
	Reset();
}

void VertexBuffer::Reset() {
	FreeCudaArray(d_positions, num_positions);
	FreeCudaArray(d_colors, num_colors);
	FreeCudaArray(d_normals, num_normals);
	FreeCudaArray(d_texcoords, num_texcoords);
}

void VertexBuffer::SetPositions(const std::vector<glm::vec3>& positions) {
	if (num_positions != positions.size()) {
		FreeCudaArray(d_positions, num_positions);
		num_positions = positions.size();
		cudaMalloc((void**)&d_positions, sizeof(glm::vec3) * positions.size());
	}
	cudaMemcpy(d_positions, positions.data(), sizeof(glm::vec3) * positions.size(), cudaMemcpyHostToDevice);
}

void VertexBuffer::SetColors(const std::vector<glm::vec3>& colors) {
	if (num_colors != colors.size()) {
		FreeCudaArray(d_colors, num_colors);
		num_colors = colors.size();
		cudaMalloc((void**)&d_colors, sizeof(glm::vec3) * colors.size());
	}
	cudaMemcpy(d_colors, colors.data(), sizeof(glm::vec3) * colors.size(), cudaMemcpyHostToDevice);
}

void VertexBuffer::SetNormals(const std::vector<glm::vec3>& normals) {
	if (num_normals != normals.size()) {
		FreeCudaArray(d_normals, num_normals);
		num_normals = normals.size();
		cudaMalloc((void**)&d_normals, sizeof(glm::vec3) * normals.size());
	}
	cudaMemcpy(d_normals, normals.data(), sizeof(glm::vec3) * normals.size(), cudaMemcpyHostToDevice);
}

void VertexBuffer::SetTexcoords(const std::vector<glm::vec2>& texcoords) {
	if (num_texcoords != texcoords.size()) {
		FreeCudaArray(d_texcoords, num_texcoords);
		num_texcoords = texcoords.size();
		cudaMalloc((void**)&d_texcoords, sizeof(glm::vec2) * texcoords.size());
	}
	cudaMemcpy(d_texcoords, texcoords.data(), sizeof(glm::vec2) * texcoords.size(), cudaMemcpyHostToDevice);
}

void VertexBuffer::SetIndices(const std::vector<glm::ivec3>& indices) {
	if (num_indices != indices.size()) {
		FreeCudaArray(d_indices, num_indices);
		num_indices = indices.size();
		cudaMalloc((void**)&d_indices, sizeof(glm::ivec3) * indices.size());
	}
	cudaMemcpy(d_indices, indices.data(), sizeof(glm::ivec3) * indices.size(), cudaMemcpyHostToDevice);
}


FrameBuffer::FrameBuffer()
: d_z(0), d_colors(0), d_depth(0), d_vindices(0), d_vweights(0), row(0), col(0)
{}

FrameBuffer::FrameBuffer(int rows, int cols, float _cx, float _cy, float _fx, float _fy)
: d_z(0), d_colors(0), d_depth(0), d_vindices(0), d_vweights(0), row(0), col(0), cx(_cx), cy(_cy), fx(_fx), fy(_fy)
{
	Initialize(rows, cols);
	ClearBuffer();
}

FrameBuffer::~FrameBuffer() {
	Reset();
}

void FrameBuffer::Create(int _rows, int _cols, float _cx, float _cy, float _fx, float _fy) {
	cx = _cx;
	cy = _cy;
	fx = _fx;
	fy = _fy;
	Initialize(_rows, _cols);
	ClearBuffer();
}

void FrameBuffer::Reset() {
	FreeCudaImage(d_depth, row, col);
	FreeCudaImage(d_vindices, row, col);
	FreeCudaImage(d_vweights, row, col);
	FreeCudaImage(d_z, row, col);
	FreeCudaImage(d_colors, row, col);
	FreeCudaImage(d_findices, row, col);
}

void FrameBuffer::Initialize(int rows, int cols) {
	if (row != rows || col != cols) {
		Reset();
		row = rows;
		col = cols;
		cudaMalloc(&d_z, sizeof(int) * row * col);
		cudaMalloc(&d_colors, sizeof(int) * row * col);
		cudaMalloc(&d_findices, sizeof(int) * row * col);
		cudaMalloc(&d_depth, sizeof(float) * row * col);
		cudaMalloc(&d_vindices, sizeof(glm::ivec3) * row * col);
		cudaMalloc(&d_vweights, sizeof(glm::vec3) * row * col);
	}
}

void FrameBuffer::ClearBuffer() {
	cudaMemset(d_z, 0, sizeof(int) * row * col);
	cudaMemset(d_depth, 0, sizeof(float) * row * col);
	cudaMemset(d_findices, 0, sizeof(int) * row * col);
	cudaMemset(d_vweights, 0, sizeof(glm::vec3) * row * col);
	cudaMemset(d_vindices, 0, sizeof(glm::ivec3) * row * col);
	cudaMemset(d_colors, 0, sizeof(int) * row * col);
}
/*
cv::Mat FrameBuffer::GetImage() {
	cv::Mat res(row, col, CV_8UC4);
	cudaMemcpy(res.data, d_colors, sizeof(int) * row * col, cudaMemcpyDeviceToHost);
	return res;
}
*/
void FrameBuffer::GetDepth(float* depth) {
	cudaMemcpy(depth, d_depth, sizeof(float) * row * col, cudaMemcpyDeviceToHost);
}

void FrameBuffer::GetVMap(glm::ivec3* vindices, glm::vec3* vweights, int* findices) {
	cudaMemcpy(vindices, d_vindices, sizeof(glm::ivec3) * row * col, cudaMemcpyDeviceToHost);
	cudaMemcpy(vweights, d_vweights, sizeof(glm::vec3) * row * col, cudaMemcpyDeviceToHost);
	cudaMemcpy(findices, d_findices, sizeof(int) * row * col, cudaMemcpyDeviceToHost);
}
