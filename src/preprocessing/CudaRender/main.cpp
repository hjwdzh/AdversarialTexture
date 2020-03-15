#include <iostream>
#include <queue>
#include <cstring>
#include "buffer.hpp"
//#include "loader.hpp"
#include "render.hpp"

extern "C" {

std::vector<VertexBuffer*> vertexBuffers;
FrameBuffer frameBuffer;

void InitializeCamera(int width, int height, float fx, float fy, float cx, float cy)
{
	frameBuffer.Create(height, width, cx, cy, fx, fy);
}

int SetMesh(glm::vec3* positions, glm::ivec3* faces, int num_v, int num_f) {
	vertexBuffers.push_back(new VertexBuffer());
	std::vector<glm::vec3> p(num_v);
	std::vector<glm::ivec3> f(num_f);
	memcpy(p.data(), positions, sizeof(glm::vec3) * num_v);
	memcpy(f.data(), faces, sizeof(glm::ivec3) * num_f);

	vertexBuffers.back()->SetPositions(p);
	vertexBuffers.back()->SetIndices(f);

	return vertexBuffers.size() - 1;
}

void SetTransform(int handle, float* transform) {
	glm::mat3 rotation;
	glm::vec3 translation;
	for (int i = 0; i < 3; ++i) {
		for (int j = 0; j < 3; ++j) {
			rotation[i][j] = transform[j * 4 + i];
		}
		translation[i] = transform[i * 4 + 3];
	}
	vertexBuffers[handle]->rotation = rotation;
	vertexBuffers[handle]->translation = translation;
}

void ClearData() {
	for (auto vb : vertexBuffers)
		delete vb;
	vertexBuffers.clear();
}

void Render(int handle) {
	frameBuffer.ClearBuffer();
	Render(*vertexBuffers[handle], frameBuffer);
}

void GetDepth(float* depth) {
	FetchDepth(frameBuffer);
	frameBuffer.GetDepth(depth);
}

void GetVMap(int handle, glm::ivec3* vindices, glm::vec3* vweights, int* findices) {
	FetchVMap(*vertexBuffers[handle], frameBuffer);
	frameBuffer.GetVMap(vindices, vweights, findices);
}

void Colorize(glm::vec4* VC, glm::ivec3* vindices, glm::vec3* vweights, unsigned char* mask, glm::vec3* image, int row, int col) {
	int offset = 0;
	for (int i = 0; i < row; ++i) {
		for (int j = 0; j < col; ++j) {
			if (mask[offset]) {
				glm::vec3& c = image[offset];
				for (int k = 0; k < 3; ++k) {
					float w = vweights[offset][k];
					int index = vindices[offset][k];
					glm::vec4 color(c * w, w);
					VC[index] += color;
				}
			}
			offset += 1;
		}
	}
}
/*
void expand(glm::vec3* textureToImage, unsigned char* mask, int width, int height) {
	struct data
	{
		data(int x, int y, glm::vec3& _coord)
		: x(x), y(y), coord(_coord)
		{}
		int x;
		int y;
		glm::vec3 coord;
	};
	std::queue<data> q;
	for (int i = 0; i < height; ++i) {
		for (int j = 0; j < width; ++j) {
			if (mask[i * width + j] == 1) {
				q.push(data(j, i, textureToImage[i * width + j]));
			}
		}
	}
	while (!q.empty()) {
		data info = q.front();
		q.pop();
		int dx[] = {-1, 1, 0, 0};
		int dy[] = {0, 0, -1, 1};
		for (int i = 0; i < 4; ++i) {
			int nx = info.x + dx[i];
			int ny = info.y + dy[i];
			if (nx < 0 || ny < 0 || nx >= width || ny >= height)
				continue;
			if (mask[ny * width + nx] == 1)
				continue;
			if (mask[ny * width + nx] == 0) {
				textureToImage[ny * width + nx] = info.coord;
			}
			mask[ny * width + nx] = 1;
			q.push(data(ny, nx, info.coord));
		}
	}
}
*/
};