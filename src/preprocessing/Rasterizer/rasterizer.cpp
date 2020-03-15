#include <glm/glm.hpp>
#include <iostream>
glm::mat3 rotation;
glm::vec3 translation;
float fx, fy, cx, cy;
int width, height;

#define MIN(a,b) (((a)<(b))?(a):(b))
#define MAX(a,b) (((a)>(b))?(a):(b))

static double calculateSignedArea2(const glm::dvec3& a, const glm::dvec3& b, const glm::dvec3& c) {
	return ((c.x - a.x) * (b.y - a.y) - (b.x - a.x) * (c.y - a.y));
}

static glm::dvec3 calculateBarycentricCoordinate(const glm::dvec3& a, const glm::dvec3& b, const glm::dvec3& c, const glm::dvec3& p) {
	double beta_tri = calculateSignedArea2(a, p, c);
	double gamma_tri = calculateSignedArea2(a, b, p);
	double tri_inv = 1.0f / calculateSignedArea2(a, b, c);
	double beta = beta_tri * tri_inv;
	double gamma = gamma_tri * tri_inv;
	double alpha = 1.0 - beta - gamma;
	return glm::vec3(alpha, beta, gamma);
}

static bool isBarycentricCoordInBounds(const glm::dvec3 barycentricCoord) {
    return barycentricCoord.x >= 0.0 && barycentricCoord.x <= 1.0 &&
           barycentricCoord.y >= 0.0 && barycentricCoord.y <= 1.0 &&
           barycentricCoord.z >= 0.0 && barycentricCoord.z <= 1.0;
}

static double getZAtCoordinate(const glm::dvec3 barycentricCoord, const glm::dvec3& a, const glm::dvec3& b, const glm::dvec3& c) {
	return (barycentricCoord.x * a.z
		+ barycentricCoord.y * b.z
		+ barycentricCoord.z * c.z);
}


extern "C" {
void Rasterize3D(glm::vec3& pt1, glm::vec3& pt2, glm::vec3& pt3, float* zbuffer, glm::vec3* color, int* findices, int idx) {
	glm::dvec3 p1 = glm::dvec3(rotation * pt1 + translation);
	glm::dvec3 p2 = glm::dvec3(rotation * pt2 + translation);
	glm::dvec3 p3 = glm::dvec3(rotation * pt3 + translation);

	//printf("<%f %f %f> <%f %f %f> <%f %f %f>\n", p1[0], p1[1], p1[2], p2[0], p2[1], p2[2], p3[0], p3[1], p3[2]);
	if (p1.z < 0.02 || p2.z < 0.02 || p3.z < 0.02)
		return;

	p1.z = 1.0f / p1.z;
	p2.z = 1.0f / p2.z;
	p3.z = 1.0f / p3.z;

	p1.x = p1.x * p1.z;
	p1.y = p1.y * p1.z;
	p2.x = p2.x * p2.z;
	p2.y = p2.y * p2.z;
	p3.x = p3.x * p3.z;
	p3.y = p3.y * p3.z;

	int minX = (std::min(p1.x, std::min(p2.x, p3.x)) * fx + cx);
	int minY = (std::min(p1.y, std::min(p2.y, p3.y)) * fy + cy);
	int maxX = (std::max(p1.x, std::max(p2.x, p3.x)) * fx + cx) + 0.999999f;
	int maxY = (std::max(p1.y, std::max(p2.y, p3.y)) * fy + cy) + 0.999999f;

	//printf("%d %d %d %d\n", minX, minY, maxX, maxY);
	minX = std::max(0, minX);
	minY = std::max(0, minY);
	maxX = std::min(width, maxX);
	maxY = std::min(height, maxY);

	for (int py = minY; py <= maxY; ++py) {
		for (int px = minX; px <= maxX; ++px) {
			//printf("Step1\n");
			if (px < 0 || px >= width || py < 0 || py >= height)
				continue;

			float x = (px - cx) / fx;
			float y = (py - cy) / fy;

			glm::dvec3 baryCentricCoordinate = calculateBarycentricCoordinate(p1, p2, p3, glm::dvec3(x, y, 0));
			//printf("<%f %f>, <%f %f>, <%f %f>, <%f %f>\n", p1[0],p1[1],p2[0],p2[1],p3[0],p3[1], x, y);
			if (isBarycentricCoordInBounds(baryCentricCoordinate)) {
				//printf("Step2\n");
				int pixel = py * width + px;

				float z = getZAtCoordinate(baryCentricCoordinate, p1, p2, p3);
				if (z > zbuffer[pixel]) {
					//printf("Step3\n");
					zbuffer[pixel] = z;
					glm::vec3 rgb = baryCentricCoordinate;
					color[pixel] = rgb;
					findices[pixel] = idx;
				}
			}
		}
	}	
}

void RasterizeTex(glm::vec2& pt1, glm::vec2& pt2, glm::vec2& pt3, float* zbuffer, glm::vec3* color, int* findices, int idx) {
	glm::dvec3 p1 = glm::dvec3(pt1.x, pt1.y, 1);
	glm::dvec3 p2 = glm::dvec3(pt2.x, pt2.y, 1);
	glm::dvec3 p3 = glm::dvec3(pt3.x, pt3.y, 1);

	p1.z = 1.0f / p1.z;
	p2.z = 1.0f / p2.z;
	p3.z = 1.0f / p3.z;

	p1.x = p1.x * p1.z;
	p1.y = p1.y * p1.z;
	p2.x = p2.x * p2.z;
	p2.y = p2.y * p2.z;
	p3.x = p3.x * p3.z;
	p3.y = p3.y * p3.z;

	float p1x = p1.x * fx + cx;
	float p2x = p2.x * fx + cx;
	float p3x = p3.x * fx + cx;
	float p1y = p1.y * fy + cy;
	float p2y = p2.y * fy + cy;
	float p3y = p3.y * fy + cy;

	int minX = (std::min(p1x, std::min(p2x, p3x)));
	int minY = (std::min(p1y, std::min(p2y, p3y)));
	int maxX = (std::max(p1x, std::max(p2x, p3x))) + 0.999999f;
	int maxY = (std::max(p1y, std::max(p2y, p3y))) + 0.999999f;

	minX = std::max(0, minX);
	minY = std::max(0, minY);
	maxX = std::min(width, maxX);
	maxY = std::min(height, maxY);

	for (int py = minY; py <= maxY; ++py) {
		for (int px = minX; px <= maxX; ++px) {
			if (px < 0 || px >= width || py < 0 || py >= height)
				continue;

			float x = (px - cx) / fx;
			float y = (py - cy) / fy;

			glm::dvec3 baryCentricCoordinate = calculateBarycentricCoordinate(p1, p2, p3, glm::dvec3(x, y, 0));
			int pixel = py * width + px;
			if (isBarycentricCoordInBounds(baryCentricCoordinate)) {
				float z = getZAtCoordinate(baryCentricCoordinate, p1, p2, p3);
				if (z > zbuffer[pixel]) {
					zbuffer[pixel] = z;
					glm::vec3 rgb = baryCentricCoordinate;
					color[pixel] = rgb;
					findices[pixel] = idx;
				}
			} else if (zbuffer[pixel] <= 0) {
				float pxf = px;
				float pyf = py;
				float z = -getZAtCoordinate(baryCentricCoordinate, p1, p2, p3);
				if (z < zbuffer[pixel]) {
					zbuffer[pixel] = z;
					glm::vec3 rgb = baryCentricCoordinate;
					color[pixel] = rgb;
					findices[pixel] = idx;
				}
			}
		}
	}	
}

void RasterizeTexture(glm::vec2* vt, glm::ivec3* ft, glm::vec3* color, float* zbuffer, int* findices, int num_face, int _width, int _height) {
	width = _width;
	height = _height;
	cx = 0;
	cy = height-1;
	fx = (width-1);
	fy = (-height+1);
	for (int i = 0; i < num_face; ++i) {
		RasterizeTex(vt[ft[i][0]], vt[ft[i][1]], vt[ft[i][2]], zbuffer, color, findices, i);
	}
}

void RasterizeImage(glm::vec3* V, glm::ivec3* F, glm::mat4* world2cam, float* intrinsic,
		glm::vec3* vweights, float* zbuffer, int* findices, int num_f, int _width, int _height) {
	glm::mat4 transform = glm::transpose(world2cam[0]);
	
	rotation = glm::mat3(transform);
	translation = glm::vec3(transform[3]);

	width = _width;
	height = _height;
	fx = intrinsic[0];
	fy = intrinsic[5];
	cx = intrinsic[2];
	cy = intrinsic[6];

	for (int i = 0; i < num_f; ++i) {
		Rasterize3D(V[F[i][0]], V[F[i][1]], V[F[i][2]], zbuffer, vweights, findices, i);
	}
}


void GenerateTextiles(glm::vec3* V, glm::ivec3* F, glm::vec3* VN, glm::ivec3* FN, glm::vec3* points, glm::vec3* normals, glm::ivec2* coords, int* finds, glm::vec3* vweights, int width, int height) {
	int top = 0;
	for (int i = 0; i < height; ++i) {
		for (int j = 0; j < width; ++j) {
			int find = finds[i * width + j];
			if (find == -1)
				continue;
			glm::vec3& w = vweights[i * width + j];
			points[top] = (V[F[find][0]] * w[0] + V[F[find][1]] * w[1] + V[F[find][2]] * w[2]);
			normals[top] = glm::normalize((VN[FN[find][0]] * w[0] + VN[FN[find][1]] * w[1] + VN[FN[find][2]] * w[2]));
			coords[top] = glm::ivec2(j, i);
			top += 1;
		}
	}
}

void RenderUV(glm::vec3* uv, glm::vec3* vweights, int* findices, glm::vec2* VT, glm::ivec3* FT, int width, int height)
{
	for (int i = 0; i < height; ++i) {
		for (int j = 0; j < width; ++j) {			
			if (findices[i * width + j] == -1) {
				uv[i * width + j] = glm::vec3(0);
			} else {
				int find = findices[i * width + j];
				glm::vec3 vw = vweights[i * width + j];
				uv[i * width + j] = glm::vec3(vw[0] * VT[FT[find][0]] + vw[1] * VT[FT[find][1]] + vw[2] * VT[FT[find][2]], 0);
			}
		}
	}
}
}