#include <glm/glm.hpp>
#include <iostream>

extern "C"{
void ProjectPaint(glm::vec3* points, glm::vec3* normals, glm::vec4* point_colors, unsigned char* color, float* depth, glm::mat4* transform, float* intrinsic,
	int num_points, int depth_width, int depth_height, int color_width, int color_height) {

	float d2c_x = (float)color_width / depth_width;
	float d2c_y = (float)color_height / depth_height;
	float d_fx = intrinsic[0];
	float d_fy = intrinsic[5];
	float d_cx = intrinsic[2];
	float d_cy = intrinsic[6];
	float c_fx = d_fx * d2c_x;
	float c_cx = d_cx * d2c_x;
	float c_fy = d_fy * d2c_y;
	float c_cy = d_cy * d2c_y;

	glm::mat4 world2cam = glm::transpose(transform[0]);
	for (int i = 0; i < num_points; ++i) {
		glm::vec4 wp(points[i], 1);
		wp = world2cam * wp;

		glm::vec4 wn(normals[i], 0);
		wn = world2cam * wn;

		if (glm::dot(wp, wn) > 0)
			continue;

		float d_px = wp[0] / wp[2] * d_fx + d_cx;
		float d_py = wp[1] / wp[2] * d_fy + d_cy;

		float gt_z = wp[2];
		if (d_px >= 0 && d_py >= 0 && d_px < depth_width && d_py < depth_height) {
			float d = depth[((int)d_py) * depth_width + (int)d_px];
			if (std::abs(d * 1e-3f - gt_z) < 0.05) {
				float c_px = d2c_x * d_px;
				float c_py = d2c_y * d_py;
				int lx = c_px;
				int ly = c_py;
				if (lx >= 0 && ly >= 0 && lx + 1 < color_width && ly + 1 < color_height) {
					float wx = c_px - lx;
					float wy = c_py - ly;

					//printf("%f %f %f %f\n", wp[0], wp[1], wp[2], wp[0] / wp[2] * d_fx + d_cx);
					glm::vec4 c(0);
					unsigned char* c1 = color + (ly * color_width + lx) * 3;
					unsigned char* c2 = c1 + 3;
					unsigned char* c3 = c1 + color_width * 3;
					unsigned char* c4 = c3 + 3;

					point_colors[i][0] += (c1[0] * (1 - wx) + c2[0] * wx) * (1 - wy) + (c3[0] * (1 - wx) + c4[0] * wx) * wy;
					point_colors[i][1] += (c1[1] * (1 - wx) + c2[1] * wx) * (1 - wy) + (c3[1] * (1 - wx) + c4[1] * wx) * wy;
					point_colors[i][2] += (c1[2] * (1 - wx) + c2[2] * wx) * (1 - wy) + (c3[2] * (1 - wx) + c4[2] * wx) * wy;
					point_colors[i][3] += 1;
				}
			}
		}

	}
}

void PaintToTexturemap(unsigned char* texture, glm::vec4* point_color, glm::ivec2* coords, int num_points, int width, int height) {
	for (int i = 0; i < num_points; ++i) {
		int pixel = (coords[i][1] * width + coords[i][0]) * 3;
		texture[pixel] = (point_color[i][0] / point_color[i][3]) + 0.5;
		texture[pixel + 1] = (point_color[i][1] / point_color[i][3]) + 0.5;
		texture[pixel + 2] = (point_color[i][2] / point_color[i][3]) + 0.5;
	}
}

void PaintToView(glm::vec3* textureToImage, glm::vec3* points_cam, unsigned char* mask,
	float* depth, glm::ivec2* coords, int num_points, int height, int width, int tex_width) {
	for (int j = 0; j < num_points; ++j) {
		glm::vec3& p = points_cam[j];
		int px = p[0], py = p[1];
		if (p[0] < 0 || p[1] < 0 || p[0] >= 639 || p[1] >= 479)
			continue;
		if (mask[py * width + px] == 0 || mask[py * width + px + 1] == 0 || mask[(py + 1) * width + px] == 0 || mask[(py + 1) * width + px + 2] == 0)
			continue;
		float max_depth = depth[py * width + px];
		max_depth = std::max(max_depth, depth[(py + 1) * width + px]);
		max_depth = std::max(max_depth, depth[py * width + px + 1]);
		max_depth = std::max(max_depth, depth[(py + 1) * width + px + 1]);
		if (p[2] <= max_depth)
			textureToImage[coords[j][1] * tex_width + coords[j][0]] = glm::vec3(p[0] / 639.0, 1.0 - (p[1] / 479.0), 0);
	}
}

void PaintToViewNorm(glm::vec3* textureToImage, glm::vec3* points_cam, glm::vec3* normals_cam, unsigned char* mask,
	float* depth, glm::ivec2* coords, int num_points, int height, int width, int tex_width) {
	for (int j = 0; j < num_points; ++j) {
		glm::vec3& p = points_cam[j];
		glm::vec3& n = normals_cam[j];
		if (n.z > 0)
			continue;
		int px = p[0], py = p[1];
		if (p[0] < 0 || p[1] < 0 || p[0] >= 639 || p[1] >= 479)
			continue;
		if (mask[py * width + px] == 0 || mask[py * width + px + 1] == 0 || mask[(py + 1) * width + px] == 0 || mask[(py + 1) * width + px + 2] == 0)
			continue;
		float max_depth = depth[py * width + px];
		max_depth = std::max(max_depth, depth[(py + 1) * width + px]);
		max_depth = std::max(max_depth, depth[py * width + px + 1]);
		max_depth = std::max(max_depth, depth[(py + 1) * width + px + 1]);
		if (p[2] <= max_depth)
			textureToImage[coords[j][1] * tex_width + coords[j][0]] = glm::vec3(p[0] / 639.0, 1.0 - (p[1] / 479.0), 0);
	}
}

void PaintSharpnessToView(float* sharp_candidates, int* frame_candidates, int* coordx, int* coordy, float* sharpness, glm::vec3* points_cam, unsigned char* mask,
	float* depth, glm::ivec2* coords, int num_points, int height, int width, int tex_width, int num_candidates, int frame_id) {
	for (int j = 0; j < num_points; ++j) {
		glm::vec3& p = points_cam[j];
		int px = p[0], py = p[1];
		if (p[0] < 0 || p[1] < 0 || p[0] >= 639 || p[1] >= 479)
			continue;
		if (mask[py * width + px] == 0 || mask[py * width + px + 1] == 0 || mask[(py + 1) * width + px] == 0 || mask[(py + 1) * width + px + 2] == 0)
			continue;
		float max_depth = depth[py * width + px];
		max_depth = std::max(max_depth, depth[(py + 1) * width + px]);
		max_depth = std::max(max_depth, depth[py * width + px + 1]);
		max_depth = std::max(max_depth, depth[(py + 1) * width + px + 1]);
		if (p[2] <= max_depth) {
			float sharp_value = sharpness[py * width + px];
			int f = coords[j][1] * tex_width + coords[j][0];

			int* candidate_list = frame_candidates + f * num_candidates;
			int* cx_list = coordx + f * num_candidates;
			int* cy_list = coordy + f * num_candidates;
			float* sharp_list = sharp_candidates + f * num_candidates;

			// check existence
			bool found = false;
			for (int k = 0; k < num_candidates; ++k) {
				if (candidate_list[k] == -1)
					break;
				if (candidate_list[k] == frame_id) {
					found = true;
					if (sharp_value > sharp_list[k]) {
						sharp_list[k] = sharp_value;
						int current_k = k;
						while (current_k > 0 && sharp_list[current_k] > sharp_list[current_k - 1]) {
							std::swap(sharp_list[current_k], sharp_list[current_k - 1]);
							std::swap(candidate_list[current_k], candidate_list[current_k - 1]);
							std::swap(cx_list[current_k], cx_list[current_k - 1]);
							std::swap(cy_list[current_k], cy_list[current_k - 1]);
							current_k -= 1;
						}
					}
					break;
				}
			}
			if (found)
				continue;
			for (int k = 0; k < num_candidates; ++k) {
				if (candidate_list[k] == -1 || sharp_list[k] < sharp_value) {
					for (int k1 = num_candidates - 1; k1 > k; k1--) {
						sharp_list[k1] = sharp_list[k1 - 1];
						candidate_list[k1] = candidate_list[k1 - 1];
						cx_list[k1] = cx_list[k1 - 1];
						cy_list[k1] = cy_list[k1 - 1];
					}
					sharp_list[k] = sharp_value;
					candidate_list[k] = frame_id;
					cx_list[k] = px;
					cy_list[k] = py;
					break;
				}
			}
		}
	}
}

void PaintSharpnessFromView(float* sharp_candidates, int* frame_candidates, int* coords,
	float* sharp, float* uv, unsigned char* mask, int image_width, int image_height, int tex_width, int tex_height,
	int num_points, int num_candidates, int frame_id) {
	for (int i = 0; i < image_height; ++i) {
		for (int j = 0; j < image_width; ++j) {
			int image_offset = i * image_width + j;
			if (mask[image_offset] == 0)
				continue;
			float x = uv[image_offset * 2];
			float y = uv[image_offset * 2 + 1];
			if (x == 0 && y == 0) {
				continue;
			}
			float sharp_value = sharp[image_offset];
			x = x * (tex_width - 1);
			y = (1 - y) * (tex_height - 1);
			int px = x, py = y;
			for (int dy = 0; dy < 2; ++dy) {
				for (int dx = 0; dx < 2; ++dx) {
					if (px + dx < tex_width && py + dy < tex_height && px + dx >= 0 && py + dy >= 0) {
						int f = coords[(py + dy) * tex_width + (px + dx)];
						if (f == -1)
							continue;
						int* candidate_list = frame_candidates + f * num_candidates;
						float* sharp_list = sharp_candidates + f * num_candidates;

						// check existence
						bool found = false;
						for (int k = 0; k < num_candidates; ++k) {
							if (candidate_list[k] == -1)
								break;
							if (candidate_list[k] == frame_id) {
								found = true;
								if (sharp_value > sharp_list[k]) {
									sharp_list[k] = sharp_value;
									int current_k = k;
									while (current_k > 0 && sharp_list[current_k] > sharp_list[current_k - 1]) {
										std::swap(sharp_list[current_k], sharp_list[current_k - 1]);
										std::swap(candidate_list[current_k], candidate_list[current_k - 1]);
										current_k -= 1;
									}
								}
								break;
							}
						}
						if (found)
							continue;
						for (int k = 0; k < num_candidates; ++k) {
							if (candidate_list[k] == -1 || sharp_list[k] < sharp_value) {
								for (int k1 = num_candidates - 1; k1 > k; k1--) {
									sharp_list[k1] = sharp_list[k1 - 1];
									candidate_list[k1] = candidate_list[k1 - 1];
								}
								sharp_list[k] = sharp_value;
								candidate_list[k] = frame_id;
								break;
							}
						}

					}					
				}
			}
		}
	}
}

void ValidateSharpnessFromView(float* sharp_candidates, int* frame_candidates, int* coords,
	float* sharp, float* uv, unsigned char* mask, int image_width, int image_height, int tex_width, int tex_height,
	int num_points, int num_candidates, int frame_id) {
	for (int i = 0; i < image_height; ++i) {
		for (int j = 0; j < image_width; ++j) {
			int image_offset = i * image_width + j;
			if (mask[image_offset] == 0)
				continue;
			float x = uv[image_offset * 2];
			float y = uv[image_offset * 2 + 1];
			if (x == 0 && y == 0) {
				continue;
			}
			float sharp_value = sharp[image_offset];
			x = x * (tex_width - 1);
			y = (1 - y) * (tex_height - 1);
			int px = x, py = y;
			bool found = false;
			for (int dy = 0; dy < 2; ++dy) {
				for (int dx = 0; dx < 2; ++dx) {
					if (px + dx < tex_width && py + dy < tex_height && px + dx >= 0 && py + dy >= 0) {
						int f = coords[(py + dy) * tex_width + (px + dx)];
						if (f == -1)
							continue;
						int* candidate_list = frame_candidates + f * num_candidates;
						float* sharp_list = sharp_candidates + f * num_candidates;
						if (sharp_list[num_candidates - 1] <= sharp_value)
							found = true;
					}					
				}
			}
			if (!found) 
				mask[image_offset] = 0;
		}
	}
}

void ComputeMask(int* coordx, int* coordy, int* frames, unsigned char* masks,
	int tex_width, int tex_height, int num_candidates, int image_width, int image_height) {
	for (int i = 0; i < tex_height; ++i) {
		for (int j = 0; j < tex_width; ++j) {
			for (int k = 0; k < num_candidates; ++k) {
				int offset = (i * tex_width + j) * num_candidates + k;
				int fid = frames[offset];
				int cx = coordx[offset];
				int cy = coordy[offset];
				masks[(fid * image_height + cy) * image_width + cx] = 255;
			}
		}
	}
}
};