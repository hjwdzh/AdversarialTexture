#ifndef MESH_H_
#define MESH_H_

#include <vector>
#include <glm/glm.hpp>

class Mesh
{
public:
	std::vector<glm::vec3> positions;
	std::vector<glm::ivec3> faces;
};

#endif