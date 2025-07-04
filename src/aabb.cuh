#ifndef AABB_HPP
#define AABB_HPP

#include <limits>
#include <glm/glm.hpp>
#include "typedefs.cuh"

struct AABB {
	glm::vec3 min{  std::numeric_limits<f32>::infinity() };
	glm::vec3 max{ -std::numeric_limits<f32>::infinity() };
};

#endif
