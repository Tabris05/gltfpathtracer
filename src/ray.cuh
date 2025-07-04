#ifndef RAY_HPP
#define RAY_HPP

#include <glm/glm.hpp>
#include <optional>
#include <cuda_runtime.h>
#include "hitrecord.cuh"
#include "aabb.cuh"

struct Ray {
	__device__ Ray() = default;
	__device__ Ray(glm::vec3 origin, glm::vec3 direction);
	
	__device__ HitRecord triangleIntersection(glm::vec3 v0, glm::vec3 v1, glm::vec3 v2, f32 max) const;
	__device__ std::optional<f32> boxIntersection(AABB box, f32 max) const;
	
	glm::vec3 origin;
	glm::vec3 direction;
};

#endif
