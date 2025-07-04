#ifndef HITRECORD_HPP
#define HITRECORD_HPP

#include <glm/glm.hpp>
#include <cuda_runtime.h>
#include "typedefs.cuh"

struct HitRecord {
	glm::uvec3 triangle;
	glm::vec3 barycentric;
	f32 distance;
	u32 materialIndex;
	bool valid = false;

	__device__ operator bool() const {
		return valid;
	}

	__device__ HitRecord& setTriangle(glm::uvec3 tri) {
		triangle = tri;
		return *this;
	}

	__device__ HitRecord& setMaterial(u32 mat) {
		materialIndex = mat;
		return *this;
	}
};

#endif
