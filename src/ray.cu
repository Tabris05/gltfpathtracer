#include "ray.cuh"

__device__ Ray::Ray(glm::vec3 origin, glm::vec3 direction) : origin{ origin }, direction{ glm::normalize(direction) } {}

__device__ HitRecord Ray::triangleIntersection(glm::vec3 v0, glm::vec3 v1, glm::vec3 v2, f32 max) const {
	static constexpr f32 epsilon = std::numeric_limits<f32>::epsilon();

	// Möller-Trumbore algorithm for ray/triangle intersection
	glm::vec3 dv1 = v1 - v0;
	glm::vec3 dv2 = v2 - v0;
	glm::vec3 pvec = glm::cross(direction, dv2);
	f32 det = glm::dot(dv1, pvec);

	// if the ray is parallel to the triangle then no intersection
	if(abs(det) < epsilon) {
		return {};
	}
	
	f32 invDet = 1.0f / det;
	
	glm::vec3 tvec = origin - v0;
	f32 u = glm::dot(tvec, pvec) * invDet;

	// barycentric coordinates are [0, 1] inside the triangle
	// if the barycentric coordinate is outside this range then no intersection
	if((u < 0.0f && abs(u) > epsilon) || (u > 1.0f && std::abs(1.0f - u) > epsilon)) {
		return {};
	}
	
	glm::vec3 qvec = glm::cross(tvec, dv1);
	f32 v = glm::dot(direction, qvec) * invDet;

	// barycentric coordinates are [0, 1] inside the triangle
	// if the barycentric coordinate is outside this range then no intersection
	if((v < 0.0f && abs(v) > epsilon) || (u + v > 1.0f && abs(u + v - 1.0f) > epsilon)) {
		return {};
	}
	
	// if we have a closer hit already then ignore this one
	f32 t = glm::dot(dv2, qvec) * invDet;
	if(t < epsilon || t > max) {
		return {};
	}

	return HitRecord{ glm::vec3(0.0f), glm::vec3(1.0f - u - v, u, v), t, 0, true };
}

__device__ std::optional<f32> Ray::boxIntersection(AABB box, f32 maximum) const {
	f32 minimum = std::numeric_limits<f32>::epsilon();

	glm::vec3 invD = 1.0f / direction;
	glm::vec3 t0 = (box.min - origin) * invD;
	glm::vec3 t1 = (box.max - origin) * invD;
	
	glm::vec3 mins = glm::min(t0, t1);
	glm::vec3 maxes = glm::max(t0, t1);

	minimum = fmaxf(minimum, fmaxf(mins.x, fmaxf(mins.y, mins.z)));
	maximum = fminf(maximum, fminf(maxes.x, fminf(maxes.y, maxes.z)));

	if(minimum <= maximum) {
		return minimum;
	}
	
	return {};
}
