#ifndef CAMERA_HPP
#define CAMERA_HPP

#include <glm/glm.hpp>
#include <cuda_runtime.h>
#include "typedefs.cuh"
#include "scene.cuh"

class Camera {
	public:
		Camera(glm::vec3 position, glm::vec3 lookAt, glm::u16vec2 resolution, f32 fovY, u32 sampleCount);
		
		void render(const Scene& scene) const;

	private:
		const u32 m_sampleCount;
		const glm::vec3 m_position;
		const glm::u16vec2 m_resolution;
		const glm::mat4 m_worldMatrix;
};

#endif

