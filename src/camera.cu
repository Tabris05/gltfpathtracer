#include "camera.cuh"
#include "ray.cuh"
#include <glm/gtc/matrix_transform.hpp>
#include <stb/stb_image_write.h>
#include <vector>
#include <device_launch_parameters.h>
#include <iostream>
#include <chrono>

__device__ glm::u8vec4 postProcess(glm::vec3 color) {
	// tone map color [0.0f, inf] -> [0.0f, 1.0f]
	const glm::mat3 matrix{
		0.842479062253094, 0.0423282422610123, 0.0423756549057051,
		0.0784335999999992, 0.878468636469772, 0.0784336,
		0.0792237451477643, 0.0791661274605434, 0.879142973793104
	};
	const glm::mat3 inverse{
		1.19687900512017, -0.0528968517574562, -0.0529716355144438,
		-0.0980208811401368, 1.15190312990417, -0.0980434501171241,
		-0.0990297440797205, -0.0989611768448433, 1.15107367264116
	};
	const glm::vec3 minEv{ -12.47393 };
	const glm::vec3 maxEv{ 4.026069 };

	color = matrix * color;
	color = clamp(glm::vec3{ log2(color.r), log2(color.g), log2(color.b) }, minEv, maxEv);
	color = (color - minEv) / (maxEv - minEv);

	glm::vec3 color2 = color * color;
	glm::vec3 color4 = color2 * color2;

	color = 15.5f * color4 * color2
		- 40.14f * color4 * color
		+ 31.96f * color4
		- 6.868f * color2 * color
		+ 0.4298f * color2
		+ 0.1191f * color
		- 0.00232f;

	color = inverse * color;
	color = glm::clamp(color, glm::vec3{ 0.0f }, glm::vec3{ 1.0f });

	// map color [0.0f, 1.0f] -> [0ub, 255ub]
	return glm::u8vec4(glm::min(color * 256.0f, glm::vec3(255.0f)), 255);
}

__global__ void rayKernel(
	glm::u8vec4* deviceImage,
	const Scene* scene,
	glm::mat4 worldMatrix,
	glm::vec3 position,
	glm::u16vec2 resolution,
	u32 sampleCount,
	curandState* states
) {
	i32 x = threadIdx.x + blockIdx.x * blockDim.x;
	i32 y = threadIdx.y + blockIdx.y * blockDim.y;

	if(x < resolution.x && y < resolution.y) {
		const glm::u16vec2 pixel{ x , y };
		i32 threadId = y * resolution.x + x;

		curandState* state = states + threadId;
		curand_init(0, threadId, 0, state);

		glm::vec3 color{ 0.0f };
		for(u32 i = 0; i < sampleCount; i++) {
			// convert from screen space to ndc space and then to world space
			u32 stratWidth = sqrtf(static_cast<f32>(sampleCount));
			glm::vec2 stratSample = glm::vec2{ i % stratWidth, i / stratWidth } / static_cast<f32>(stratWidth);
			glm::vec4 ndc{ (glm::vec2(pixel) + stratSample) / glm::vec2(resolution) * 2.0f - 1.0f, 0.0f, 1.0f };
			ndc.y *= -1.0f;
			glm::vec4 worldPos = worldMatrix * ndc;
			worldPos /= worldPos.w;

			Ray ray(position, glm::normalize(glm::vec3(worldPos) - position));
			glm::vec3 result = glm::min(scene->trace(ray, state), glm::vec3(100.0f)) / static_cast<f32>(sampleCount);
			
			// guard against NaN pollution
			if(isnan(result.r)) {
				result.r = 0.0f;
			}
			if(isnan(result.g)) {
				result.g = 0.0f;
			}
			if(isnan(result.b)) {
				result.b = 0.0f;
			}
			color += result;
		}
		
		deviceImage[threadId] = postProcess(color);
	}
}

Camera::Camera(glm::vec3 position, glm::vec3 lookAt, glm::u16vec2 resolution, f32 fovY, u32 sampleCount) :
	m_sampleCount{ sampleCount },
	m_position{ position },
	m_resolution { resolution },
	m_worldMatrix{ glm::inverse(glm::perspective(glm::radians(fovY / 2.0f), static_cast<f32>(resolution.x) / resolution.y, 0.1f, 100.0f) * glm::lookAt(position, lookAt, glm::vec3{ 0.0f, 1.0f, 0.0f })) } {}

void Camera::render(const Scene& scene) const {
	glm::u8vec4* deviceImage;
	cudaMalloc(&deviceImage, m_resolution.x * m_resolution.y * sizeof(glm::u8vec4));
	std::vector<glm::u8vec4> hostImage(m_resolution.x * m_resolution.y);

	const uint3 localSize = make_uint3(8, 8, 1);
	const uint3 globalSize = make_uint3(
		(m_resolution.x + localSize.x - 1) / localSize.x,
		(m_resolution.y + localSize.y - 1) / localSize.y,
		1
	);

	u32 numThreads = static_cast<u32>(m_resolution.x) * m_resolution.y;
	curandState* states;
	cudaMalloc(&states, numThreads * sizeof(curandState));

	Scene* devScene;
	cudaMalloc(&devScene, sizeof(Scene));
	cudaMemcpy(devScene, &scene, sizeof(Scene), cudaMemcpyHostToDevice);

	auto t1 = std::chrono::steady_clock::now();
	rayKernel<<<globalSize, localSize>>>(deviceImage, devScene, m_worldMatrix, m_position, m_resolution, m_sampleCount, states);
	auto err = cudaDeviceSynchronize();
	auto t2 = std::chrono::steady_clock::now();

	std::cout << "Finished in: " << std::chrono::duration<double>(t2 - t1).count() << "s\n";

 	cudaMemcpy(hostImage.data(), deviceImage, hostImage.size() * sizeof(glm::u8vec4), cudaMemcpyDeviceToHost);

	cudaFree(deviceImage);
	cudaFree(states);
	cudaFree(devScene);

	stbi_write_png("output.png", m_resolution.x, m_resolution.y, 4, hostImage.data(), m_resolution.x * 4);

}