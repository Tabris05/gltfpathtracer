#ifndef TEXTURE_HPP
#define TEXTURE_HPP

#include <glm/glm.hpp>
#include <fastgltf/core.hpp>
#include <cuda_runtime.h>
#include "typedefs.cuh"

class Texture {
public:
	Texture(std::filesystem::path path);
	Texture(const fastgltf::StaticVector<std::byte>& encodedData, bool decodeSRGB = false);

	Texture(const Texture&) = delete;
	void operator=(const Texture&) = delete;

	Texture(Texture&&);
	void operator=(Texture&&);

	~Texture();

	__device__ glm::vec4 sample(glm::vec2 uv) const;

private:

	glm::vec3 srgbToLinear(glm::vec3 color) const;

	glm::vec4* m_data = nullptr;
	i32 m_width;
	i32 m_height;
};

#endif
