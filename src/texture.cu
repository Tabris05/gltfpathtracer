#include "texture.cuh"
#include <stb/stb_image.h>
#include <cuda.h>

Texture::Texture(std::filesystem::path path) {
	f32* rawData = stbi_loadf(path.string().c_str(), &m_width, &m_height, nullptr, STBI_rgb);
	std::vector<glm::vec4> data;
	data.reserve(m_width * m_height);

	for(u32 i = 0; i < m_width * m_height * 3; i += 3) {
		// no meaningful visual difference once hdr values get too bright and clamping helps mitigate fireflies
		glm::vec3 clampedColor = glm::min(glm::vec3{ rawData[i], rawData[i + 1], rawData[i + 2] }, glm::vec3{ 100.0f });
		data.emplace_back(clampedColor, 1.0f);
	}

	cudaMalloc(&m_data, data.size() * sizeof(glm::vec4));
	cudaMemcpy(m_data, data.data(), data.size() * sizeof(glm::vec4), cudaMemcpyHostToDevice);

	stbi_image_free(rawData);
}

Texture::Texture(const fastgltf::StaticVector<std::byte>& encodedData, bool decodeSRGB) {
	stbi_uc* rawData = stbi_load_from_memory(reinterpret_cast<const stbi_uc*>(encodedData.data()), encodedData.size(), &m_width, &m_height, nullptr, STBI_rgb_alpha);
	std::vector<glm::vec4> data;
	data.reserve(m_width * m_height);

	for(u32 i = 0; i < m_width * m_height * 4; i += 4) {
		glm::vec4 normalizedColor = glm::vec4{ rawData[i], rawData[i + 1], rawData[i + 2], rawData[i + 3] } / 255.0f;
		
		// typically albedo textures are srgb but material textures are not. either way the data must be in linear color space during tracing
		if(decodeSRGB) {
			data.emplace_back(srgbToLinear(normalizedColor), normalizedColor.a);
		}
		else {
			data.emplace_back(normalizedColor);
		}
	}

	cudaMalloc(&m_data, data.size() * sizeof(glm::vec4));
	cudaMemcpy(m_data, data.data(), data.size() * sizeof(glm::vec4), cudaMemcpyHostToDevice);

	stbi_image_free(rawData);
}

Texture::Texture(Texture&& src) {
	m_data = src.m_data;
	m_width = src.m_width;
	m_height = src.m_height;

	src.m_data = nullptr;
}

void Texture::operator=(Texture&& src) {
	this->~Texture();
	new (this) Texture(std::move(src));
}

Texture::~Texture() {
	if(m_data) {
		cudaFree(m_data);
	}
}

__device__ glm::vec4 Texture::sample(glm::vec2 uv) const {
	uv = glm::mod(uv, glm::vec2(1.0f));
	uv *= glm::vec2(m_width, m_height);
	glm::uvec2 lower = glm::floor(uv);
	glm::uvec2 upper = glm::ceil(uv);

	if(upper.x == lower.x) {
		upper.x++;
	}

	if(upper.y == lower.y) {
		upper.y++;
	}

	while(upper.x >= m_width) {
		upper.x--;
		lower.x--;
	}
	
	while(upper.y >= m_height) {
		upper.y--;
		lower.y--;
	}

	f32 w1 = (upper.x - uv.x) * (upper.y - uv.y);
	f32 w2 = (uv.x - lower.x) * (upper.y - uv.y);
	f32 w3 = (upper.x - uv.x) * (uv.y - lower.y);
	f32 w4 = (uv.x - lower.x) * (uv.y - lower.y);

	glm::vec4 s1 = m_data[lower.y * m_width + lower.x];
	glm::vec4 s2 = m_data[lower.y * m_width + upper.x];
	glm::vec4 s3 = m_data[upper.y * m_width + lower.x];
	glm::vec4 s4 = m_data[upper.y * m_width + upper.x];

	// bilinear interpolation between 4 nearest texels
	return s1 * w1 + s2 * w2 + s3 * w3 + s4 * w4;
}

glm::vec3 Texture::srgbToLinear(glm::vec3 color) const {
	// based on the srgb spec
	for(u8 i = 0; i < 3; i++) {
		color[i] = color[i] <= 0.04045f ? (color[i] / 12.92f) : std::pow((color[i] + 0.055f) / 1.055f, 2.4f);
	}

	return color;
}