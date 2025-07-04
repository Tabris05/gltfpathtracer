#ifndef SCENE_HPP
#define SCENE_HPP

#define _USE_MATH_DEFINES
#include <cmath>
#include <vector>
#include <filesystem>
#include <glm/glm.hpp>
#include <fastgltf/core.hpp>
#include <cuda_runtime.h>
#include <curand_kernel.h>
#include "typedefs.cuh"
#include "ray.cuh"
#include "texture.cuh"
#include "aabb.cuh"

class Scene {
	public:
		Scene(std::filesystem::path modelPath, f32 modelScale, std::optional<std::filesystem::path> skyboxPath = {});
		~Scene();

		__device__ glm::vec3 trace(Ray r, curandState* state) const;

	private:
		struct Material {
			std::optional<u32> albedoIndex;
			std::optional<u32> normalIndex;
			std::optional<u32> metallicRoughnessIndex;
			std::optional<u32> emissiveIndex;
			glm::vec4 baseColor;
			glm::vec3 emissiveColor;
			f32 emissiveStrength;
			f32 metallic;
			f32 roughness;
			f32 alphaCutoff;
		};

		struct Attribute {
			glm::vec3 normal;
			glm::vec3 tangent;
			glm::vec2 uv;
			f32 bitangentSign;
		};

		struct Mesh {
			AABB boundingBox;
			u32 blasIndex;
			u32 materialIndex;
		};

		struct BLASNode {
			AABB boundingBox;
			u32 leftChildIndex;
			u32 rightChildIndex;
		};

		struct TLASNode {
			AABB boundingBox;
			u32 leftChildIndex;
			u32 rightChildIndex;
		};

		enum Axis : u8 {
			X = 0,
			Y = 1,
			Z = 2
		};

		struct BsdfResult {
			__device__ BsdfResult() = default;
			__device__ BsdfResult(glm::vec3 color) : color{ color }, shouldTerminate{ true } {}
			__device__ BsdfResult(glm::vec3 color, Ray nextRay) : color{ color }, nextRay{ nextRay }, shouldTerminate { glm::all(glm::equal(color, glm::vec3 { 0.0f }))} {}

			Ray nextRay;
			glm::vec3 color;
			bool shouldTerminate;
		};

		void processNode(
			const fastgltf::Asset& asset,
			std::vector<glm::vec3>& positionBuffer,
			std::vector<Attribute>& attributeBuffer,
			std::vector<glm::uvec3>& indexBuffer,
			std::vector<Mesh>& meshes,
			std::vector<BLASNode>& blasPool,
			u64 index,
			glm::mat4 transform
		);
		u32 makeBLAS(std::vector<BLASNode>& blasPool, std::vector<glm::vec3>& positionBuffer, std::vector<glm::uvec3>& indexBuffer, u32 begin, u32 end);
		u32 makeTLAS(std::vector<TLASNode>& tlasPool, std::vector<Mesh>& meshes, u32 begin, u32 end);
		__device__ HitRecord traceBLAS(Ray ray, u32 blasIndex, f32 max) const;
		__device__ HitRecord traceTLAS(Ray ray) const;

		__device__ BsdfResult bsdf(HitRecord record, glm::vec3 viewDir, curandState* state) const;
		__device__ glm::vec3 ggxVndf(glm::vec3 viewDir, f32 alpha, f32 r1, f32 r2) const;
		__device__ f32 ggxMasking(glm::vec3 normal, glm::vec3 viewDir, f32 alpha2) const;
		__device__ f32 ggxMaskingShadowing(glm::vec3 normal, glm::vec3 rayDir, glm::vec3 viewDir, f32 alpha2) const;
		__device__ glm::vec3 fresnelSchlick(glm::vec3 f0, glm::vec3 viewDir, glm::vec3 half) const;

		// Texture is not trivially destructable so we must keep host-side copy of texture data to ensure destructor is called
		std::vector<Texture> m_texVec;
		Texture* m_textures;
		Material* m_materials;
		glm::vec3* m_positionBuffer;
		Attribute* m_attributeBuffer;
		glm::uvec3* m_indexBuffer;
		Mesh* m_meshes;
		BLASNode* m_blasPool;
		TLASNode* m_tlasPool;

		std::optional<Texture> m_skyboxTex;

		// same as Blender Cycles default
		static constexpr u8 m_maxBounces = 12;
};

#endif