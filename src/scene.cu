#include "scene.cuh"

#include <fastgltf/glm_element_traits.hpp>
#include <glm/gtc/type_ptr.hpp>
#include <glm/gtx/quaternion.hpp>
#include <mikktspace/mikktspace.h>
#include <random>
#include <chrono>


Scene::Scene(std::filesystem::path modelPath, f32 modelScale, std::optional<std::filesystem::path> skyboxPath) {
	std::vector<Material> materials;
	std::vector<glm::vec3> positionBuffer;
	std::vector<Attribute> attributeBuffer;
	std::vector<glm::uvec3> indexBuffer;
	std::vector<Mesh> meshes;
	std::vector<BLASNode> blasPool;
	std::vector<TLASNode> tlasPool;

	const fastgltf::Extensions extensions =
		fastgltf::Extensions::KHR_materials_emissive_strength;

	fastgltf::Parser parser{ extensions };
	fastgltf::GltfDataBuffer data = std::move(fastgltf::GltfDataBuffer::FromPath(modelPath).get());

	const fastgltf::Options options =
		fastgltf::Options::GenerateMeshIndices |
		fastgltf::Options::LoadExternalBuffers |
		fastgltf::Options::LoadExternalImages;

	const fastgltf::Asset asset{ std::move(parser.loadGltf(data, modelPath.parent_path(), options).get()) };

	for(u32 index = 0; index < asset.textures.size(); index++) {
		bool decodeSRGB = false;
		for(const fastgltf::Material& mat : asset.materials) {
			if(mat.pbrData.baseColorTexture.has_value() && mat.pbrData.baseColorTexture.value().textureIndex == index) {
				decodeSRGB = true;
				break;
			}
		}
		m_texVec.push_back({ std::get<fastgltf::sources::Array>(asset.images[asset.textures[index].imageIndex.value()].data).bytes, decodeSRGB });
	}

	for(const fastgltf::Material& mat : asset.materials) {
		materials.push_back({
			mat.pbrData.baseColorTexture.has_value() ? mat.pbrData.baseColorTexture.value().textureIndex : std::optional<u32>{},
			mat.normalTexture.has_value() ? mat.normalTexture.value().textureIndex : std::optional<u32>{},
			mat.pbrData.metallicRoughnessTexture.has_value() ? mat.pbrData.metallicRoughnessTexture.value().textureIndex : std::optional<u32>{},
			mat.emissiveTexture.has_value() ? mat.emissiveTexture.value().textureIndex : std::optional<u32>{},
			glm::make_vec4(mat.pbrData.baseColorFactor.data()),
			glm::make_vec3(mat.emissiveFactor.data()),
			mat.emissiveStrength,
			mat.pbrData.metallicFactor,
			mat.pbrData.roughnessFactor,
			mat.alphaMode == fastgltf::AlphaMode::Mask ? mat.alphaCutoff : 0.0f
		});
	}

	for(u64 i : asset.scenes[asset.defaultScene.value_or(0)].nodeIndices) {
		glm::mat4 transform{ 1.0f };
		processNode(asset, positionBuffer, attributeBuffer, indexBuffer, meshes, blasPool, i, transform);
	}

	makeTLAS(tlasPool, meshes, 0, meshes.size());

	// normalize scale of model
	glm::vec3 size = tlasPool[0].boundingBox.max - tlasPool[0].boundingBox.min;
	glm::mat4 baseTransform = glm::scale(glm::mat4{ 1.0f }, glm::vec3(modelScale / std::max(size.x, std::max(size.y, size.z))));
	glm::mat3 normalTransform = glm::mat3{ glm::transpose(glm::inverse(baseTransform)) };
	
	for(u32 i = 0; i < positionBuffer.size(); i++) {
		positionBuffer[i] = glm::vec3{ baseTransform * glm::vec4 { positionBuffer[i], 1.0f } };
		attributeBuffer[i].tangent = glm::normalize(glm::vec3{ baseTransform * glm::vec4{ attributeBuffer[i].tangent, 1.0f } });
		attributeBuffer[i].normal = glm::normalize(normalTransform * attributeBuffer[i].normal);
	}
	
	for(u32 i = 0; i < meshes.size(); i++) {
		meshes[i].boundingBox.min = glm::vec3{ baseTransform * glm::vec4 { meshes[i].boundingBox.min, 1.0f } };
		meshes[i].boundingBox.max = glm::vec3{ baseTransform * glm::vec4 { meshes[i].boundingBox.max, 1.0f } };
	}
	
	for(u32 i = 0; i < blasPool.size(); i++) {
		blasPool[i].boundingBox.min = glm::vec3{ baseTransform * glm::vec4 { blasPool[i].boundingBox.min, 1.0f } };
		blasPool[i].boundingBox.max = glm::vec3{ baseTransform * glm::vec4 { blasPool[i].boundingBox.max, 1.0f } };
	}
	
	for(u32 i = 0; i < tlasPool.size(); i++) {
		tlasPool[i].boundingBox.min = glm::vec3{ baseTransform * glm::vec4 { tlasPool[i].boundingBox.min, 1.0f } };
		tlasPool[i].boundingBox.max = glm::vec3{ baseTransform * glm::vec4 { tlasPool[i].boundingBox.max, 1.0f } };
	}

	if(skyboxPath.has_value()) {
		m_skyboxTex = Texture(skyboxPath.value());
	}
	
	cudaMalloc(&m_textures, m_texVec.size() * sizeof(m_texVec[0]));
	cudaMemcpy(m_textures, m_texVec.data(), m_texVec.size() * sizeof(m_texVec[0]), cudaMemcpyHostToDevice);
	cudaMalloc(&m_materials, materials.size() * sizeof(materials[0]));
	cudaMemcpy(m_materials, materials.data(), materials.size() * sizeof(materials[0]), cudaMemcpyHostToDevice);
	cudaMalloc(&m_positionBuffer, positionBuffer.size() * sizeof(positionBuffer[0]));
	cudaMemcpy(m_positionBuffer, positionBuffer.data(), positionBuffer.size() * sizeof(positionBuffer[0]), cudaMemcpyHostToDevice);
	cudaMalloc(&m_attributeBuffer, attributeBuffer.size() * sizeof(attributeBuffer[0]));
	cudaMemcpy(m_attributeBuffer, attributeBuffer.data(), attributeBuffer.size() * sizeof(attributeBuffer[0]), cudaMemcpyHostToDevice);
	cudaMalloc(&m_indexBuffer, indexBuffer.size() * sizeof(indexBuffer[0]));
	cudaMemcpy(m_indexBuffer, indexBuffer.data(), indexBuffer.size() * sizeof(indexBuffer[0]), cudaMemcpyHostToDevice);
	cudaMalloc(&m_meshes, meshes.size() * sizeof(meshes[0]));
	cudaMemcpy(m_meshes, meshes.data(), meshes.size() * sizeof(meshes[0]), cudaMemcpyHostToDevice);
	cudaMalloc(&m_blasPool, blasPool.size() * sizeof(blasPool[0]));
	cudaMemcpy(m_blasPool, blasPool.data(), blasPool.size() * sizeof(blasPool[0]), cudaMemcpyHostToDevice);
	cudaMalloc(&m_tlasPool, tlasPool.size() * sizeof(tlasPool[0]));
	cudaMemcpy(m_tlasPool, tlasPool.data(), tlasPool.size() * sizeof(tlasPool[0]), cudaMemcpyHostToDevice);
}

Scene::~Scene() {
	cudaFree(m_textures);
	cudaFree(m_materials);
	cudaFree(m_positionBuffer);
	cudaFree(m_attributeBuffer);
	cudaFree(m_indexBuffer);
    cudaFree(m_meshes);
	cudaFree(m_blasPool);
	cudaFree(m_tlasPool);
}

__device__ glm::vec3 Scene::trace(Ray ray, curandState* state) const {
	glm::vec3 color{ 1.0f };

	for(u8 i = 0; i < m_maxBounces; i++) {

		// if we hit, evaluate the bsdf at the hit point
		if(HitRecord record = traceTLAS(ray)) {
			BsdfResult result = bsdf(record, -ray.direction, state);
			color *= result.color;
			ray = result.nextRay;
			
			if(result.shouldTerminate) {
				return color;
			}
		}

		// if we miss but have a skybox, look up the skybox value in the given direction
		else if(m_skyboxTex.has_value()) {
			glm::vec2 uv = glm::vec2{ atan2(ray.direction.z, ray.direction.x) * 0.5f, asin(ray.direction.y) } / static_cast<f32>(M_PI) + 0.5f;
			uv.y = 1.0f - uv.y;
			color *= glm::vec3(m_skyboxTex.value().sample(uv));
			return color;
		}

		// otherwise assume a black background
		else {
			return glm::vec3{ 0.0f };
		}
	}

	// we early return if we hit an emissive surface or the skybox
	// if we hit neither of these then the pixel must be unlit
	return glm::vec3{ 0.0f };
}

void Scene::processNode(const fastgltf::Asset& asset,
	std::vector<glm::vec3>& positionBuffer,
	std::vector<Attribute>& attributeBuffer,
	std::vector<glm::uvec3>& indexBuffer,
	std::vector<Mesh>& meshes,
	std::vector<BLASNode>& blasPool,
	u64 index,
	glm::mat4 transform
) {
	const fastgltf::Node& curNode = asset.nodes[index];

	transform *= std::visit(fastgltf::visitor{
		[](fastgltf::math::fmat4x4 matrix) {
			return glm::make_mat4(matrix.data());
		},
		[](fastgltf::TRS trs) {
			return glm::translate(glm::mat4{ 1.0f }, glm::make_vec3(trs.translation.data()))
			* glm::toMat4(glm::make_quat(trs.rotation.value_ptr()))
			* glm::scale(glm::mat4{ 1.0f }, glm::make_vec3(trs.scale.data()));
		}
	}, curNode.transform);

	if(curNode.meshIndex.has_value()) {
		const glm::mat3 normalTransform{ glm::transpose(glm::inverse(transform)) };
		const fastgltf::Mesh& curMesh = asset.meshes[curNode.meshIndex.value()];
		for(const fastgltf::Primitive& curPrimitive : curMesh.primitives) {
			AABB boundingBox;
			u32 oldVerticesSize = positionBuffer.size();
			u32 oldIndicesSize = indexBuffer.size();

			const fastgltf::Accessor& indexAccessor = asset.accessors[curPrimitive.indicesAccessor.value()];
			fastgltf::iterateAccessorWithIndex<u32>(asset, indexAccessor, [&indexBuffer, &oldVerticesSize](u32 vertexIndex, u32 index) {
				if(index % 3 == 0) {
					indexBuffer.push_back({ vertexIndex + oldVerticesSize, 0, 0 });
				}
				else {
					indexBuffer.back()[index % 3] = vertexIndex + oldVerticesSize;
				}
			});

			const fastgltf::Accessor& positionAccessor = asset.accessors[curPrimitive.findAttribute("POSITION")->accessorIndex];
			fastgltf::iterateAccessor<glm::vec3>(asset, positionAccessor, [&positionBuffer, transform, &boundingBox](glm::vec3 pos) {
				glm::vec3 vertex = glm::vec3{ transform * glm::vec4{ pos, 1.0f } };
				boundingBox.min = glm::min(boundingBox.min, vertex);
				boundingBox.max = glm::max(boundingBox.max, vertex);
				positionBuffer.push_back(vertex);
			});

			const fastgltf::Accessor& normalAccessor = asset.accessors[curPrimitive.findAttribute("NORMAL")->accessorIndex];
			fastgltf::iterateAccessor<glm::vec3>(asset, normalAccessor, [&attributeBuffer, oldVerticesSize, normalTransform](glm::vec3 normal) {
				attributeBuffer.push_back({ glm::normalize(normalTransform * normal) });
			});

			const fastgltf::Attribute* uvAccessorIndex;
			if((uvAccessorIndex = curPrimitive.findAttribute("TEXCOORD_0")) != curPrimitive.attributes.cend()) {
				const fastgltf::Accessor& uvAccessor = asset.accessors[uvAccessorIndex->accessorIndex];
				fastgltf::iterateAccessorWithIndex<glm::vec2>(asset, uvAccessor, [&attributeBuffer, oldVerticesSize](glm::vec2 uv, u32 index) {
					attributeBuffer[index + oldVerticesSize].uv = uv;
				});
			}

			const fastgltf::Attribute* tangentAccessorIndex;
			if((tangentAccessorIndex = curPrimitive.findAttribute("TANGENT")) != curPrimitive.attributes.cend()) {
				const fastgltf::Accessor& tangentAccessor = asset.accessors[tangentAccessorIndex->accessorIndex];
				fastgltf::iterateAccessorWithIndex<glm::vec4>(asset, tangentAccessor, [&attributeBuffer, oldVerticesSize, transform](glm::vec4 tangent, size_t index) {
					attributeBuffer[index + oldVerticesSize].tangent = glm::normalize(glm::vec3{ transform * glm::vec4{ glm::vec3{ tangent }, 1.0f } });
					attributeBuffer[index + oldVerticesSize].bitangentSign = tangent.w;
				});
			}
			else if(uvAccessorIndex != curPrimitive.attributes.cend()) {
				struct UsrPtr {
					u64 indexOffset;
					std::vector<glm::vec3>& positions;
					std::vector<Attribute>& attributes;
					std::vector<glm::uvec3>& indices;
				} usrPtr{ oldIndicesSize, positionBuffer, attributeBuffer, indexBuffer };

				SMikkTSpaceInterface interface {
					// get number of faces (triangles) in primitive
					[](const SMikkTSpaceContext* ctx) -> i32 {
						UsrPtr* data = static_cast<UsrPtr*>(ctx->m_pUserData);
						return (data->indices.size() - data->indexOffset) / 3;
						},

						// get number of vertices in face
						[](const SMikkTSpaceContext*, const i32) -> i32 {
						return 3;
						},

						// get position of specified vertex
						[](const SMikkTSpaceContext* ctx, f32 outPos[], const i32 face, const i32 vert) {
						UsrPtr* data = static_cast<UsrPtr*>(ctx->m_pUserData);
						memcpy(outPos, &data->positions[data->indices[data->indexOffset + face][vert]], sizeof(glm::vec3));
						},

						// get normal of specified vertex
						[](const SMikkTSpaceContext* ctx, f32 outNorm[], const i32 face, const i32 vert) {
						UsrPtr* data = static_cast<UsrPtr*>(ctx->m_pUserData);
						memcpy(outNorm, &data->attributes[data->indices[data->indexOffset + face][vert]].normal, sizeof(glm::vec3));
						},

						// get uv of specified vertex
						[](const SMikkTSpaceContext* ctx, f32 outUV[], const i32 face, const i32 vert) {
						UsrPtr* data = static_cast<UsrPtr*>(ctx->m_pUserData);
						memcpy(outUV, &data->attributes[data->indices[data->indexOffset + face][vert]].uv, sizeof(glm::vec2));
						},

						// set tangent of specified vertex
						[](const SMikkTSpaceContext* ctx, const f32 inTangent[], const f32 sign, const i32 face, const i32 vert) {
						UsrPtr* data = static_cast<UsrPtr*>(ctx->m_pUserData);
						u64 vertexIndex = data->indices[data->indexOffset + face][vert];
						memcpy(&data->attributes[vertexIndex].tangent, inTangent, sizeof(glm::vec3));
						data->attributes[vertexIndex].bitangentSign = sign;
						}
				};
				SMikkTSpaceContext ctx{ &interface, &usrPtr };
				genTangSpaceDefault(&ctx);
			}

			meshes.push_back({ boundingBox, makeBLAS(blasPool, positionBuffer, indexBuffer, oldIndicesSize, indexBuffer.size()), static_cast<u32>(curPrimitive.materialIndex.value_or(0)) });
		}
	}
	for(u64 i : curNode.children) {
		processNode(asset, positionBuffer, attributeBuffer, indexBuffer, meshes, blasPool, i, transform);
	}
}

u32 Scene::makeBLAS(std::vector<BLASNode>& blasPool, std::vector<glm::vec3>& positionBuffer, std::vector<glm::uvec3>& indexBuffer, u32 begin, u32 end) {
	AABB boundingBox;
	for(u32 i = begin; i < end; i++) {
		for(u8 j = 0; j < 3; j++) {
			boundingBox.min = glm::min(boundingBox.min, positionBuffer[indexBuffer[i][j]]);
			boundingBox.max = glm::max(boundingBox.max, positionBuffer[indexBuffer[i][j]]);
		}
	}

	// pad the aabb so that objects that lay entirely in one plane do not produce 0-width aabbs
	boundingBox.min += std::numeric_limits<f32>::epsilon();
	boundingBox.max -= std::numeric_limits<f32>::epsilon();

	u32 index = blasPool.size();
	blasPool.push_back({ boundingBox });
	
	// since 0 is the root and can never be a child, having a child of 0 can represent being a leaf
	u32 span = end - begin;
	if(span == 1) {
		blasPool[index].leftChildIndex = 0;
		blasPool[index].rightChildIndex = begin;
		return index;
	}

	// divide bounding box along longest axis
	Axis axis;
	f32 longest = 0.0f;
	glm::vec3 lengths = boundingBox.max - boundingBox.min;
	for(u8 i = 0; i < 3; i++) {
		if(lengths[i] > longest) {
			longest = lengths[i];
			axis = static_cast<Axis>(i);
		}
	}

	// sort the meshes from closest to furthest based on the selected axis
	std::sort(indexBuffer.begin() + begin, indexBuffer.begin() + end, [&positionBuffer, axis](glm::uvec3 a, glm::uvec3 b) {
		f32 aMin = std::numeric_limits<f32>::infinity();
		f32 bMin = std::numeric_limits<f32>::infinity();
		for(u8 i = 0; i < 3; i++) {
			aMin = std::min(aMin, positionBuffer[a[i]][axis]);
			bMin = std::min(aMin, positionBuffer[b[i]][axis]);
		}

		return aMin < bMin;
	});

	// create child nodes containing the left and right halfs of the sorted range of triangles
	u32 midpoint = begin + (end - begin) / 2;
	blasPool[index].leftChildIndex = makeBLAS(blasPool, positionBuffer, indexBuffer, begin, midpoint);
	blasPool[index].rightChildIndex = makeBLAS(blasPool, positionBuffer, indexBuffer, midpoint, end);

	return index;
}

u32 Scene::makeTLAS(std::vector<TLASNode>& tlasPool, std::vector<Mesh>& meshes, u32 begin, u32 end) {
	AABB boundingBox;
	for(u32 i = begin; i < end; i++) {
			boundingBox.min = glm::min(boundingBox.min, meshes[i].boundingBox.min);
			boundingBox.max = glm::max(boundingBox.max, meshes[i].boundingBox.max);
	}

	// pad the aabb so that objects that lay entirely in one plane do not produce 0-width aabbs
	boundingBox.min += std::numeric_limits<f32>::epsilon();
	boundingBox.max -= std::numeric_limits<f32>::epsilon();

	u32 index = tlasPool.size();
	tlasPool.push_back({ boundingBox });

	// since 0 is the root and can never be a child, having a child of 0 can represent being a leaf
	u32 span = end - begin;
	if(span == 1) {
		tlasPool[index].leftChildIndex = 0;
		tlasPool[index].rightChildIndex = begin;
		return index;
	}

	// divide bounding box along longest axis
	Axis axis;
	f32 longest = 0.0f;
	glm::vec3 lengths = boundingBox.max - boundingBox.min;
	for(u8 i = 0; i < 3; i++) {
		if(lengths[i] > longest) {
			longest = lengths[i];
			axis = static_cast<Axis>(i);
		}
	}

	// sort the meshes from closest to furthest based on the selected axis
	std::sort(meshes.begin() + begin, meshes.begin() + end, [this, axis](Mesh a, Mesh b) {
		return a.boundingBox.min[axis] < b.boundingBox.min[axis];
	});

	// create child nodes containing the left and right halfs of the sorted range of meshes
	u32 midpoint = begin + (end - begin) / 2;
	tlasPool[index].leftChildIndex = makeTLAS(tlasPool, meshes, begin, midpoint);
	tlasPool[index].rightChildIndex = makeTLAS(tlasPool, meshes, midpoint, end);

	return index;
}

__device__ HitRecord Scene::traceBLAS(Ray ray, u32 blasIndex, f32 max) const {
	u32 stack[32];
	u8 stackPtr = 0;
	stack[stackPtr++] = blasIndex;

	HitRecord ret;

	while(stackPtr > 0) {
		const BLASNode cur = m_blasPool[stack[--stackPtr]];

		// if we are at a leaf node transition to tracing the triangle at this leaf and record the vertex data attached to the triangle
		if(cur.leftChildIndex == 0) {
			const glm::uvec3 index = m_indexBuffer[cur.rightChildIndex];
			if(HitRecord record = ray.triangleIntersection(m_positionBuffer[index.x], m_positionBuffer[index.y], m_positionBuffer[index.z], max)) {
				ret = record.setTriangle(index);
				max = ret.distance;
			}
		}
		else {
			std::optional<f32> t1 = ray.boxIntersection(m_blasPool[cur.leftChildIndex].boundingBox, max);
			std::optional<f32> t2 = ray.boxIntersection(m_blasPool[cur.rightChildIndex].boundingBox, max);

			// always evaluate the closer child first, since we may be able to skip evaluating the further child entirely
			if(t1 && t2) {
				u32 closerChild = t1.value() < t2.value() ? cur.leftChildIndex : cur.rightChildIndex;
				u32 furtherChild = t1.value() >= t2.value() ? cur.leftChildIndex : cur.rightChildIndex;

				stack[stackPtr++] = furtherChild;
				stack[stackPtr++] = closerChild;
			}
			else if(t1) {
				stack[stackPtr++] = cur.leftChildIndex;
			}
			else if(t2) {
				stack[stackPtr++] = cur.rightChildIndex;
			}
		}
	}

	return ret;
}

__device__ HitRecord Scene::traceTLAS(Ray ray) const {
	f32 max = std::numeric_limits<f32>::infinity();
	u32 stack[32];
	u8 stackPtr = 0;
	stack[stackPtr++] = 0;

	HitRecord ret;

	while(stackPtr > 0) {
		const TLASNode cur = m_tlasPool[stack[--stackPtr]];

		// if we are at a leaf node transition to tracing the blas at this leaf and record the material attached to the blas
		if(cur.leftChildIndex == 0) {
			if(HitRecord record = traceBLAS(ray, m_meshes[cur.rightChildIndex].blasIndex, max)) {
				ret = record.setMaterial(m_meshes[cur.rightChildIndex].materialIndex);
				max = ret.distance;
			}
		}
		else {
			std::optional<f32> t1 = ray.boxIntersection(m_tlasPool[cur.leftChildIndex].boundingBox, max);
			std::optional<f32> t2 = ray.boxIntersection(m_tlasPool[cur.rightChildIndex].boundingBox, max);

			// always evaluate the closer child first, since we may be able to skip evaluating the further child entirely
			if(t1 && t2) {
				u32 closerChild = t1.value() < t2.value() ? cur.leftChildIndex : cur.rightChildIndex;
				u32 furtherChild = t1.value() >= t2.value() ? cur.leftChildIndex : cur.rightChildIndex;

				stack[stackPtr++] = furtherChild;
				stack[stackPtr++] = closerChild;
			}
			else if(t1) {
				stack[stackPtr++] = cur.leftChildIndex;
			}
			else if(t2) {
				stack[stackPtr++] = cur.rightChildIndex;
			}
		}
	}

	return ret;
}

__device__ Scene::BsdfResult Scene::bsdf(HitRecord record, glm::vec3 viewDir, curandState* state) const {

	// interpolate vertex attributes based on hit position
	Material mat = m_materials[record.materialIndex];
	glm::vec3 position = m_positionBuffer[record.triangle.x] * record.barycentric.x
		+ m_positionBuffer[record.triangle.y] * record.barycentric.y
		+ m_positionBuffer[record.triangle.z] * record.barycentric.z;

	glm::vec2 uv = m_attributeBuffer[record.triangle.x].uv * record.barycentric.x
		+ m_attributeBuffer[record.triangle.y].uv * record.barycentric.y
		+ m_attributeBuffer[record.triangle.z].uv * record.barycentric.z;

	glm::vec3 normal = glm::normalize(
		m_attributeBuffer[record.triangle.x].normal * record.barycentric.x
		+ m_attributeBuffer[record.triangle.y].normal * record.barycentric.y
		+ m_attributeBuffer[record.triangle.z].normal * record.barycentric.z
	);

	glm::vec3 tangent = glm::normalize(
		m_attributeBuffer[record.triangle.x].tangent * record.barycentric.x
		+ m_attributeBuffer[record.triangle.y].tangent * record.barycentric.y
		+ m_attributeBuffer[record.triangle.z].tangent * record.barycentric.z
	);

	glm::vec3 emissive = mat.emissiveColor * mat.emissiveStrength * (mat.emissiveIndex.has_value() ? m_textures[mat.emissiveIndex.value()].sample(uv) : glm::vec3{ 1.0f });

	// not considering emissive surfaces that are still dark enough to see other light reflected onto it
	if(glm::any(glm::greaterThan(emissive, glm::vec3{ 0.0f }))) {
		return BsdfResult(emissive);
	}

	// material properties
	glm::vec4 baseColor = mat.baseColor * (mat.albedoIndex.has_value() ? m_textures[mat.albedoIndex.value()].sample(uv) : glm::vec4{ 1.0f });
	glm::vec3 albedo = baseColor;
	f32 metallic = mat.metallic * (mat.metallicRoughnessIndex.has_value() ? m_textures[mat.metallicRoughnessIndex.value()].sample(uv).b : 1.0f);
	f32 roughness = mat.roughness * (mat.metallicRoughnessIndex.has_value() ? m_textures[mat.metallicRoughnessIndex.value()].sample(uv).g : 1.0f);
	f32 alpha = roughness * roughness;
	glm::vec3 diffuseCol = albedo * (1.0f - metallic);

	// ignore hits that fail alpha-test
	if(baseColor.a < mat.alphaCutoff) {
		return BsdfResult(glm::vec3{ 1.0f }, Ray(position, -viewDir));
	}

	// offset position to avoid self-intersection
	if(baseColor.a < 1.0f) {
		position += normal * std::numeric_limits<f32>::epsilon();
	}

	// check if we hit a front face or back face
	bool frontFacing = true;
	if(glm::dot(normal, viewDir) < 0) {
		normal *= -1.0f;
		tangent *= -1.0f;
		frontFacing = false;
	}

	// use per-pixel normal if normal map is present
	if (mat.normalIndex.has_value()) {
		glm::vec3 bitangent = glm::cross(normal, tangent) * m_attributeBuffer[record.triangle.x].bitangentSign;
		glm::mat3 tbn{ tangent, bitangent, normal };
		normal = glm::normalize(tbn * (m_textures[mat.normalIndex.value()].sample(uv) * 2.0f - 1.0f));
	}

	// orthonormal basis matrix converts from tangent space to world space
	glm::mat3 onb;
	onb[2] = normal;
	onb[1] = glm::normalize(glm::cross(normal, (std::abs(normal.x) > 0.9f) ? glm::vec3(0.0f, 1.0f, 0.0f) : glm::vec3(1.0f, 0.0f, 0.0f)));
	onb[0] = glm::cross(onb[2], onb[1]);

	// importance-sample the ggx ndf to get microfacet normal
	// (vndf is in tangent space so must convert view vector to tangent space using inverse onb and then convert result back to world space using onb)
	glm::vec3 half = glm::normalize(onb * ggxVndf(glm::normalize(glm::transpose(onb) * viewDir), alpha, curand_uniform(state), curand_uniform(state)));

	// compute fresnel term (percentage of light that is reflected specularly)
	glm::vec3 f0 = glm::mix(glm::vec3{ 0.04f }, albedo, metallic);
	glm::vec3 f = fresnelSchlick(f0, viewDir, half);

	// compute transmittance (probability that light passes through the surface instead of bouncing off of it
	f32 ri = frontFacing ? 1.0f / 1.5f : 1.5f;
	f32 cosTheta = glm::dot(viewDir, normal);
	f32 sinTheta = sqrt(1.0f - cosTheta * cosTheta);
	f32 r0 = (1.0f - ri) / (1.0f + ri);
	r0 *= r0;
	f32 fr = r0 + (1.0f - r0) * pow(1.0f - cosTheta, 5.0f);
	f32 transmittance = (1.0f - fr) * (1.0f - baseColor.a) + std::numeric_limits<f32>::epsilon();

	// btdf (light transmits through the surface, surfaces are considered infinitely thin so no refraction)
	if(ri * sinTheta <= 1.0f && transmittance >= curand_uniform(state)) {
		return BsdfResult(glm::vec3{ 1.0f / transmittance }, Ray(position, -viewDir));
	}

	f32 pSpecular = (f.x + f.y + f.z) / 3.0f;

	BsdfResult result;

	// diffuse brdf (light bounces in a random direction)
	if(pSpecular < curand_uniform(state)) {

		// diffuse surfaces reflect light proportionally to the cosine between the light angle and the surface angle
		// this segment of code generates ray directions proportional to the cosine of the angle with the normal
		// this ensures more important directions are sampled more often
		f32 r1 = curand_uniform(state);
		f32 r2 = curand_uniform(state);
		f32 phi = 2.0f * static_cast<f32>(M_PI) * r1;
		glm::vec3 rayDir = glm::normalize(onb * glm::vec3{ cos(phi) * sqrt(r2), sin(phi) * sqrt(r2), sqrt(1 - r2) });

		result = BsdfResult((1.0f - f) * diffuseCol * (1.0f - transmittance) / (1.0f - pSpecular), Ray(position, rayDir));
	}

	// specular brdf (light is reflected about the microfacet normal of the surface)
	else {
		f32 alpha2 = alpha * alpha;
		glm::vec3 rayDir = glm::reflect(-viewDir, half);
	
		if (glm::dot(normal, rayDir) > 0.0f && glm::dot(normal, viewDir) > 0.0f) {
			f32 g1 = ggxMasking(normal, viewDir, alpha2);
			f32 g2 = ggxMaskingShadowing(normal, rayDir, viewDir, alpha2);
		
			result = BsdfResult(f * g2 / g1 * (1.0f - transmittance) / pSpecular, Ray(position, rayDir));
		}
		else {
			result = BsdfResult(glm::vec3{ 0.0f });
		}
	}

	return result;
}

__device__ glm::vec3 Scene::ggxVndf(glm::vec3 viewDir, f32 alpha, f32 r1, f32 r2) const {
	glm::vec3 v = glm::normalize(viewDir * glm::vec3{ alpha, alpha, 1.0f });
	glm::vec3 t1 = (v.z < 0.999f) ? glm::normalize(glm::cross(v, glm::vec3{ 0.0f, 0.0f, 1.0f })) : glm::vec3{ 0.0f, 1.0f, 0.0f };
	glm::vec3 t2 = glm::cross(t1, v);

	f32 a = 1.0f / (1.0f + v.z);
	f32 r = sqrt(r1);
	f32 phi = (r2 < a ? r2 / a : static_cast<f32>(M_PI) + (r2 - a) / (1.0f - a)) * static_cast<f32>(M_PI);
	f32 p1 = r * cos(phi);
	f32 p2 = r * sin(phi) * (r2 < a ? 1.0f : v.z);

	glm::vec3 n = p1 * t1 + p2 * t2 + sqrt(fmaxf(1.0f - p1 * p1 - p2 * p2, 0.0f)) * v;

	return glm::normalize(glm::vec3(alpha * n.x, alpha * n.y, fmaxf(n.z, 0.0f)));
}

__device__ f32 Scene::ggxMasking(glm::vec3 normal, glm::vec3 viewDir, f32 alpha2) const {
	f32 nDotV = fmaxf(glm::dot(normal, viewDir), std::numeric_limits<f32>::epsilon());
	return 2.0f * nDotV / (sqrt(alpha2 + (1.0f - alpha2) * nDotV * nDotV) + nDotV);
}

__device__ f32 Scene::ggxMaskingShadowing(glm::vec3 normal, glm::vec3 rayDir, glm::vec3 viewDir, f32 alpha2) const {
	f32 nDotL = fmaxf(glm::dot(normal, rayDir), std::numeric_limits<f32>::epsilon());
	f32 nDotV = fmaxf(glm::dot(normal, viewDir), std::numeric_limits<f32>::epsilon());

	f32 denomA = nDotV * sqrt(alpha2 + (1.0f - alpha2) * nDotL * nDotL);
	f32 denomB = nDotL * sqrt(alpha2 + (1.0f - alpha2) * nDotV * nDotV);

	return 2.0f * nDotL * nDotV / (denomA + denomB);
}

__device__ glm::vec3 Scene::fresnelSchlick(glm::vec3 f0, glm::vec3 viewDir, glm::vec3 half) const {
	return f0 + (1.0f - f0) * pow(1.0f - glm::clamp(glm::dot(viewDir, half), 0.0f, 1.0f), 5.0f);
}
