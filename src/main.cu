#include "camera.cuh"
#include <iostream>
#include <sstream>

int main() {
	std::stringstream ss;
	std::string line;
	std::string modelPath;
	f32 modelScale;
	std::string skyboxPath;
	glm::vec3 camPos;
	glm::vec3 lookAt;
	glm::u16vec2 resolution;
	f32 fovY;
	u32 sampleCount;

	std::cout << "Enter path to model: ";
	std::getline(std::cin, modelPath);

	std::cout << "Enter model scale (float, positive): ";
	std::getline(std::cin, line);
	ss = std::stringstream(line);
	ss >> modelScale;

	std::cout << "Enter path to skybox (optional): ";
	std::getline(std::cin, skyboxPath);

	std::cout << "Enter camera position (3 floats seperated by spaces): ";
	std::getline(std::cin, line);
	ss = std::stringstream(line);
	ss >> camPos.x >> camPos.y >> camPos.z;

	std::cout << "Enter camera look-at position (3 floats seperated by spaces): ";
	std::getline(std::cin, line);
	ss = std::stringstream(line);
	ss >> lookAt.x >> lookAt.y >> lookAt.z;

	std::cout << "Enter image resolution (2 integers seperated by spaces): ";
	std::getline(std::cin, line);
	ss = std::stringstream(line);
	ss >> resolution.x >> resolution.y;

	std::cout << "Enter vertical FOV (float, in degrees): ";
	std::getline(std::cin, line);
	ss = std::stringstream(line);
	ss >> fovY;

	std::cout << "Enter sample count: ";
	std::getline(std::cin, line);
	ss = std::stringstream(line);
	ss >> sampleCount;

	std::cout << "\nStarting trace...\n";

	Scene scene(modelPath, modelScale, skyboxPath.empty() ? std::nullopt : std::optional<std::filesystem::path>(skyboxPath));
	Camera cam(camPos, lookAt, resolution, fovY, sampleCount);

	cam.render(scene);
}