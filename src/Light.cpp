#include "Light.h"

LightManager::LightManager() 
    : core_(nullptr)
    , buffers_initialized_(false) {
}

LightManager::~LightManager() {
    // Buffers will be automatically cleaned up by unique_ptr
}

void LightManager::Initialize(grassland::graphics::Core* core) {
    core_ = core;
    buffers_initialized_ = false;
    
    // 创建光源缓冲区
    size_t max_lights = 16; // 最大支持16个光源
    size_t light_buffer_size = max_lights * sizeof(Light);
    
    core_->CreateBuffer(light_buffer_size, grassland::graphics::BUFFER_TYPE_DYNAMIC, &lights_buffer_);
    core_->CreateBuffer(sizeof(int), grassland::graphics::BUFFER_TYPE_DYNAMIC, &light_count_buffer_);
    
    buffers_initialized_ = true;
    grassland::LogInfo("Light manager initialized with max {} lights", max_lights);
}

void LightManager::AddLight(const Light& light) {
    lights_.push_back(light);
    UpdateBuffers();
    grassland::LogInfo("Added light (total: {})", lights_.size());
}

void LightManager::RemoveLight(size_t index) {
    if (index < lights_.size()) {
        lights_.erase(lights_.begin() + index);
        UpdateBuffers();
        grassland::LogInfo("Removed light at index {} (total: {})", index, lights_.size());
    } else {
        grassland::LogWarning("Cannot remove light at invalid index: {}", index);
    }
}

void LightManager::UpdateLight(size_t index, const Light& light) {
    if (index < lights_.size()) {
        lights_[index] = light;
        UpdateBuffers();
        grassland::LogInfo("Updated light at index {}", index);
    } else {
        grassland::LogWarning("Cannot update light at invalid index: {}", index);
    }
}

void LightManager::UpdateBuffers() {
    if (!buffers_initialized_ || !core_) {
        return;
    }
    
    if (!lights_.empty()) {
        size_t data_size = lights_.size() * sizeof(Light);
        lights_buffer_->UploadData(lights_.data(), data_size);
    }
    
    int enabled_count = GetEnabledLightCount();
    light_count_buffer_->UploadData(&enabled_count, sizeof(int));
    
}

void LightManager::CreateDefaultLights() {
    if (!core_) {
        grassland::LogError("Light manager not initialized before creating default lights");
        return;
    }
    
    lights_.clear();
    
    Light main_directional = Light::CreateDirectionalLight(
        glm::vec3(0.0f, 5.0f, 0.0f),      // 位置（参考点）
        glm::vec3(-0.5f, -1.0f, -0.5f),   // 方向
        glm::vec3(1.0f, 0.95f, 0.9f),     // 暖白色
        1.5f,                             // 强度
        45.0f                             // 锥角
    );
    lights_.push_back(main_directional);
    
    Light red_point = Light::CreatePointLight(
        glm::vec3(-1.5f, 2.0f, 1.0f),     // 位置
        glm::vec3(1.0f, 0.3f, 0.3f),      // 红色
        1.2f                              // 强度
    );
    lights_.push_back(red_point);
    
    Light blue_point = Light::CreatePointLight(
        glm::vec3(1.5f, 2.0f, -1.0f),     // 位置
        glm::vec3(0.3f, 0.3f, 1.0f),      // 蓝色
        1.2f                              // 强度
    );
    lights_.push_back(blue_point);
    
    Light top_area = Light::CreateAreaLight(
        glm::vec3(0.0f, 4.0f, 0.0f),      // 位置
        glm::vec3(0.0f, -1.0f, 0.0f),     // 方向（向下）
        glm::vec2(3.0f, 3.0f),            // 尺寸
        glm::vec3(0.9f, 0.9f, 1.0f),      // 冷白色
        0.8f                              // 强度
    );
    lights_.push_back(top_area);
    
    // 更新缓冲区
    UpdateBuffers();
    grassland::LogInfo("Created {} default lights", lights_.size());
}

int LightManager::GetEnabledLightCount() const {
    int count = 0;
    for (const auto& light : lights_) {
        if (light.enabled) {
            count++;
        }
    }
    return count;
}

const Light* LightManager::FindMainDirectionalLight() const {
    for (const auto& light : lights_) {
        if (light.enabled && light.type == 2) { // 类型2是方向光
            return &light;
        }
    }
    return nullptr;
}