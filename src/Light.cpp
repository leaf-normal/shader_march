#include "Light.h"
#include "Entity.h"
#include "Material.h"
#include "glm/gtc/matrix_transform.hpp"

LightManager::LightManager() 
    : core_(nullptr)
    , scene_(nullptr)
    , buffers_initialized_(false) {
}

LightManager::~LightManager() {
    // Buffers will be automatically cleaned up by unique_ptr
}

float CalculateLightPowerCPU(const Light& light) ;

void LightManager::Initialize(grassland::graphics::Core* core, Scene* scene) {
    core_ = core;
    scene_ = scene;
    buffers_initialized_ = false;
    total_power_ = 0.0f;
    
    size_t max_lights = 32; 
    size_t light_buffer_size = max_lights * sizeof(Light);
    
    core_->CreateBuffer(light_buffer_size, grassland::graphics::BUFFER_TYPE_DYNAMIC, &lights_buffer_);
 
    // 创建功率相关缓冲区
    size_t power_weights_size = max_lights * sizeof(float);
  
    core_->CreateBuffer(power_weights_size, grassland::graphics::BUFFER_TYPE_DYNAMIC, &power_weights_buffer_);

    buffers_initialized_ = true;
    grassland::LogInfo("Light manager initialized with max {} lights", max_lights);
}

void LightManager::AddLight(const Light& light) {
    lights_.push_back(light);
    
    if (light.visible && (light.type == 1 || light.type == 3) && scene_ != nullptr) {
        std::shared_ptr<Entity> light_entity = CreateLightEntity(light, lights_.size() - 1);
        if (light_entity) {
            light_entities_.push_back(light_entity);
            scene_->AddEntity(light_entity);
            grassland::LogInfo("Created visible light entity for light index {}", lights_.size() - 1);
        }
    }
    
    UpdateBuffers();
    grassland::LogInfo("Added light (total: {})", lights_.size());
}

void LightManager::RemoveLight(size_t index) { // unnecessary
}

void LightManager::UpdateLight(size_t index, const Light& light) { // unnecessary
    if (index < lights_.size()) {
        lights_[index] = light;
        UpdateBuffers();
        grassland::LogInfo("Updated light at index {}", index);
    } else {
        grassland::LogWarning("Cannot update light at invalid index: {}", index);
    }
}

void LightManager::UpdatePowerData() {
    if (!buffers_initialized_ || !core_) return;
    
    // Reset
    power_weights_.clear();
    total_power_ = hdr_power_;
    float max_power = hdr_power_;
    
    // Power calculation
    std::vector<float> powers;
    powers.reserve(lights_.size() + 1);
    
    for (size_t i = 0; i < lights_.size(); ++i) {
        float power = CalculateLightPowerCPU(lights_[i]);
        powers.push_back(power);
        total_power_ += power;
        max_power = std::max(max_power, hdr_power_);
    }

    float low = max_power / (lights_.size() + 1) / 16;

    total_power_ = hdr_power_;

    for(size_t i = 0; i < lights_.size(); ++i) {
        if(powers[i] < low){
            grassland::LogWarning("Increase power light {} from {} to {} because power is too low", i, powers[i], low);
            powers[i] = low;
        }
        total_power_ += powers[i];
    }

    powers.push_back(hdr_power_);    
    
    // Calculate normalized weights
    if (total_power_ > 1e-6) {
        for (size_t i = 0; i < powers.size(); ++i) {
            float weight = powers[i] / total_power_;
            power_weights_.push_back(weight);
        }
        grassland::LogInfo("Total power is {}", total_power_);    
    } else {
        power_weights_.resize(powers.size(), 0.0f);
        grassland::LogWarning("Total power too low !");        
    }
    
    // Upload to GPU
    if (!power_weights_.empty()) {
        power_weights_buffer_->UploadData(power_weights_.data(), 
                                         power_weights_.size() * sizeof(float));
    }
    grassland::LogInfo("power buffer updated with {} entries", power_weights_.size());
}

void LightManager::UpdateBuffers() {
    if (!buffers_initialized_ || !core_) {
        return;
    }
    
    size_t data_size = lights_.size() * sizeof(Light);
    lights_buffer_->UploadData(lights_.data(), data_size);
    
    grassland::LogInfo("light_buffer updated");
    
    UpdatePowerData();
}


std::shared_ptr<Entity> LightManager::CreateLightEntity(const Light& light, size_t light_index) {
    if (!scene_ || !core_) {
        return nullptr;
    }
    
    std::string mesh_path;
    glm::mat4 transform = glm::mat4(1.0f);
    Material material;
    material.light_index = light_index;
    
    if (light.type == 1) { // surface light
        mesh_path = "meshes/square_light.obj";
        
        transform = CalculateAreaLightTransform(light);
    } 
    else if (light.type == 3) { // sphere light
        mesh_path = "meshes/ball.obj";

        float scale = light.radius;
        transform = glm::scale(
            glm::translate(glm::mat4(1.0f), light.position),
            glm::vec3(scale)
        );
    }
    else {
        return nullptr;
    }
    
    try {
        auto entity = std::make_shared<Entity>(mesh_path, material, transform);
        if (entity->IsValid()) {
            return entity;
        } else {
            grassland::LogWarning("Failed to create light entity for light index {}", light_index);
            return nullptr;
        }
    } catch (const std::exception& e) {
        grassland::LogError("Exception while creating light entity: {}", e.what());
        return nullptr;
    }
}

glm::mat4 LightManager::CalculateAreaLightTransform(const Light& light) const {
    if (light.type != 1) {
        return glm::mat4(1.0f);
    }
    
    glm::vec3 bitangent = glm::normalize(glm::cross(light.direction, light.tangent));
    
    // rotation matrix
    glm::mat3 rotation;
    rotation[0] = light.tangent;     // x
    rotation[1] = bitangent;         // y  
    rotation[2] = light.direction;   // z
    
    // transform matrix
    glm::mat4 transform = glm::mat4(1.0f);
    transform = glm::translate(transform, light.position);
    transform = transform * glm::mat4(rotation);
    transform = glm::scale(transform, glm::vec3(light.size.x, light.size.y, 1.0f));
    
    return transform;
}

float CalculateLightPowerCPU(const Light& light) {
    if (!light.enabled) return 0.0f;
    
    glm::vec3 linear_color = light.color * light.intensity;
    float luminance = glm::dot(linear_color, glm::vec3(0.2126f, 0.7152f, 0.0722f));
    const float PI = 3.14159265359f;
    
    switch (light.type) {
        case 0: // point light
            return luminance * 4.0f * PI;
        case 1: // area light
            return luminance * light.size.x * light.size.y * PI;
        case 2: // spot light
            {
                float cos_half_angle = cos(glm::radians(light.cone_angle * 0.5f));
                float solid_angle = 2.0f * PI * (1.0f - cos_half_angle);
                return luminance * solid_angle;
            }
        case 3: // sphere light
            {
                float surface_area = 4.0f * PI * light.radius * light.radius;
                return luminance * surface_area * PI;
            }
        default:
            return luminance;
    }
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


void LightManager::CreateDefaultLights() {
    if (!core_) {
        grassland::LogError("Light manager not initialized before creating default lights");
        return;
    }
    
    lights_.clear();
    light_entities_.clear();
        
    // Light visible_area_light = Light::CreateAreaLight(
    //     glm::vec3(0.0f, 10.0f, 0.0f),      
    //     glm::vec3(0.0f, -1.0f, 0.0f),     
    //     glm::vec3(1.0f, 0.0f, 0.0f),      
    //     glm::vec2(5.0f, 5.0f),            
    //     glm::vec3(1.0f, 0.8f, 0.6f),      
    //     1.5f,                             
    //     true                              
    // );
    // lights_.push_back(visible_area_light);

    // Light light_2 = Light::CreateAreaLight(
    //     glm::vec3(-8.0f, 6.0f, 10.0f),    
    //     glm::vec3(0.6f, -0.3, -1.0f),     
    //     glm::vec3(1.0f, 0.0f, 0.0f),      
    //     glm::vec2(5.0f, 5.0f),            
    //     glm::vec3(0.9f, 0.8f, 0.6f),      
    //     5.5f,                             
    //     true                              
    // );
    // lights_.push_back(light_2);    

    // Light visible_sphere_light = Light::CreateSphereLight(
    //     glm::vec3(50.0f, 10.0f, 50.0f),    
    //     0.25f,                             
    //     glm::vec3(1.0f, 0.8f, 0.6f),      
    //     50000.0f,                             
    //     true                              
    // );
    // lights_.push_back(visible_sphere_light);
    
    // Light invisible_point = Light::CreatePointLight(
    //     // glm::vec3(1.5f, 2.0f, -3.0f),     
    //     glm::vec3(50.0f, 15.0f, 50.0f),     
    //     glm::vec3(0.8f, 0.9f, 1.0f),    
    //     9000.0f,                            
    //     false                            
    // );
    // lights_.push_back(invisible_point);
    
    // Light invisible_spot = Light::CreateSpotLight(
    //     // glm::vec3(5.0f, 5.0f, 3.0f),      
    //     // glm::vec3(-0.5f, -1.0f, -0.5f),    
    //     glm::vec3(2.0f, 2.00f, -4.0f),
    //     glm::vec3(0.0f, -0.065f, 1.0f),
    //     glm::vec3(1.0f, 0.95f, 0.9f),    
    //     150.0f,                             
    //     8.0f,                            
    //     false                             
    // );
    // lights_.push_back(invisible_spot);
    
    UpdateBuffers();
    
    // visible light entities
    for (size_t i = 0; i < lights_.size(); ++i) {
        const auto& light = lights_[i];
        if (light.visible && (light.type == 1 || light.type == 3) && scene_ != nullptr) {
            std::shared_ptr<Entity> light_entity = CreateLightEntity(light, i);
            if (light_entity) {
                light_entities_.push_back(light_entity);
                scene_->AddEntity(light_entity);
            }
        }
    }
    
    grassland::LogInfo("Created {} default lights ({} visible entities)", 
                      lights_.size(), light_entities_.size());
}
