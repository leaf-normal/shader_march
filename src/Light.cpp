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
    
    // 如果光源可见且是面光源或球光源，创建对应的Entity
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

void LightManager::RemoveLight(size_t index) { // unnecessary, keep empty
}

void LightManager::UpdateLight(size_t index, const Light& light) { // unnecessary, no need for modification
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
    
    // 重置数据
    power_weights_.clear();
    total_power_ = 0.0f;
    
    // 计算每个光源的功率
    std::vector<float> powers;
    powers.reserve(lights_.size());
    
    for (size_t i = 0; i < lights_.size(); ++i) {
        float power = CalculateLightPowerCPU(lights_[i]);
        powers.push_back(power);
        total_power_ += power;
    }
    
    // 计算归一化的功率权重
    if (total_power_ > 1e-6) {
        for (size_t i = 0; i < lights_.size(); ++i) {
            float weight = powers[i] / total_power_;
            power_weights_.push_back(weight);
        }
    } else {
        power_weights_.resize(lights_.size(), 0.0f);
        grassland::LogWarning("Total power too low !");        
    }
    
    // 上传功率权重数据到GPU
    if (!power_weights_.empty()) {
        power_weights_buffer_->UploadData(power_weights_.data(), 
                                         power_weights_.size() * sizeof(float));
    }
    grassland::LogInfo("power buffer updated");
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
    
    if (light.type == 1) { // 面光源
        mesh_path = "meshes/square_light.obj"; // 边长为1的正方形，法向指向z
        
        // 计算面光源的变换矩阵
        transform = CalculateAreaLightTransform(light);
    } 
    else if (light.type == 3) { // 球光源
        mesh_path = "meshes/icosahedron_light.obj"; // 半径为0.5的正二十面体
        // 移动到光源位置并缩放到正确半径
        float scale = light.radius * 2.0f; // 因为原始模型半径为0.5，所以要×2
        transform = glm::scale(
            glm::translate(glm::mat4(1.0f), light.position),
            glm::vec3(scale)
        );
    }
    else {
        // 其他类型的光源不需要实体
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
    
    // 构建旋转矩阵
    glm::mat3 rotation;
    rotation[0] = light.tangent;     // X轴
    rotation[1] = bitangent;         // Y轴  
    rotation[2] = light.direction;   // Z轴（法线）
    
    // 构建完整的变换矩阵
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
        case 0: // 点光源
            return luminance * 4.0f * PI;
        case 1: // 面光源
            return luminance * light.size.x * light.size.y * PI;
        case 2: // 聚光灯
            {
                float cos_half_angle = cos(glm::radians(light.cone_angle * 0.5f));
                float solid_angle = 2.0f * PI * (1.0f - cos_half_angle);
                return luminance * solid_angle;
            }
        case 3: // 球光源
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
    
    //创建一个可见的面光源作为示例
    Light visible_area_light = Light::CreateAreaLight(
        glm::vec3(0.0f, 3.0f, 0.0f),      // 位置
        glm::vec3(0.0f, -1.0f, 0.0f),     // 法线方向（向下）
        glm::vec3(1.0f, 0.0f, 0.0f),      // 切向量（指向X轴）
        glm::vec2(1.5f, 1.5f),            // 尺寸
        glm::vec3(1.0f, 1.0f, 0.9f),      // 暖白色
        2.0f,                             // 强度
        true                              // 可见
    );
    lights_.push_back(visible_area_light);
    
    //创建一个可见的球光源
    Light visible_sphere_light = Light::CreateSphereLight(
        glm::vec3(-3.4f, 1.3f, 0.0f),     // 位置
        0.15f,                             // 半径
        glm::vec3(1.0f, 0.8f, 0.6f),      // 暖黄色
        13.0f,                             // 强度
        true                              // 可见
    );
    lights_.push_back(visible_sphere_light);
    
    // 不可见的点光源
    // Light invisible_point = Light::CreatePointLight(
    //     // glm::vec3(1.5f, 2.0f, -3.0f),      // 位置
    //     glm::vec3(0.0f, 2.7f, 0.7f),      // 位置
    //     glm::vec3(0.8f, 0.9f, 1.0f),      // 冷蓝色
    //     1.6f,                             // 强度
    //     false                             // 不可见
    // );
    // lights_.push_back(invisible_point);
    
    // 不可见的聚光灯
    Light invisible_spot = Light::CreateSpotLight(
        glm::vec3(5.0f, 5.0f, 3.0f),      // 位置
        glm::vec3(-0.5f, -1.0f, -0.5f),    // 方向
        glm::vec3(1.0f, 0.95f, 0.9f),     // 暖白色
        10.0f,                             // 强度
        30.0f,                            // 锥角
        false                             // 不可见
    );
    lights_.push_back(invisible_spot);
    
    // 更新缓冲区
    UpdateBuffers();
    
    // 创建可见光源的实体
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

// void LightManager::CreateDefaultLights() {
//     if (!core_) {
//         grassland::LogError("Light manager not initialized before creating default lights");
//         return;
//     }
    
//     lights_.clear();
    
//     Light main_directional = Light::CreateDirectionalLight(
//         glm::vec3(5.0f, 5.0f, 2.5f),      // 位置（参考点）
//         glm::vec3(-0.5f, -1.0f, -0.5f),   // 方向
//         glm::vec3(1.0f, 0.95f, 0.9f),     // 暖白色
//         3.0f,                             // 强度
//         45.0f                             // 锥角
//     );
//     lights_.push_back(main_directional);
    
//     Light red_point = Light::CreatePointLight(
//         glm::vec3(-5.0f, 2.0f, 1.0f),     // 位置
//         glm::vec3(1.0f, 0.3f, 0.3f),      // 红色
//         1.2f                              // 强度
//     );
//     lights_.push_back(red_point);
    
//     // Light blue_point = Light::CreatePointLight(
//     //     glm::vec3(1.5f, 3.0f, -1.0f),     // 位置
//     //     glm::vec3(0.3f, 0.3f, 1.0f),      // 蓝色
//     //     1.2f                              // 强度
//     // );
//     // lights_.push_back(blue_point);
    
//     Light top_area = Light::CreateAreaLight(
//         glm::vec3(0.0f, 5.0f, 0.0f),      // 位置
//         glm::vec3(0.0f, -1.0f, 0.0f),     // 方向（向下）
//         glm::vec2(1.0f, 1.0f),            // 尺寸
//         glm::vec3(0.9f, 0.9f, 1.0f),      // 冷白色
//         1.0f                              // 强度
//     );
//     lights_.push_back(top_area);
    
//     // 更新缓冲区
//     UpdateBuffers();
//     grassland::LogInfo("Created {} default lights", lights_.size());
// }
