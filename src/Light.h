#pragma once
#include "long_march.h"
#include "Scene.h"  // 需要包含Scene，用于添加实体
#define MUL_INTENS 4

struct Light {
    glm::vec3 position;       // 光源位置
    glm::vec3 direction;      // 方向向量：
                             // - 点光源：不使用（置0）
                             // - 面光源：法线方向
                             // - 聚光灯：光照方向
    glm::vec3 tangent;        // 面光源的切向量（新增）
    glm::vec3 color;          // 光源颜色
    float intensity;          // 光源强度
    glm::vec2 size;           // 面光源尺寸(矩形-长宽)
    float radius;             // 球光源半径
    float cone_angle;         // 聚光灯锥角（角度，不能为 0）
    int type;                 // 0=点光源, 1=面光源, 2=聚光灯, 3=球光源
    int enabled;             // 光源是否启用，防止对齐问题，使用 int 而不是 bool
    int visible;             // 光源是否可见

    Light(): position(0.0f), direction(0.0f,-1.0f,0.0f), 
            tangent(1.0f,0.0f,0.0f),  // 默认切向量指向X轴
            color(1.0f), intensity(1.0f), size(1.0f,1.0f), 
            radius(0.5f), type(0), enabled(true), visible(false), cone_angle(0.0f){}

    // 创建点光源
    static Light CreatePointLight(const glm::vec3& pos, const glm::vec3& col=glm::vec3(1.0f),
                                  float intens=1.0f, bool visible=false)
    {
        Light light;
        light.position = pos;
        light.direction = glm::vec3(0.0f,0.0f,0.0f);
        light.tangent = glm::vec3(0.0f,0.0f,0.0f);
        light.color = col;
        light.intensity = intens * MUL_INTENS;
        light.type = 0;
        light.enabled = true;
        light.visible = visible;
        return light;
    }
    
    // 创建面光源 - 添加切向量参数
    static Light CreateAreaLight(const glm::vec3& pos, const glm::vec3& norm, 
                                 const glm::vec3& tangent, const glm::vec2& sz,
                                 const glm::vec3& col=glm::vec3(1.0f), float intens=1.0f,
                                 bool visible=false)
    {
        Light light;
        light.position = pos;
        light.direction = glm::normalize(norm);
        light.tangent = glm::normalize(tangent);
        
        // 确保切向量与法向量垂直
        if (std::abs(glm::dot(light.direction, light.tangent)) > 0.001f) {
            // 如果不垂直，重新计算切向量
            light.tangent = glm::normalize(light.tangent - glm::dot(light.tangent, light.direction) * light.direction);
        }
        
        light.color = col;
        light.intensity = intens * MUL_INTENS;
        light.size = sz;
        light.type = 1;
        light.enabled = true;
        light.visible = visible;
        return light;
    }
    
    // 创建聚光灯
    static Light CreateSpotLight(const glm::vec3& pos, const glm::vec3& dir,
                                        const glm::vec3& col=glm::vec3(1.0f), float intens=1.0f,
                                        float angle=45.0f, bool visible=false)
    {
        Light light;
        light.position = pos;
        light.direction = glm::normalize(dir);
        light.tangent = glm::vec3(0.0f,0.0f,0.0f);
        light.color = col;
        light.intensity = intens * MUL_INTENS;
        light.type = 2; 
        light.enabled = true;
        light.visible = visible;
        light.cone_angle = angle;
        return light;
    }

    // 创建球光源
    static Light CreateSphereLight(const glm::vec3& pos, float radius,
                                   const glm::vec3& col=glm::vec3(1.0f), float intens=1.0f,
                                   bool visible=false)
    {
        Light light;
        light.position = pos;
        light.radius = radius;
        light.direction = glm::vec3(0.0f,0.0f,0.0f);
        light.tangent = glm::vec3(0.0f,0.0f,0.0f);
        light.color = col;
        light.intensity = intens * MUL_INTENS;
        light.type = 3; // 球光源类型
        light.enabled = true;
        light.visible = visible;
        return light;
    }
};

class LightManager {
public:
    LightManager();
    ~LightManager();
    
    // 初始化光源管理器
    void Initialize(grassland::graphics::Core* core, Scene* scene = nullptr);
    
    // 添加光源 - 如果visible=true，会创建对应的Entity
    void AddLight(const Light& light);
    
    // 移除光源
    void RemoveLight(size_t index);
    
    // 获取光源数量
    size_t GetLightCount() const { return lights_.size(); }
    
    // 获取光源列表
    const std::vector<Light>& GetLights() const { return lights_; }
    std::vector<Light>& GetLights() { return lights_; }
    
    // 更新光源
    void UpdateLight(size_t index, const Light& light);
    
    // 获取光源缓冲区
    grassland::graphics::Buffer* GetLightsBuffer() const { return lights_buffer_.get(); }
    
    // 更新GPU缓冲区
    void UpdateBuffers();
    
    // 创建默认光源场景
    void CreateDefaultLights();
    
    // 获取启用的光源数量
    int GetEnabledLightCount() const;

    // 设置场景（如果初始化时未设置）
    void SetScene(Scene* scene) { scene_ = scene; }
    
    // 获取光源实体列表
    const std::vector<std::shared_ptr<Entity>>& GetLightEntities() const { return light_entities_; }

    static float CalculateLightPower(const Light& light);
    
    // 获取总功率
    float GetTotalPower() const { return total_power_; }
    
    // 获取归一化的功率权重数组
    const std::vector<float>& GetPowerWeights() const { return power_weights_; }
    grassland::graphics::Buffer* GetPowerWeightsBuffer() const { return power_weights_buffer_.get(); }

private:
    std::vector<Light> lights_;
    std::vector<std::shared_ptr<Entity>> light_entities_; // 存储可见光源的实体
    std::unique_ptr<grassland::graphics::Buffer> lights_buffer_;
    std::unique_ptr<grassland::graphics::Buffer> light_count_buffer_;
    grassland::graphics::Core* core_;
    Scene* scene_; // 指向场景，用于添加光源实体
    bool buffers_initialized_;

    std::vector<float> power_weights_;          // 归一化的功率权重
    float total_power_;                          // 总功率
    std::unique_ptr<grassland::graphics::Buffer> power_weights_buffer_;    // 功率权重缓冲区

    void UpdatePowerData();

    // 为可见光源创建Entity
    std::shared_ptr<Entity> CreateLightEntity(const Light& light, size_t light_index);
    
    // 创建自发光材质
    Material CreateEmissiveMaterial(const Light& light) const;
    
    // 计算面光源的变换矩阵
    glm::mat4 CalculateAreaLightTransform(const Light& light) const;
};