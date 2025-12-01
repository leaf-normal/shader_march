#pragma once
#include "long_march.h"
#include <memory>
#include <vector>

struct Light {
    glm::vec3 position;       // 光源位置（点光源和面光源需要，方向光忽略）
    glm::vec3 direction;      // 方向向量：
                             // - 点光源：不使用（置0）
                             // - 面光源：法线方向
                             // - 方向光：光照方向
    glm::vec3 color;          // 光源颜色
    float intensity;          // 光源强度
    glm::vec2 size;           // 面光源尺寸(矩形-长宽)
    int type;                 // 0=点光源, 1=面光源, 2=方向光
    bool enabled;             // 光源是否启用
    float cone_angle;         // 方向光锥角（角度，0=无锥角限制）

    Light():position(0.0f),direction(0.0f,-1.0f,0.0f), 
            color(1.0f),intensity(1.0f),size(1.0f,1.0f), 
            type(0),enabled(true),cone_angle(0.0f){}

    // 创建点光源
    static Light CreatePointLight(const glm::vec3& pos,const glm::vec3& col=glm::vec3(1.0f),float intens=1.0f)
    {
        Light light;
        light.position=pos;
        light.direction=glm::vec3(0.0f,0.0f,0.0f);
        light.color=col;
        light.intensity=intens;
        light.type=0;
        light.enabled=true;
        return light;
    }
    
    // 创建面光源
    static Light CreateAreaLight(const glm::vec3& pos,const glm::vec3& norm,const glm::vec2& sz,
                                 const glm::vec3& col=glm::vec3(1.0f),float intens=1.0f)
    {
        Light light;
        light.position=pos;
        light.direction=glm::normalize(norm);
        light.color=col;
        light.intensity=intens;
        light.size=sz;
        light.type=1;
        light.enabled=true;
        return light;
    }
    
    // 创建聚光方向光
    static Light CreateDirectionalLight(const glm::vec3& pos,const glm::vec3& dir,
                                        const glm::vec3& col=glm::vec3(1.0f),float intens=1.0f,float angle=45.0f)
    {
        Light light;
        light.position=pos;
        light.direction=glm::normalize(dir);
        light.color=col;
        light.intensity=intens;
        light.type=2; 
        light.enabled=true;
        light.cone_angle=angle;
        return light;
    }
};

class LightManager {
public:
    LightManager();
    ~LightManager();
    // 初始化光源管理器
    void Initialize(grassland::graphics::Core* core);
    // 添加光源
    void AddLight(const Light& light);
    // 移除光源
    void RemoveLight(size_t index);
    // 获取光源数量
    size_t GetLightCount() const{return lights_.size();}
    // 获取光源列表
    const std::vector<Light>& GetLights() const {return lights_;}
    std::vector<Light>& GetLights() {return lights_;}
    // 更新光源
    void UpdateLight(size_t index, const Light& light);
    // 获取光源缓冲区
    grassland::graphics::Buffer* GetLightsBuffer() const { return lights_buffer_.get(); }
    // 获取光源数量缓冲区
    grassland::graphics::Buffer* GetLightCountBuffer() const { return light_count_buffer_.get(); }
    // 更新GPU缓冲区
    void UpdateBuffers();
    // 创建默认光源场景
    void CreateDefaultLights();
    // 获取启用的光源数量
    int GetEnabledLightCount() const;
    // 查找主方向光（用于阴影计算等）
    const Light* FindMainDirectionalLight() const;
    
private:
    std::vector<Light> lights_;
    std::unique_ptr<grassland::graphics::Buffer> lights_buffer_;
    std::unique_ptr<grassland::graphics::Buffer> light_count_buffer_;
    grassland::graphics::Core* core_;
    bool buffers_initialized_;
};