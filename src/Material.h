#pragma once
#include "long_march.h"

struct Material {
    glm::vec3 base_color;
    float roughness;
    float metallic;
    unsigned int light_index;

    glm::vec3 emission;        // 自发光颜色
    float ior;              // 折射率
    float transparency;     // 透明度
    int texture_id;       
  
    float subsurface;       // 次表面散射
    float specular;         // 镜面反射强度
    float specular_tint;    // 镜面反射
    float anisotropic;      // 各向异性[-1,1]
    float sheen;            // 光泽层
    float sheen_tint;       // 光泽层染色
    float clearcoat;        // 清漆层强度
    float clearcoat_roughness; // 清漆层粗糙度
    float specular_transmission; // 镜面透射

    Material() : 
        base_color(0.8f, 0.8f, 0.8f), 
        roughness(0.5f), 
        metallic(0.0f), 
        light_index(0xFFFFFFFF),
        emission(glm::vec3(0.0f, 0.0f, 0.0f)),
        ior(1.0f),
        transparency(0.0f),
        texture_id(-1),  // -1 表示没有纹理
        subsurface(0.0f),
        specular(0.0f),
        specular_tint(0.0f),
        anisotropic(0.0f),
        sheen(0.0f),
        sheen_tint(0.0f),
        clearcoat(0.0f),
        clearcoat_roughness(0.0f),
        specular_transmission(0.0f) {}
    
    
    Material(const glm::vec3& color, float rough = 0.5f, float metal = 0.0f, 
             unsigned int index = 0xFFFFFFFF, const glm::vec3& emit = glm::vec3(0.0f, 0.0f, 0.0f),
             float refractive_index = 1.0f, float trans = 0.0f, int tex_id = -1,
             float sub = 0.0f, float spec = 0.0f, float spec_tint = 0.0f,
             float aniso = 0.0f, float sh = 0.0f, float sh_tint = 0.0f,
             float coat = 0.0f, float coat_rough = 0.0f, float spec_trans = 0.0f) : 
        base_color(color), 
        roughness(rough), 
        metallic(metal), 
        light_index(index),
        emission(emit),
        ior(refractive_index),
        transparency(trans),
        texture_id(tex_id),
        subsurface(sub),
        specular(spec),
        specular_tint(spec_tint),
        anisotropic(aniso),
        sheen(sh),
        sheen_tint(sh_tint),
        clearcoat(coat),
        clearcoat_roughness(coat_rough),
        specular_transmission(spec_trans) {}
};