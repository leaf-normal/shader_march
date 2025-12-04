#pragma once
#include "long_march.h"

// Simple material structure for ray tracing
struct Material {
    glm::vec3 base_color;
    float roughness;
    float metallic;
    unsigned int light_index;

    Material() : 
        base_color(0.8f, 0.8f, 0.8f), 
        roughness(0.5f), 
        metallic(0.0f), 
        light_index(0xFFFFFFFF) {}
    
    
    Material(const glm::vec3& color, float rough = 0.5f, float metal = 0.0f, unsigned int index = 0xFFFFFFFF): 
        base_color(color), 
        roughness(rough), 
        metallic(metal), 
        light_index(index) {}
};

