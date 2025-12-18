#pragma once
#include "long_march.h"
#include "glm/gtc/matrix_transform.hpp"

struct MotionParams {
    glm::vec3 linear_velocity;      // 线性速度 (m/s)
    glm::vec3 angular_velocity;     // 角速度 (弧度/s) 表示绕哪个轴旋转
    glm::vec3 pivot_point;          // 旋转中心点（世界坐标）
    int is_static;                 // 是否静态
    int group_id;                   // 运动组ID
    
    MotionParams() 
        : linear_velocity(0.0f)
        , angular_velocity(0.0f)
        , pivot_point(0.0f)
        , is_static(true)
        , group_id(0) {}
    
    MotionParams(const glm::vec3& linear_vel, const glm::vec3& angular_vel, 
                 const glm::vec3& pivot = glm::vec3(0.0f), int group = 0)
        : linear_velocity(linear_vel)
        , angular_velocity(angular_vel)
        , pivot_point(pivot)
        , is_static(linear_vel == glm::vec3(0.0f) && angular_vel == glm::vec3(0.0f))
        , group_id(group) {}
    
    // 在时间t的变换矩阵
    glm::mat4 GetTransformAtTime(float t, const glm::mat4& base_transform) const {
        if (is_static) return base_transform;
        
        glm::mat4 transform = base_transform;
        
        // 应用线性位移
        glm::vec3 translation = linear_velocity * t;
        transform = glm::translate(glm::mat4(1.0f), translation) * transform;
        
        // 应用旋转
        if (glm::length(angular_velocity) > 0.0f) {
            float angle = glm::length(angular_velocity) * t;
            glm::vec3 axis = glm::normalize(angular_velocity);
            
            transform = glm::translate(glm::mat4(1.0f), pivot_point) *
                        glm::rotate(glm::mat4(1.0f), angle, axis) *
                        glm::translate(glm::mat4(1.0f), -pivot_point) *
                        transform;
        }
        
        return transform;
    }
};