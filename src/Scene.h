#pragma once
#include "long_march.h"
#include "Entity.h"
#include "Material.h"
#include "Geometry.h"
#include "Motion.h"
#include <vector>
#include <memory>

#define MAX_MOTION_GROUPS 4

// Scene manages a collection of entities and builds the TLAS

struct MotionGroup {
    MotionParams motion;                    // 该组的运动参数
    std::vector<std::shared_ptr<Entity>> entities;  // 属于该组的实体
    std::vector<int> entity_id_;  // 运动组ID到实体ID的映射
    std::vector<grassland::graphics::RayTracingInstance> instances;  // 实例数据
};

class Scene {
public:
    Scene(grassland::graphics::Core* core);
    ~Scene();

    // Add an entity to the scene with motion parameters
    void AddEntity(std::shared_ptr<Entity> entity);

    // Remove all entities
    void Clear();

    // Build/rebuild all TLAS structures (including motion groups)
    void BuildAccelerationStructures();

    // Update TLAS instances for animation at time t (0-1 range)
    // void UpdateInstancesAtTime(float t);

    // start of modification

    grassland::graphics::AccelerationStructure* GetTLAS(int group_id = 0) const {
        if (group_id < 0 || group_id >= tlas_array_.size()) {
            return empty_tlas_.get();
        }
        return tlas_array_[group_id] ? tlas_array_[group_id].get() : empty_tlas_.get();
    }

    // end of modification

    // Get all motion group TLAS
    const std::vector<MotionGroup>& GetMotionGroups() const { return motion_groups_; }
    
    // Get number of motion groups
    size_t GetMotionGroupCount() const { return motion_groups_.size(); }

    // Get motion group parameters buffer (for shader)
    grassland::graphics::Buffer* GetMotionGroupsBuffer() const { return motion_groups_buffer_.get(); }

    // Get materials buffer for all entities
    grassland::graphics::Buffer* GetMaterialsBuffer() const { return materials_buffer_.get(); }

    // Get all entities
    const std::vector<std::shared_ptr<Entity>>& GetEntities() const { return entities_; }

    // Get number of entities
    size_t GetEntityCount() const { return entities_.size(); }

    grassland::graphics::Buffer* GetVertexInfoBuffer() const { return global_vertex_info_buffer_.get(); }
    grassland::graphics::Buffer* GetIndexBuffer() const { return global_index_buffer_.get(); }
    grassland::graphics::Buffer* GetGeometryDescriptorsBuffer() const { return geometry_descriptors_buffer_.get(); }

private:
    void UpdateMaterialsBuffer();
    void BuildGeometryBuffers();
    void BuildMotionGroupsBuffer();  // 构建运动组
    
    // 计算实体应该属于哪个运动组
    int CalculateMotionGroup(const MotionParams& motion);
    
    // 为运动组构建TLAS
    void BuildGroupTLAS(int group_id);
    
    grassland::graphics::Core* core_;
    std::vector<std::shared_ptr<Entity>> entities_;
    
    // 运动组管理
    std::vector<MotionGroup> motion_groups_;

    std::unique_ptr<grassland::graphics::Buffer> motion_groups_buffer_;
    std::vector<std::unique_ptr<grassland::graphics::AccelerationStructure>> tlas_array_; // 使用 unique_ptr 来管理所有权
    std::unique_ptr<grassland::graphics::AccelerationStructure> empty_tlas_;

    std::unique_ptr<grassland::graphics::Buffer> materials_buffer_;
    std::unique_ptr<grassland::graphics::Buffer> geometry_descriptors_buffer_;
    std::unique_ptr<grassland::graphics::Buffer> global_vertex_info_buffer_; 
    std::unique_ptr<grassland::graphics::Buffer> global_index_buffer_;
};