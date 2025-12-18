#include "Scene.h"
#include "Geometry.h"
#include "Motion.h"
#include "glm/gtc/matrix_transform.hpp"

Scene::Scene(grassland::graphics::Core* core)
    : core_(core) {
    // 初始化运动组（至少有一个静态组）
    motion_groups_.resize(1);
    motion_groups_[0].motion = MotionParams();  // 静态组
    std::vector<grassland::graphics::RayTracingInstance> empty_instance;
    core_->CreateTopLevelAccelerationStructure(empty_instance, &empty_tlas_);
}

Scene::~Scene() {
    Clear();
}
void Scene::AddEntity(std::shared_ptr<Entity> entity) {
    if (!entity || !entity->IsValid()) {
        grassland::LogError("Cannot add invalid entity to scene");
        return;
    }    
    // 计算应该属于哪个运动组
    int group_id = CalculateMotionGroup(entity->GetMotionParams());
    entity->SetMotionGroups(group_id);

    // 构建BLAS
    entity->BuildBLAS(core_);
    
    // 添加到实体列表
    entities_.push_back(entity);

    // 添加到对应的运动组
    if (group_id >= motion_groups_.size()) {
        motion_groups_.resize(group_id + 1);
    }
    motion_groups_[group_id].entities.push_back(entity);
    motion_groups_[group_id].entity_id_.push_back(entities_.size() - 1);
    
    grassland::LogInfo("Added entity to scene (total: {}, group: {})", 
                      entities_.size(), group_id);
}

void Scene::Clear() {
    entities_.clear();
    motion_groups_.clear();
    tlas_array_.clear();
    motion_groups_buffer_.reset();
    materials_buffer_.reset();
    geometry_descriptors_buffer_.reset();
    global_vertex_info_buffer_.reset();
    global_index_buffer_.reset();
}

int Scene::CalculateMotionGroup(const MotionParams& motion) {
    if (motion.is_static) {
        return 0;  // 静态物体归到第0组
    }
    
    // 查找是否有相同运动参数的组
    for (int i = 1; i < motion_groups_.size(); ++i) {
        const MotionParams& group_motion = motion_groups_[i].motion;
        if (group_motion.linear_velocity == motion.linear_velocity &&
            group_motion.angular_velocity == motion.angular_velocity &&
            group_motion.pivot_point == motion.pivot_point) {
            return i;
        }
    }
    
    // 没有找到，创建新组
    int new_group_id = motion_groups_.size();
    if (new_group_id >= MAX_MOTION_GROUPS) {
        grassland::LogWarning("Too many motion groups, falling back to group 0");
        return 0;
    }
    
    MotionGroup new_group;
    new_group.motion = motion;
    new_group.motion.group_id = new_group_id;
    motion_groups_.push_back(new_group);
    
    return new_group_id;
}

void Scene::BuildAccelerationStructures() {
    if (entities_.empty()) {
        grassland::LogWarning("No entities to build acceleration structures");
        return;
    }
    tlas_array_.clear();
    tlas_array_.resize(MAX_MOTION_GROUPS);

    // 为每个运动组构建TLAS
    for (int i = 0; i < motion_groups_.size(); ++i) {
        BuildGroupTLAS(i);
    }

    // 更新材质缓冲区
    UpdateMaterialsBuffer();

    // 构建几何缓冲区
    BuildGeometryBuffers();
    
    // 构建运动组参数缓冲区
    BuildMotionGroupsBuffer();
    
    grassland::LogInfo("Built {} motion groups with {} TLAS", 
                      motion_groups_.size(), tlas_array_.size());
}

void Scene::BuildGroupTLAS(int group_id) {
    if (group_id < 0 || group_id >= motion_groups_.size()) {
        return;
    }
    
    MotionGroup& group = motion_groups_[group_id];
    
    if (tlas_array_.size() <= group_id) {
        tlas_array_.resize(group_id + 1);
    }

    std::vector<grassland::graphics::RayTracingInstance> instances;
    for (size_t i = 0; i < group.entities.size(); ++i) {
        auto& entity = group.entities[i];
        if (entity->GetBLAS()) {
            glm::mat4x3 transform_3x4 = glm::mat4x3(entity->GetTransform());
            
            auto instance = entity->GetBLAS()->MakeInstance(
                transform_3x4,
                static_cast<uint32_t>(group.entity_id_[i]),
                0xFF,
                0,
                grassland::graphics::RAYTRACING_INSTANCE_FLAG_NONE
            );
            instances.push_back(instance);
        }
    }
    
    if (!instances.empty()) {
        std::unique_ptr<grassland::graphics::AccelerationStructure> new_tlas;
        core_->CreateTopLevelAccelerationStructure(instances, &new_tlas);
        tlas_array_[group_id] = std::move(new_tlas);
        group.instances = std::move(instances);
        grassland::LogInfo("Successfully built TLAS for motion group {} with {} instances.", 
                            group_id, group.instances.size());

    } else {
        tlas_array_[group_id].reset();
        group.instances.clear();
        grassland::LogInfo("Motion group {} has no instances, its TLAS slot is now null.", group_id);
    }
}

//void Scene::UpdateInstances() {
//    if (!tlas_ || entities_.empty()) {
//        return;
//    }
//
//    grassland::LogError("UpdataInstances is not supported. Do not call this function!");
//    return;

    // // Recreate instances with updated transforms
    // std::vector<grassland::graphics::RayTracingInstance> instances;
    // instances.reserve(entities_.size());

    // for (size_t i = 0; i < entities_.size(); ++i) {
    //     auto& entity = entities_[i];
    //     if (entity->GetBLAS()) {
    //         // Convert mat4 to mat4x3
    //         glm::mat4x3 transform_3x4 = glm::mat4x3(entity->GetTransform());
            
    //         auto instance = entity->GetBLAS()->MakeInstance(
    //             transform_3x4,
    //             static_cast<uint32_t>(i),
    //             0xFF,
    //             0,
    //             grassland::graphics::RAYTRACING_INSTANCE_FLAG_NONE
    //         );
    //         instances.push_back(instance);
    //     }
    // }

    // // Update TLAS
    // tlas_->UpdateInstances(instances);
    
    // grassland::LogInfo("Updated TLAS with {} entities", entities_.size());    
//}

void Scene::UpdateMaterialsBuffer() {
    if (entities_.empty()) {
        return;
    }

    // Collect all materials
    std::vector<Material> materials;
    materials.reserve(entities_.size());

    for (const auto& entity : entities_) {
        materials.push_back(entity->GetMaterial());
    }

    size_t buffer_size = materials.size() * sizeof(Material);
    
    if (!materials_buffer_) {
        core_->CreateBuffer(buffer_size, 
                          grassland::graphics::BUFFER_TYPE_DYNAMIC, 
                          &materials_buffer_);
    }
    
    materials_buffer_->UploadData(materials.data(), buffer_size);
    grassland::LogInfo("Updated materials buffer with {} materials", materials.size());
}

// *add
// 几何信息 Buffer，静态

void Scene::BuildGeometryBuffers() {
    if (entities_.empty()) {
        return;
    }

    size_t total_vertices = 0;
    size_t total_indices = 0;

    for (const auto& entity : entities_) {
        total_vertices += entity->GetVertexCount();
        total_indices += entity->GetIndexCount();
    }

    // 2. 创建几何描述符数组
    std::vector<GeometryDescriptor> geometry_descriptors;
    geometry_descriptors.reserve(entities_.size());

    // 3. 创建合并的顶点/法线/索引数据
    std::vector<VertexInfo> all_vertex_infos;
    std::vector<uint32_t> all_indices;
    
    all_vertex_infos.reserve(total_vertices);
    all_indices.reserve(total_indices);

    uint32_t vertex_offset = 0;
    uint32_t index_offset = 0;

    for (size_t i = 0; i < entities_.size(); ++i) {
        const auto& entity = entities_[i];
        
        size_t vertex_count = entity->GetVertexCount();
        size_t index_count = entity->GetIndexCount();

        // *modified
        std::vector<VertexInfo> vertex_infos(vertex_count);
        std::vector<uint32_t> indices(index_count);

        // *modified
        entity->GetVertexInfoBuffer()->DownloadData(vertex_infos.data(), vertex_infos.size() * sizeof(VertexInfo));
        entity->GetIndexBuffer()->DownloadData(indices.data(), indices.size() * sizeof(uint32_t));

        all_vertex_infos.insert(all_vertex_infos.end(), vertex_infos.begin(), vertex_infos.end());
        all_indices.insert(all_indices.end(), indices.begin(), indices.end());

        // 创建几何描述符
        GeometryDescriptor desc{};
        desc.vertex_offset = vertex_offset;
        desc.index_offset = index_offset;
        desc.vertex_count = static_cast<uint32_t>(vertex_count);
        desc.index_count = static_cast<uint32_t>(index_count);
        
        geometry_descriptors.push_back(desc);

        vertex_offset += vertex_count;
        index_offset += index_count;
    }

    // 5. 创建GPU缓冲区
    size_t vertex_buffer_size = all_vertex_infos.size() * sizeof(VertexInfo);
    size_t index_buffer_size = all_indices.size() * sizeof(uint32_t);
    size_t descriptors_size = geometry_descriptors.size() * sizeof(GeometryDescriptor);

    // *modified
    core_->CreateBuffer(vertex_buffer_size, grassland::graphics::BUFFER_TYPE_DYNAMIC, &global_vertex_info_buffer_);
    core_->CreateBuffer(index_buffer_size, grassland::graphics::BUFFER_TYPE_DYNAMIC, &global_index_buffer_);
    core_->CreateBuffer(descriptors_size, grassland::graphics::BUFFER_TYPE_DYNAMIC, &geometry_descriptors_buffer_);

    // *modified
    global_vertex_info_buffer_->UploadData(all_vertex_infos.data(), vertex_buffer_size);
    global_index_buffer_->UploadData(all_indices.data(), index_buffer_size);
    geometry_descriptors_buffer_->UploadData(geometry_descriptors.data(), descriptors_size);

    grassland::LogInfo("Built geometry buffers: {} vertices, {} indices across {} entities",
        all_vertex_infos.size(),  all_indices.size(), entities_.size());
}

void Scene::BuildMotionGroupsBuffer() {
    // 构建运动组参数缓冲区
    std::vector<MotionParams> group_data;
    group_data.reserve(motion_groups_.size());
    
    for (const auto& group : motion_groups_) group_data.push_back(group.motion);
    
    size_t buffer_size = group_data.size() * sizeof(MotionParams);
    core_->CreateBuffer(buffer_size, 
                       grassland::graphics::BUFFER_TYPE_DYNAMIC, 
                       &motion_groups_buffer_);
    motion_groups_buffer_->UploadData(group_data.data(), buffer_size);
}
