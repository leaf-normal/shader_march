#include "Scene.h"

Scene::Scene(grassland::graphics::Core* core)
    : core_(core) {
}

Scene::~Scene() {
    Clear();
}

void Scene::AddEntity(std::shared_ptr<Entity> entity) {
    if (!entity || !entity->IsValid()) {
        grassland::LogError("Cannot add invalid entity to scene");
        return;
    }

    // Build BLAS for the entity
    entity->BuildBLAS(core_);
    
    entities_.push_back(entity);
    grassland::LogInfo("Added entity to scene (total: {})", entities_.size());
}

void Scene::Clear() {
    entities_.clear();
    tlas_.reset();
    materials_buffer_.reset();
}

void Scene::BuildAccelerationStructures() {
    if (entities_.empty()) {
        grassland::LogWarning("No entities to build acceleration structures");
        return;
    }

    // Create TLAS instances from all entities
    std::vector<grassland::graphics::RayTracingInstance> instances;
    instances.reserve(entities_.size());

    for (size_t i = 0; i < entities_.size(); ++i) {
        auto& entity = entities_[i];
        if (entity->GetBLAS()) {
            // Create instance with entity's transform
            // instanceCustomIndex is used to index into materials buffer
            // Convert mat4 to mat4x3 (drop the last row which is always [0,0,0,1] for affine transforms)
            glm::mat4x3 transform_3x4 = glm::mat4x3(entity->GetTransform());
            
            auto instance = entity->GetBLAS()->MakeInstance(
                transform_3x4,
                static_cast<uint32_t>(i),  // instanceCustomIndex for material lookup
                0xFF,                       // instanceMask
                0,                          // instanceShaderBindingTableRecordOffset
                grassland::graphics::RAYTRACING_INSTANCE_FLAG_NONE
            );
            instances.push_back(instance);
        }
    }

    // Build TLAS
    core_->CreateTopLevelAccelerationStructure(instances, &tlas_);
    grassland::LogInfo("Built TLAS with {} instances", instances.size());

    // Update materials buffer
    UpdateMaterialsBuffer();
}

void Scene::UpdateInstances() {
    if (!tlas_ || entities_.empty()) {
        return;
    }

    // Recreate instances with updated transforms
    std::vector<grassland::graphics::RayTracingInstance> instances;
    instances.reserve(entities_.size());

    for (size_t i = 0; i < entities_.size(); ++i) {
        auto& entity = entities_[i];
        if (entity->GetBLAS()) {
            // Convert mat4 to mat4x3
            glm::mat4x3 transform_3x4 = glm::mat4x3(entity->GetTransform());
            
            auto instance = entity->GetBLAS()->MakeInstance(
                transform_3x4,
                static_cast<uint32_t>(i),
                0xFF,
                0,
                grassland::graphics::RAYTRACING_INSTANCE_FLAG_NONE
            );
            instances.push_back(instance);
        }
    }

    // Update TLAS
    tlas_->UpdateInstances(instances);
}

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

    // Create/update materials buffer
    size_t buffer_size = materials.size() * sizeof(Material);
    
    if (!materials_buffer_) {
        core_->CreateBuffer(buffer_size, 
                          grassland::graphics::BUFFER_TYPE_DYNAMIC, 
                          &materials_buffer_);
    }
    
    materials_buffer_->UploadData(materials.data(), buffer_size);
    grassland::LogInfo("Updated materials buffer with {} materials", materials.size());
}

