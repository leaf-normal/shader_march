#pragma once
#include "long_march.h"
#include "Entity.h"
#include "Material.h"
#include <vector>
#include <memory>

// Scene manages a collection of entities and builds the TLAS
class Scene {
public:
    Scene(grassland::graphics::Core* core);
    ~Scene();

    // Add an entity to the scene
    void AddEntity(std::shared_ptr<Entity> entity);

    // Remove all entities
    void Clear();

    // Build/rebuild the TLAS from all entities
    void BuildAccelerationStructures();

    // Update TLAS instances (e.g., for animation)
    void UpdateInstances();

    // Get the TLAS for rendering
    grassland::graphics::AccelerationStructure* GetTLAS() const { return tlas_.get(); }

    // Get materials buffer for all entities
    grassland::graphics::Buffer* GetMaterialsBuffer() const { return materials_buffer_.get(); }

    // Get all entities
    const std::vector<std::shared_ptr<Entity>>& GetEntities() const { return entities_; }

    // Get number of entities
    size_t GetEntityCount() const { return entities_.size(); }

private:
    void UpdateMaterialsBuffer();

    grassland::graphics::Core* core_;
    std::vector<std::shared_ptr<Entity>> entities_;
    std::unique_ptr<grassland::graphics::AccelerationStructure> tlas_;
    std::unique_ptr<grassland::graphics::Buffer> materials_buffer_;
};

