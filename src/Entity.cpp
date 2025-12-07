#include "Entity.h"
#include "Geometry.h"

#define ENABLE_NORMAL_INTERPOLATION 1
#define CALCULATE_MISSING_NORMALS 0

Entity::Entity(const std::string& obj_file_path, 
               const Material& material,
               const glm::mat4& transform)
    : material_(material)
    , transform_(transform)
    , mesh_loaded_(false) {
    
    LoadMesh(obj_file_path);
}

Entity::~Entity() {
    blas_.reset();
    index_buffer_.reset();
    vertex_info_buffer_.reset();
}

bool Entity::LoadMesh(const std::string& obj_file_path) {
    // Try to load the OBJ file
    std::string full_path = grassland::FindAssetFile(obj_file_path);
    
    if (mesh_.LoadObjFile(full_path) != 0) {
        grassland::LogError("Failed to load mesh from: {}", obj_file_path);
        mesh_loaded_ = false;
        return false;
    }

    grassland::LogInfo("Successfully loaded mesh: {} ({} vertices, {} indices)", 
                       obj_file_path, mesh_.NumVertices(), mesh_.NumIndices());
    
    mesh_loaded_ = true;
    return true;
}

void Entity::BuildBLAS(grassland::graphics::Core* core) {
    if (!mesh_loaded_) {
        grassland::LogError("Cannot build BLAS: mesh not loaded");
        return;
    }

    // *add

    std::vector<VertexInfo> vertex_infos;
    vertex_infos.reserve(mesh_.NumVertices());

    const glm::vec3* positions = reinterpret_cast<const glm::vec3*>(mesh_.Positions());
    const glm::vec3* normals = reinterpret_cast<const glm::vec3*>(mesh_.Normals());

    if (ENABLE_NORMAL_INTERPOLATION && normals) {
        grassland::LogInfo("Use provided normals in the mesh.");
        for (size_t i = 0; i < mesh_.NumVertices(); ++i) {
            vertex_infos.emplace_back(positions[i], normals[i]);
        }
    } 
    else if(ENABLE_NORMAL_INTERPOLATION && CALCULATE_MISSING_NORMALS){
        grassland::LogInfo("Mesh has no normals. Calculating average normals...");

        std::vector<glm::vec3> accumulated_normals(mesh_.NumVertices(), glm::vec3(0.0f));
        
        const uint32_t* indices = mesh_.Indices();

        for (size_t i = 0; i < mesh_.NumIndices(); i += 3) {
            uint32_t i0 = indices[i];
            uint32_t i1 = indices[i + 1];
            uint32_t i2 = indices[i + 2];

            glm::vec3 v0 = positions[i0];
            glm::vec3 v1 = positions[i1];
            glm::vec3 v2 = positions[i2];

            glm::vec3 face_normal = glm::cross(v1 - v0, v2 - v0); // weighted average

            accumulated_normals[i0] += face_normal;
            accumulated_normals[i1] += face_normal;
            accumulated_normals[i2] += face_normal;
        }

        for (auto& normal : accumulated_normals) {
            if (glm::length(normal) > 1e-6f) {
                normal = glm::normalize(normal);
            } else {
                normal = glm::vec3(0.0f, 0.0f, 0.0f);
            }
        }

        for (size_t i = 0; i < mesh_.NumVertices(); ++i) {
            vertex_infos.emplace_back(positions[i], accumulated_normals[i]);
        }
        grassland::LogInfo("Calculated {} average normals for the mesh.", mesh_.NumVertices());

    } else {

        grassland::LogWarning("Mesh has no normals! Results may be incorrect");
        for (size_t i = 0; i < mesh_.NumVertices(); ++i) {
            vertex_infos.emplace_back(positions[i], glm::vec3(0.0f, 0.0f, 0.0f));
        }
    }


    size_t vertex_info_buffer_size = mesh_.NumVertices() * sizeof(VertexInfo);
    core->CreateBuffer(vertex_info_buffer_size, 
                      grassland::graphics::BUFFER_TYPE_DYNAMIC, 
                      &vertex_info_buffer_);
    vertex_info_buffer_->UploadData(vertex_infos.data(), vertex_info_buffer_size);

    size_t index_buffer_size = mesh_.NumIndices() * sizeof(uint32_t);
    core->CreateBuffer(index_buffer_size, 
                      grassland::graphics::BUFFER_TYPE_DYNAMIC, 
                      &index_buffer_);
    index_buffer_->UploadData(mesh_.Indices(), index_buffer_size);


    core->CreateBottomLevelAccelerationStructure(
        vertex_info_buffer_.get(),
        index_buffer_.get(), 
        sizeof(VertexInfo),
        &blas_);

    grassland::LogInfo("Built BLAS for entity");
}
