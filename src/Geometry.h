#pragma once
#include "long_march.h"

struct GeometryDescriptor {
    uint32_t vertex_offset;
    uint32_t index_offset;
    uint32_t vertex_count;
    uint32_t index_count;
};

struct VertexInfo {
    glm::vec3 position;
    glm::vec3 normal;

    VertexInfo() = default;

    VertexInfo(const glm::vec3& pos, const glm::vec3& n)
        : position(pos)
        , normal(n) {}
};