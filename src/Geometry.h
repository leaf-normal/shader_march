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
    glm::vec2 texcoord;

    VertexInfo() = default;

    VertexInfo(const glm::vec3& pos, const glm::vec3& n, const glm::vec2& t)
        : position(pos)
        , normal(n)
        , texcoord(t){}
};
