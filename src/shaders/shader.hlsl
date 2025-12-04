struct CameraInfo {
  float4x4 screen_to_camera;
  float4x4 camera_to_world;
};

struct Material {
  float3 base_color;
  float roughness;
  float metallic;
  uint light_index;
};

struct HoverInfo {
  int hovered_entity_id;
};

// ====================== 常量缓冲区定义 ======================
RaytracingAccelerationStructure as : register(t0, space0);
RWTexture2D<float4> output : register(u0, space1);
ConstantBuffer<CameraInfo> camera_info : register(b0, space2);
StructuredBuffer<Material> materials : register(t0, space3);
ConstantBuffer<HoverInfo> hover_info : register(b0, space4);
RWTexture2D<int> entity_id_output : register(u0, space5);
RWTexture2D<float4> accumulated_color : register(u0, space6);
RWTexture2D<int> accumulated_samples : register(u0, space7);

struct GeometryDescriptor {
    uint vertex_offset;
    uint index_offset;
    uint vertex_count;
    uint index_count;
};

struct VertexInfo {
    float3 position;
    float3 normal;    
};

StructuredBuffer<GeometryDescriptor> geometry_descriptors : register(t0, space8);
StructuredBuffer<VertexInfo> vertices : register(t0, space9);
StructuredBuffer<uint> indices : register(t0, space10);

struct RenderSettings {
    uint frame_count;
    uint samples_per_pixel;
    uint max_depth;
    uint enable_accumulation;
};

ConstantBuffer<RenderSettings> render_setting : register(b0, space11);

// ====================== 光源系统 ======================
#define MAX_LIGHTS 32

struct Light {
    float3 position;       // 光源位置
    float3 direction;      // 方向向量：
                             // - 点光源：不使用（置0）
                             // - 面光源：法线方向
                             // - 聚光灯：光照方向
    float3 tangent;        // 面光源的切向量，保证正交归一化
    float3 color;          // 光源颜色
    float intensity;          // 辐射亮度（面光源，球光源） / 辐射强度（点光源，聚光灯）
    float2 size;           // 面光源尺寸(矩形-长宽)
    float radius;             // 球光源半径
    float cone_angle; 
    uint type;                 // 0=点光源, 1=面光源, 2=聚光灯, 3=球光源
    uint enabled;             // 光源是否启用
    uint visible;             // 光源是否可见
};

StructuredBuffer<Light> lights : register(t0, space12);
StructuredBuffer<float> light_power_weights : register(t0, space13);

struct RayPayload {
  float3 color;
  bool hit;
  uint instance_id;
  float3 normal;
  float3 hit_point;
};

// ====================== 常量定义 ======================
#define MAX_DEPTH 16
#define RR_THRESHOLD 0.95f
#define t_min 0.001
#define t_max 10000.0
#define eps 5e-4
#define PI 3.14159265359
#define TWO_PI 6.28318530718
#define DECAY 0.2 // 光衰减系数，越大衰减越快

#define PHYSICAL_ATTENUATION 1  // 启用物理正确的衰减
#define INV_PI 0.31830988618
#define DECAY 0.2

// ====================== 随机数生成 ======================
uint pcg_hash(inout uint state) {
    state = state * 747796405u + 2891336453u;
    uint word = ((state >> ((state >> 28u) + 4u)) ^ state) * 277803737u;
    return (word >> 22u) ^ word;
}

float random(inout uint seed) {
    return (float)pcg_hash(seed) / 4294967296.0;
}

float2 random2(inout uint seed) {
    return float2(random(seed), random(seed));
}

// ====================== 几何辅助函数 ======================
float3 cosine_sample_hemisphere(inout uint seed, float3 normal) {
    float2 u = random2(seed);
    float phi = TWO_PI * u.x;
    float z = u.y;
    float r = sqrt(1.0 - z * z);
    
    float3 local_dir = float3(r * cos(phi), r * sin(phi), z);
    
    float3 tangent = normalize(cross(abs(normal.x) > 0.1 ? float3(0, 1, 0) : float3(1, 0, 0), normal));
    float3 bitangent = cross(normal, tangent);
    
    return local_dir.x * tangent + local_dir.y * bitangent + local_dir.z * normal;
}

uint generate_seed(uint2 pixel, uint frame) {
    return (pixel.x * 1973u + pixel.y * 9277u + frame * 26699u) ^ 0x6f5ca34du;
}

// ====================== 光源采样系统 ======================
struct LightSample {
    float3 position;
    float3 direction;
    float3 radiance;
    float pdf;
    int light_index;
    bool valid;
};

bool is_enable_lights(inout uint light_count){
  for(int i = 0; i < light_count; ++i){
    if(light_power_weights[i] > 0.0){
      return true;
    }
  }
  return false;
}

// 按功率重要性采样光源
int sample_light_by_power(inout uint light_count, inout uint seed) {
    
    // 使用预计算的权重进行采样
    float u = random(seed);
    uint last = 0;
    
    for (int i = 0; i < light_count; i++) {
        if(light_power_weights[i] > 0.0){
          u -= light_power_weights[i];
          if (u <= 0) {
              return i;
          }
          last = i;    
        }

    }
    
    // 浮点误差保护
    return last;
}

void sample_point_light(inout Light light, inout float3 hit_point, inout uint seed, out LightSample sample) {
    sample = (LightSample)0;
    
    if (!light.enabled) return;
    
    float3 to_light = light.position - hit_point;
    float distance = length(to_light);
    
    if (distance < 1e-6) return;
    
    sample.position = light.position;
    sample.direction = to_light / distance;
    
    // 物理正确的辐射亮度（朗伯体点光源）
    sample.radiance = light.color * light.intensity / (distance * distance + 1e-6);
    
    sample.pdf = 1;
    sample.light_index = -1; // 将由调用者设置
    sample.valid = true;
}

// 面光源采样 - 使用切向量
void sample_area_light(inout Light light, inout float3 hit_point, inout uint seed, out LightSample sample) {
    sample = (LightSample)0;
    
    if (!light.enabled || light.size.x <= 0.0 || light.size.y <= 0.0) return;
    
    // 构建局部坐标系
    float3 bitangent = normalize(cross(light.direction, light.tangent));

    // 在矩形上均匀采样（面积采样）
    float3 local_pos = float3((random(seed) - 0.5) * light.size.x, 
                              (random(seed) - 0.5) * light.size.y, 
                              0.0);
    
    // 转换到世界空间
    sample.position = light.position + 
                     local_pos.x * light.tangent + 
                     local_pos.y * bitangent;
    
    // 计算方向、距离、余弦项
    float3 to_light = sample.position - hit_point;
    float distance = length(to_light);
    sample.direction = to_light / distance;
    
    // 检查可见性（面光源只有一面发光）
    float cos_theta_l = dot(-sample.direction, light.direction);
    if (cos_theta_l <= 0.0) {
        return; // 从背面看不到
    }
    
    // 几何项
    float area = light.size.x * light.size.y;
    
    // 物理正确的辐射亮度（朗伯体面光源）
    sample.radiance = light.color * light.intensity;
    
    // PDF转换：面积PDF -> 立体角PDF
    float solid_angle_pdf = (distance * distance) / (cos_theta_l * area);
    sample.pdf = solid_angle_pdf;
    
    sample.light_index = -1;
    sample.valid = (sample.pdf > 0.0);
}

// 球光源采样 - 新增
void sample_sphere_light(inout Light light, inout float3 hit_point, inout uint seed, out LightSample sample) {
    sample = (LightSample)0;
    
    if (!light.enabled || light.radius <= 0.0) return;
    
    // 在球面上均匀采样
    float z = 2.0 * random(seed) - 1.0;
    float phi = TWO_PI * random(seed);
    float r = sqrt(max(0.0, 1.0 - z * z));
    
    float3 sphere_dir = float3(r * cos(phi), r * sin(phi), z);
    
    // 转换到世界空间
    sample.position = light.position + light.radius * sphere_dir;
    
    // 计算方向、距离、余弦项
    float3 to_light = sample.position - hit_point;
    float distance = length(to_light);
    sample.direction = to_light / distance;
    
    // 球面法线（指向外）
    float3 sphere_normal = sphere_dir;
    float cos_theta_l = max(0.0, dot(-sample.direction, sphere_normal));
    if (cos_theta_l <= 0.0) {
        return; // 从背面看不到
    }
    
    // 几何项
    float surface_area = 4.0 * PI * light.radius * light.radius;
    
    // 物理正确的辐射亮度（朗伯体球光源）
    sample.radiance = light.color * light.intensity;
    
    // PDF转换：面积PDF -> 立体角PDF
    float area_pdf = 1.0 / surface_area;
    float solid_angle_pdf = (distance * distance) / (cos_theta_l * surface_area);
    sample.pdf = solid_angle_pdf;
    
    sample.light_index = -1;
    sample.valid = (sample.pdf > 0.0);
}

void sample_spot_light(inout Light light, inout float3 hit_point, inout uint seed, out LightSample sample) {
    sample = (LightSample)0;
    
    if (!light.enabled) return;
    
    float3 to_light = light.position - hit_point;
    float distance = length(to_light);
    
    if (distance < 1e-6) return;

    sample.position = light.position;
    sample.direction = normalize(to_light);
    
    // // 锥角检查
    float cos_angle = dot(-sample.direction, light.direction);
    float cos_cutoff = cos(radians(light.cone_angle * 0.5));

    if (cos_angle < cos_cutoff) {
        return; // 在锥角外
    }
    
    // 平滑过渡
    float epsilon = 0.1 * (1.0 - cos_cutoff);
    float spot_factor = smoothstep(cos_cutoff, cos_cutoff + epsilon, cos_angle);
    
    // // 物理正确的辐射亮度
    float solid_angle = 2.0 * PI * (1.0 - cos_cutoff);
    sample.radiance = light.color * light.intensity * spot_factor / (distance * distance + 1e-6);
    
    sample.pdf = 1.0;  
    sample.light_index = -1;
    sample.valid = true;
}

// ====================== MIS核心 ======================

// 平衡启发式 MIS 权重
float mis_balance_weight(float pdf_a, float pdf_b) {
    float w = pdf_a ;
    float total = w + pdf_b ;
    return total > 0.0 ? w / total : 0.0;
}

// 功率启发式 MIS 权重
float mis_power_weight(float pdf_a, float pdf_b) {
    float w = pdf_a * pdf_a ; // β=2
    float total = w + pdf_b * pdf_b ;
    return total > 0.0 ? w / total : 0.0;
}

// ====================== BRDF系统 ======================
float3 evaluate_brdf(float3 wi, float3 wo, float3 normal, Material mat) {
    // 漫反射分量
    float3 diffuse = mat.base_color / PI;
    
    // 镜面分量（简化版）
    float3 h = normalize(wi + wo);
    float ndoth = max(0.0, dot(normal, h));
    float specular = pow(ndoth, 32.0 * (1.0 - mat.roughness));
    
    float3 specular_color = lerp(float3(0.04, 0.04, 0.04), mat.base_color, mat.metallic);
    
    return diffuse + specular_color * specular;
}

float bsdf_pdf(float3 wi, float3 wo, float3 normal, Material mat) {
    if (mat.metallic < 0.5) {
        // 漫反射：余弦加权半球采样
        return max(0.0, dot(wi, normal)) / PI;
    } else {
        // 镜面反射：理想镜面
        return 1.0;
    }
}

// ====================== 阴影测试 ======================
bool test_shadow(float3 hit_point, float3 normal, float3 light_dir, float max_distance) {
    RayDesc shadow_ray;
    shadow_ray.Origin = hit_point + normal * eps;
    shadow_ray.Direction = light_dir;
    shadow_ray.TMin = t_min;
    shadow_ray.TMax = max_distance - eps * 3.0; // prevent hit the light
    
    RayPayload shadow_payload;
    shadow_payload.hit = true; // HitMain is skipped, but miss is not
    
    TraceRay(as, 
             RAY_FLAG_ACCEPT_FIRST_HIT_AND_END_SEARCH | 
             RAY_FLAG_FORCE_OPAQUE | 
             RAY_FLAG_SKIP_CLOSEST_HIT_SHADER,
             0xFF, 0, 1, 0, shadow_ray, shadow_payload);
    
    return shadow_payload.hit;
}

// ====================== MIS直接光照计算 ======================
float3 mis_direct_lighting(inout uint light_count, inout float3 hit_point, inout float3 normal, inout Material mat, 
                           inout float3 wo, inout uint seed) {
    float3 total_light = float3(0, 0, 0);

    if (!is_enable_lights(light_count) || mat.metallic > 0.5) return total_light;

    
    for (int i = 0; i < light_count && i < 4; ++i) {
        // 按功率重要性采样光源
        int light_idx = sample_light_by_power(light_count, seed);
        if (light_idx < 0) continue;
        
        Light light = lights[light_idx];
        LightSample light_sample;

        switch (light.type) {
            case 0: sample_point_light(light, hit_point, seed, light_sample); break;
            case 1: sample_area_light(light, hit_point, seed, light_sample); break;
            case 2: sample_spot_light(light, hit_point, seed, light_sample); break;
            case 3: sample_sphere_light(light, hit_point, seed, light_sample); break;
        }

        
        if (!light_sample.valid){
          continue;
        }

        // 阴影测试
        float light_distance = length(light_sample.position - hit_point);
        if (test_shadow(hit_point, normal, light_sample.direction, light_distance)) {
            continue;
        }
        
        // 计算 BRDF 贡献
        float3 wi = light_sample.direction;
        float3 brdf = evaluate_brdf(wi, wo, normal, mat);
        float ndotl = max(0.0, dot(normal, wi));
        float3 light_contrib = light_sample.radiance * brdf * ndotl;
        
        if (any(light_contrib > 0.0)) {
            // 计算 BSDF PDF（用于 MIS）
            float light_select_pdf = light_power_weights[light_idx];
            float light_pdf = light_sample.pdf * light_select_pdf;
            
            if(light.type == 0 || light.type == 2){
                total_light += light_contrib / light_pdf; // 点光源，聚光灯直接加
            }
            else{
              float bsdf_pdf_val = bsdf_pdf(wi, wo, normal, mat);
              float mis_weight = mis_balance_weight(light_pdf, bsdf_pdf_val);
              
              total_light += light_contrib * mis_weight / light_pdf;
            }
        }
    }
    return total_light;
}

float f3_max(float3 u){
  return max(u[0], max(u[1], u[2]));
}

// // ====================== 主渲染逻辑 ======================
[shader("raygeneration")]
void RayGenMain() {
    uint2 pixel_coords = DispatchRaysIndex().xy;
    uint seed = generate_seed(pixel_coords, render_setting.frame_count);

    // 生成相机射线
    float2 pixel_center = (float2)DispatchRaysIndex() + 
                         (render_setting.enable_accumulation ? random2(seed) : float2(0.5, 0.5));
    float2 uv = pixel_center / float2(DispatchRaysDimensions().xy);
    uv.y = 1.0 - uv.y;
    
    float2 d = uv * 2.0 - 1.0;
    float4 origin = mul(camera_info.camera_to_world, float4(0, 0, 0, 1));
    float4 target = mul(camera_info.screen_to_camera, float4(d, 1, 1));
    float4 direction = mul(camera_info.camera_to_world, float4(target.xyz, 0));

    // 路径追踪
    float3 color = float3(0.0, 0.0, 0.0);
    float3 throughput = float3(1.0, 1.0, 1.0);
    RayDesc ray;
    ray.Origin = origin.xyz;
    ray.Direction = normalize(direction.xyz);
    ray.TMin = t_min;
    ray.TMax = t_max;

    entity_id_output[pixel_coords] = -1;

    float3 prev_hit_point, prev_eval_brdf;
    float prev_bsdf_pdf;
    bool prev_is_specular = false;

    uint light_count, stride_;
    
    lights.GetDimensions(light_count, stride_);

    for (int depth = 0; depth < min(render_setting.max_depth, MAX_DEPTH); ++depth) {
        RayPayload payload;
        payload.hit = false;
        
        TraceRay(as, RAY_FLAG_NONE, 0xFF, 0, 1, 0, ray, payload);
        
        if (!payload.hit) {
            // 天空盒
            color += payload.color * throughput;
            break;
        }
        
        // 记录首次命中的实体ID
        if (depth == 0) {
            entity_id_output[pixel_coords] = (int)payload.instance_id;
        }

        Material mat = materials[payload.instance_id];
        float3 wo = -ray.Direction;
        
        if (mat.light_index != 0xFFFFFFFF) {
            if (depth == 0 || prev_is_specular) {
                // 第一次直接击中光源或镜面反射击中光源
                Light light = lights[mat.light_index];
                color += light.color * light.intensity * throughput;
            } else {
                // BSDF采样击中光源，需要MIS
                Light light = lights[mat.light_index];
                if(!light.enabled){
                  break;
                }
                
                float3 to_light = payload.hit_point - prev_hit_point;
                float distance = length(to_light);
                float3 light_dir = to_light / distance;
                
                // 计算光源采样PDF
                float light_pdf = 0.0;
                float cos_theta_l = max(0.0, dot(-light_dir, payload.normal));
                
                switch (light.type) {
                    case 1: // 面光源
                        {
                            float area = light.size.x * light.size.y;
                            light_pdf = (distance * distance) / (cos_theta_l * area);
                        }
                        break;
                    case 3: // 球光源
                        {
                            float surface_area = 4.0 * PI * light.radius * light.radius;
                            light_pdf = (distance * distance) / (cos_theta_l * surface_area);
                        }
                        break;
                }
                
                float light_select_pdf = light_power_weights[mat.light_index];
                light_pdf *= light_select_pdf;

                // 计算MIS权重
                float mis_weight = mis_balance_weight(prev_bsdf_pdf, light_pdf);
                
                // 应用MIS
                color += light.color * light.intensity * throughput * prev_eval_brdf * mis_weight / prev_bsdf_pdf;
            }
            break;
        }

        prev_hit_point = payload.hit_point;

        // MIS直接光照（只考虑非自发光表面）
        float3 direct_light = mis_direct_lighting(light_count, payload.hit_point, payload.normal, mat, wo, seed);
        color += direct_light * throughput;
        
        // 俄罗斯轮盘赌终止
        if (depth > 4) {
            float p_survive = min(f3_max(throughput), RR_THRESHOLD);
            if (random(seed) > p_survive) break;
            throughput /= p_survive;
        }
        
        // 采样下一跳方向（BSDF采样）
        float3 next_dir;
        
        if (mat.metallic < 0.5) {
            // 漫反射：余弦半球采样
            next_dir = cosine_sample_hemisphere(seed, payload.normal);
            prev_bsdf_pdf = max(0.0, dot(next_dir, payload.normal)) / PI;
            
            // 更新吞吐量
            float3 brdf = evaluate_brdf(next_dir, wo, payload.normal, mat);
            float ndotl = max(0.0, dot(payload.normal, next_dir));
            prev_eval_brdf = brdf * ndotl;
            throughput *= prev_eval_brdf / prev_bsdf_pdf;
            prev_is_specular = false;
        } else {
            // 镜面反射：理想镜面
            next_dir = reflect(ray.Direction, payload.normal);
            
            // Fresnel项
            float3 h = normalize(next_dir + wo);
            float ndoth = max(0.0, dot(payload.normal, h));
            float3 fresnel = lerp(float3(0.04, 0.04, 0.04), mat.base_color, mat.metallic);
            throughput *= fresnel;
            prev_is_specular = true;
        }
        
        // 准备下一次光线
        ray.Origin = payload.hit_point + payload.normal * eps;
        ray.Direction = normalize(next_dir);
        ray.TMin = t_min;
        ray.TMax = t_max;
    }
    
    // 输出和累积
    output[pixel_coords] = float4(color, 1);
    
    if (render_setting.enable_accumulation) {
        float4 prev_color = accumulated_color[pixel_coords];
        int prev_samples = accumulated_samples[pixel_coords];
        accumulated_color[pixel_coords] = prev_color + float4(color, 1);
        accumulated_samples[pixel_coords] = prev_samples + 1;
    }
}


// // ====================== 命中着色器 ======================
[shader("miss")]
void MissMain(inout RayPayload payload) {
    // 简化的天空渐变
    float t = 0.5 * (normalize(WorldRayDirection()).y + 1.0);
    payload.color = lerp(float3(1.0, 1.0, 1.0), float3(0.5, 0.7, 1.0), t) * 0.5;
    payload.hit = false;
    payload.instance_id = 0xFFFFFFFF;
}

[shader("closesthit")]
void ClosestHitMain(inout RayPayload payload, in BuiltInTriangleIntersectionAttributes attr) {
    payload.hit = true;
    payload.instance_id = InstanceID();
    uint primitive_id = PrimitiveIndex();
    
    // 获取几何数据
    GeometryDescriptor geo_desc = geometry_descriptors[payload.instance_id];
    
    // 计算世界空间命中点
    float3 ray_origin = WorldRayOrigin();
    float3 ray_direction = WorldRayDirection();
    float hit_distance = RayTCurrent();
    payload.hit_point = ray_origin + hit_distance * ray_direction;
    
    // 获取三角形顶点
    uint index_offset = geo_desc.index_offset + primitive_id * 3;
    uint i0 = indices[index_offset];
    uint i1 = indices[index_offset + 1];
    uint i2 = indices[index_offset + 2];
    
    VertexInfo v0 = vertices[geo_desc.vertex_offset + i0];
    VertexInfo v1 = vertices[geo_desc.vertex_offset + i1];
    VertexInfo v2 = vertices[geo_desc.vertex_offset + i2];
    
    // 计算法线
    float3 object_space_normal;
    if (length(v0.normal) < 1e-6) {
        // 使用几何法线
        object_space_normal = cross(v1.position - v0.position, v2.position - v0.position);
    } else {
        // 插值顶点法线
        float w0 = attr.barycentrics.x;
        float w1 = attr.barycentrics.y;
        object_space_normal = w0 * v1.normal + w1 * v2.normal + (1.0 - w0 - w1) * v0.normal;
    }
    
    object_space_normal = normalize(object_space_normal);
    
    // 转换到世界空间
    float3x3 normal_matrix = (float3x3)transpose(WorldToObject4x3());
    payload.normal = normalize(mul(normal_matrix, object_space_normal));
    
    // 确保法线朝向射线方向
    if (dot(payload.normal, ray_direction) > 0) {
        payload.normal = -payload.normal;
    }
}