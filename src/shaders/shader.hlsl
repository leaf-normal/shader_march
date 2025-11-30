
struct CameraInfo {
  float4x4 screen_to_camera;
  float4x4 camera_to_world;
};

struct Material {
  float3 base_color;
  float roughness;
  float metallic;

  // float3 emission;        // 自发光颜色
  // float ior;                // 折射率
  // float transparency;        
  // int material_type;        // 材质类型: 0=漫反射, 1=镜面, 2=玻璃, 3=发射
  // int texture_id;       
};

struct HoverInfo {
  int hovered_entity_id;
};

RaytracingAccelerationStructure as : register(t0, space0);
RWTexture2D<float4> output : register(u0, space1);
ConstantBuffer<CameraInfo> camera_info : register(b0, space2);
StructuredBuffer<Material> materials : register(t0, space3);
ConstantBuffer<HoverInfo> hover_info : register(b0, space4);
RWTexture2D<int> entity_id_output : register(u0, space5);
RWTexture2D<float4> accumulated_color : register(u0, space6);
RWTexture2D<int> accumulated_samples : register(u0, space7);

// *add
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

// *add
StructuredBuffer<GeometryDescriptor> geometry_descriptors : register(t0, space8);
StructuredBuffer<VertexInfo> vertices : register(t0, space9);
StructuredBuffer<uint> indices : register(t0, space10);

struct RayPayload {
  float3 color;
  bool hit;
  uint instance_id;
  float3 normal;
  float3 hit_point;
};

#define MAX_DEPTH 16
#define RR_THRESHOLD 0.95f
#define t_min 0.001
#define t_max 10000.0
#define eps 5e-4

void wanghash(inout uint seed)
{
  seed=(seed^61)^(seed>>16);
  seed=seed*9;
  seed=seed^(seed>>4);
  seed=seed*0x27d4eb2d;
  seed=seed^(seed>>15);
}

float random(inout uint seed)
{
  wanghash(seed);
  return float(seed)/4294967296.0;
}

float f3_max(float3 u){
  return max(u[0], max(u[1], u[2]));
}

void SampleBSDF(Material material, float3 ray, float3 normal, out float3 wi, out float pdf, inout uint seed){
  // do something
}

[shader("raygeneration")] void RayGenMain() {

  float2 pixel_center = (float2)DispatchRaysIndex() + float2(0.5,0.5); //float2(random(), random());
  float2 uv = pixel_center / float2(DispatchRaysDimensions().xy);
  uv.y = 1.0 - uv.y; 
  float2 d = uv * 2.0 - 1.0;
  float4 origin = mul(camera_info.camera_to_world, float4(0, 0, 0, 1));
  float4 target = mul(camera_info.screen_to_camera, float4(d, 1, 1));
  float4 direction = mul(camera_info.camera_to_world, float4(target.xyz, 0));

  uint2 pixel_coords = DispatchRaysIndex().xy;

  int seed = pixel_coords[0] * pixel_coords[1];

  float3 color = float3(0.0, 0.0, 0.0);
  float3 throughout = float3(1.0, 1.0, 1.0);

  RayDesc ray;
  ray.Origin = origin.xyz;
  ray.Direction = normalize(direction.xyz);
  ray.TMin = t_min;
  ray.TMax = t_max;

  entity_id_output[pixel_coords] =  -1;


  for(int depth = 0; depth < MAX_DEPTH; ++depth){

    RayPayload payload;
    payload.color = float3(0, 0, 0);
    payload.hit = false;
    payload.instance_id = 0;

    TraceRay(as, RAY_FLAG_NONE, 0xFF, 0, 1, 0, ray, payload); // direction of ray should always be normalized
        
    if(!payload.hit){
      color += payload.color * throughout;
      break;
    }

    if(depth == 0){
      // Write entity ID to the ID buffer
      // If no hit, write -1; otherwise write the instance ID
      entity_id_output[pixel_coords] = (int)payload.instance_id;
    }

    Material mat = materials[payload.instance_id];
    
    if (mat.metallic < 0.5){
      color += payload.color * throughout;
      break; 
    }

    if(depth > 4){
      float p_survive = min(f3_max(throughout), RR_THRESHOLD);
      if(random(seed) > p_survive){
        break;
      }
      throughout /= p_survive;

    }
    
    // color += throughout * material.emission;

    // float pdf, wi;
    // float3 bsdf = SampleBSDF(mat, -ray.Direction, payload.normal, wi, pdf, seed);

    // if(pdf <= 0.0) break;

    // float cosTheta = dot(wi, payload.normal);

    // throughout *= bsdf * abs(cosTheta) / pdf;

    throughout = throughout * mat.base_color;

    ray.Origin = payload.hit_point + eps * payload.normal;
    ray.Direction = normalize( reflect(ray.Direction, payload.normal) ); // reflect(u, v) = u - 2 dot(u, v) * v
    ray.TMin = t_min;
    ray.TMax = t_max;

  }
  
  // Write to immediate output (for camera movement mode)
  output[pixel_coords] = float4(color, 1);
  
  // Accumulate color for progressive rendering (when camera is stationary)
  float4 prev_color = accumulated_color[pixel_coords];
  int prev_samples = accumulated_samples[pixel_coords];
  
  accumulated_color[pixel_coords] = prev_color + float4(color, 1);
  accumulated_samples[pixel_coords] = prev_samples + 1;

}

[shader("miss")] void MissMain(inout RayPayload payload) {
  // Sky gradient
  float t = 0.5 * (normalize(WorldRayDirection()).y + 1.0);
  payload.color = lerp(float3(1.0, 1.0, 1.0), float3(0.5, 0.7, 1.0), t);
  payload.hit = false;
  payload.instance_id = 0xFFFFFFFF; // Invalid ID for miss
}

[shader("closesthit")] void ClosestHitMain(inout RayPayload payload, in BuiltInTriangleIntersectionAttributes attr) {
  // *add, all below is modified

  payload.hit = true;

  uint instance_id = InstanceID();
  uint primitive_id = PrimitiveIndex();
  payload.instance_id = instance_id;

  uint geometry_descriptors_count, _;
  geometry_descriptors.GetDimensions(geometry_descriptors_count, _);
  if (instance_id >= geometry_descriptors_count) {  // remove this will result in an unexpected error 
      payload.hit = false;
      return;
  }
  
  // 3. 获取几何描述符
  GeometryDescriptor geo_desc = geometry_descriptors[instance_id];
  
  // 4. 计算世界空间命中点
  float3 ray_origin = WorldRayOrigin();
  float3 ray_direction = WorldRayDirection();
  float hit_distance = RayTCurrent();
  payload.hit_point = ray_origin + hit_distance * ray_direction;
  
  // 5. 获取三角形顶点索引
  uint index_offset = geo_desc.index_offset + primitive_id * 3;
  uint i0 = indices[index_offset];
  uint i1 = indices[index_offset + 1];
  uint i2 = indices[index_offset + 2];
  
  // 6. 获取顶点位置（用于精确命中点计算）

  VertexInfo v0 = vertices[geo_desc.vertex_offset + i0];
  VertexInfo v1 = vertices[geo_desc.vertex_offset + i1];
  VertexInfo v2 = vertices[geo_desc.vertex_offset + i2];
  float3 object_space_normal;
  
  if(length(v0.normal) < 1e-6){
    // normal undefined (set to 0), use surface normal
    object_space_normal = cross(v1.position - v0.position, v2.position - v0.position); 
  }
  else{
    float w0 = attr.barycentrics.x;
    float w1 = attr.barycentrics.y;

    object_space_normal = w0 * v1.normal + w1 * v2.normal + (1.0 - w0 - w1) * v0.normal; // Attention to the sequence
  }

  object_space_normal = normalize(object_space_normal);

  float3x3 normal_matrix = (float3x3)transpose(WorldToObject4x3()); // transpose inverse matrix
  payload.normal = normalize(mul(normal_matrix , object_space_normal));

  if(dot(payload.normal, ray_direction) > 0){
    payload.normal = - payload.normal;
  }

  Material mat = materials[instance_id];
  
  // 对于非金属，计算直接光照
  if (mat.metallic < 0.5) {
    float3 light_dir = normalize(float3(1, 2, 1));  // 简单的方向光
    float3 view_dir = -ray_direction;
    
    // 漫反射分量
    float ndotl = max(0.0, dot(payload.normal, light_dir));
    float3 diffuse = mat.base_color * ndotl;
    
    // 环境光分量
    float3 ambient = mat.base_color * 0.2;
    
    // 简单的镜面反射（基于粗糙度）
    float3 reflect_dir = reflect(-light_dir, payload.normal);
    float spec = pow(max(0.0, dot(view_dir, reflect_dir)), 32.0 * (1.0 - mat.roughness));
    float3 specular = float3(0.3, 0.3, 0.3) * spec * (1.0 - mat.roughness);
    
    // 将计算出的直接光照结果存入 payload
    payload.color = ambient + diffuse + specular;
  }
  else {
    payload.color = float3(0.0 ,0.0, 0.0);
  }
}