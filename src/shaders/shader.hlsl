struct CameraInfo {
  float4x4 screen_to_camera;
  float4x4 camera_to_world;
  float focal_distance;     // 焦点距离（世界空间）
  float aperture_size;      // 光圈直径（控制模糊强度）
  float focal_length;       // 焦距（控制视角）
  float lens_radius;        // 透镜半径 = aperture_size/2
  int enable_depth_of_field;

  float3 camera_linear_velocity;  // 相机线性速度
  float camera_angular_velocity; // 相机角速度（绕相机方向的旋转）
  int enable_motion_blur;            // 是否启用运动模糊
  float exposure_time;    
};

struct Material {
  float3 base_color;
  float roughness;
  float metallic;
  uint light_index;
  
  float3 emission;          // 自发光颜色
  float ior;                // 折射率
  float transparency;       // 透明度 
  int texture_id;
  int normal_tex_id;
  int attribute_tex_id;
  
  float subsurface;     //次表面散射
  float specular;      //镜面反射强度
  float specular_tint; //镜面反射
  float anisotropic;   //各向异性[-1,1]
  float sheen;         //光泽层
  float sheen_tint;    //光泽层染色
  float clearcoat;     //清漆层强度
  float clearcoat_roughness; //清漆层粗糙度

  uint group_id;
};

struct HoverInfo {
  int hovered_entity_id;
};

// ====================== 常量缓冲区定义 ======================

#define MAX_MOTION_GROUPS 1

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
    float2 texcoord;
};

StructuredBuffer<GeometryDescriptor> geometry_descriptors : register(t0, space8);
StructuredBuffer<VertexInfo> vertices : register(t0, space9);
StructuredBuffer<uint> indices : register(t0, space10);

struct RenderSettings {
    uint frame_count;
    uint samples_per_pixel;
    uint max_depth;
    uint enable_accumulation;
    uint light_count;
    int skybox_texture_id_;            // If not able, set to -1
};

ConstantBuffer<RenderSettings> render_setting : register(b0, space11);

// ====================== 光源系统 ======================
#define MAX_LIGHTS 32

struct Light {
    float3 position;       // 光源位置
    float3 direction;      // 方向向量：
                             // - 点光源：不使用（置0）
                             // - 面光源：法线方向（单向）
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

StructuredBuffer<Light> lights : register(t0, space12);                // hdr is not included in lights
StructuredBuffer<float> light_power_weights : register(t0, space13);   // Range from [0, light_count + 1), with power weight of hdr added to last position

// 物体运动

struct MotionParams {
    float3 linear_velocity;
    float3 angular_velocity;
    float3 pivot_point;          
    int is_static;
    int group_id; 
};
StructuredBuffer<MotionParams> motion_groups : register(t0, space14);

// Texture
Texture2D<float4> g_Textures[128] : register(t0, space18);
SamplerState g_Sampler : register(s0, space19);

// hdr_cdf
StructuredBuffer<float> hdr_cdf : register(t0, space20);   // [width, height, total_luminance] + conditional_cdf + marginal_cdf


struct RayPayload {
  float3 color;
  bool hit;
  uint instance_id;
  float3 normal;
  float3 hit_point;
  bool is_filp;
  float4 attribute;
};

// ====================== 常量定义 ======================
#define MAX_DEPTH 8
#define RR_THRESHOLD 0.95f
#define t_min 1e-3
#define t_max 10000.0
#define eps 5e-4 // used for geometry
#define EPS 1e-6 // used for division
#define PI 3.14159265359
#define TWO_PI 6.28318530718
#define INV_PI 0.31830988618

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

  if(render_setting.skybox_texture_id_ >= 0) return true;
  
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
    
    return render_setting.skybox_texture_id_ >= 0 ? light_count : last;
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
        sample.valid = 0;
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
    
    // 锥角检查
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

uint binary_search_cdf(float rand_value, uint start_idx, uint count) {
    uint left = 0;
    uint right = count - 1;
    uint result = 0;
    
    while (left <= right) {
        uint mid = (left + right) >> 1;  // 除以2的位运算优化
        float cdf_value = hdr_cdf[start_idx + mid];
        
        if (rand_value <= cdf_value) {
            result = mid;
            if (mid == 0 || rand_value > hdr_cdf[start_idx + mid - 1]) {
                break;
            }
            right = mid - 1;
        } else {
            left = mid + 1;
        }
    }
    return result;
}

void sample_hdr(inout float3 hit_point, inout uint seed, out LightSample sample) {
    sample = (LightSample)0;
    
    if (render_setting.skybox_texture_id_ < 0) {
        return;
    }
    
    // 从CDF缓冲区读取宽度和高度
    uint width = (uint)hdr_cdf[0];
    uint height = (uint)hdr_cdf[1];
    float hdr_total_luminance = hdr_cdf[2];
    
    uint cdf_offset = 3; 
    
    // 1. 使用二分查找采样边缘CDF（行）
    uint marginal_cdf_start = cdf_offset + (uint)(width * height);
    uint row = binary_search_cdf(random(seed), marginal_cdf_start, height);
    
    // 2. 使用二分查找采样条件CDF（列）
    uint conditional_cdf_start = cdf_offset + row * (uint)width;
    uint col = binary_search_cdf(random(seed), conditional_cdf_start, width);
    
    float u = (float(col) + 0.5) / width;
    float v = (float(row) + 0.5) / height;
    
    float phi = u * TWO_PI;
    float theta = v * PI;
    
    float sin_theta = sin(theta);
    float cos_theta = cos(theta);
    float sin_phi = sin(phi);
    float cos_phi = cos(phi);
    
    float3 direction = float3(sin_theta * cos_phi, cos_theta, sin_theta * sin_phi);
    
    float3 hdr_color = g_Textures[render_setting.skybox_texture_id_].SampleLevel(g_Sampler, float2(u, v), 0).rgb;
    float luminance = dot(hdr_color, float3(0.2126, 0.7152, 0.0722));
    
    float solid_angle_pdf = luminance / hdr_total_luminance;
    
    // 5. 设置采样结果
    sample.position = hit_point + direction * 10000.0; // 远点
    sample.direction = direction;
    sample.radiance = hdr_color;
    sample.pdf = solid_angle_pdf;
    sample.valid = (solid_angle_pdf > 0.0) && (luminance > 0.0);
}


// ====================== MIS核心 ======================

// 平衡启发式 MIS 权重
float mis_balance_weight(float pdf_a, float pdf_b) {
    float total = pdf_a + pdf_b ;
    return total > 0.0 ? pdf_a / total : 0.0;
}

// 功率启发式 MIS 权重
float mis_power_weight(float pdf_a, float pdf_b) {
    float w = pdf_a * pdf_a ; // β=2
    float total = w + pdf_b * pdf_b ;
    return total > 0.0 ? w / total : 0.0;
}

// ====================== BSDF系统 ======================
float f3_max(inout float3 u){
  return max(u[0], max(u[1], u[2]));
}
float sqr(float x)
{
  return x*x;
}
float3 sqr(float3 x)
{
  return x*x;
}
bool Refract(float3 v, float3 n, float eta, out float3 t) {
    float c = dot(v, n);
    float k = 1.0 - eta * eta * (1.0 - c * c);
    if (k < 0.0) return false; // 全反射 (TIR)
    t = eta * v - (eta * c + sqrt(k)) * n;
    return true;
}

float SchlickWeight(float cosTheta)
{
  return pow( clamp(1.0 - cosTheta, 0.0, 1.0), 5.0);
}
float SchlickFresnelScalar(float F0, float cosTheta)
{
  return F0 + (1.0 - F0) * SchlickWeight(cosTheta);
}
float3 SchlickFresnel(float3 F0, float cosTheta)
{
  return F0 + (1.0 - F0) * SchlickWeight(cosTheta);
}
float GTR1(float NdotH, float a)
{
  if(a >= 1.0) return INV_PI;
  float a2 = sqr(a);
  float t = 1.0 + (a2 - 1.0) * sqr(NdotH);
  return (a2 - 1.0) / (PI * log(a2 + EPS) * t + EPS);
}
float GTR2(float NdotH, float a)
{
    float a2 = sqr(a);
    float t = 1.0 + (a2 - 1.0) * sqr(NdotH);
    return a2 / (PI * sqr(t) + EPS);
}
float GTR2_Anisotropic(float NdotH, float HdotX, float HdotY, float ax, float ay)
{
    return 1.0 / (PI * ax * ay * sqr(sqr(HdotX / ax) + sqr(HdotY / ay) + NdotH * NdotH) + EPS);
}

float SmithG_GGX(float NdotV, float alphaG) {
    float a=sqr(alphaG);
    float b=sqr(NdotV);
    return 2.0*NdotV/(NdotV+sqrt(a+(1.0-a)*b) + EPS);
}
float SmithG_GGX_Anisotropic(float NdotV, float VdotX, float VdotY, float ax, float ay)
{
    return 2.0*NdotV/(NdotV+sqrt(sqr(VdotX*ax)+sqr(VdotY*ay)+sqr(NdotV)) + EPS);
}
float SmithG_GGX_Correlated(float NdotV, float NdotL, float alpha)
{
    NdotV = max(NdotV, EPS);
    NdotL = max(NdotL, EPS);
    float a2 = alpha * alpha;
    float lambdaV = (-1.0 + sqrt(1.0 + a2 * (1.0 - NdotV * NdotV) / (NdotV * NdotV))) * 0.5;
    float lambdaL = (-1.0 + sqrt(1.0 + a2 * (1.0 - NdotL * NdotL) / (NdotL * NdotL))) * 0.5;
    float G = 1.0 / (1.0 + lambdaV + lambdaL);
    return G;
}
float SmithG_GGX_Correlated_Anisotropic(float NdotV,float NdotL,float VdotX,float VdotY,float LdotX,float LdotY,float ax,float ay)
{
    NdotV = max(NdotV, EPS);
    NdotL = max(NdotL, EPS);
    float lambdaV = (-1.0 + sqrt(1.0 + (sqr(VdotX * ax) + sqr(VdotY * ay)) / sqr(NdotV))) * 0.5;
    float lambdaL = (-1.0 + sqrt(1.0 + (sqr(LdotX * ax) + sqr(LdotY * ay)) / sqr(NdotL))) * 0.5;
    float G = 1.0 / (1.0 + lambdaV + lambdaL);
    return G;
}
float3 SampleHemisphereCos(float2 randd, inout float3 normal)
{
  float r=sqrt(randd.x);
  float theta=2.0*PI*randd.y;
  float x=r*cos(theta);
  float y=r*sin(theta);
  float z=sqrt(max(0.0,1.0-randd.x));
  float3 up=abs(normal.z)<1-eps?float3(0.0,0.0,1.0):float3(1.0,0.0,0.0);
  float3 tangent=normalize(cross(up, normal));
  float3 bitangent=cross(normal,tangent);
  return normalize(x*tangent+y*bitangent+z*normal);
}

void GetTangent(inout float3 normal, out float3 tangent, out float3 bitangent)
{
  float3 up=abs(normal.z)<0.999?float3(0.0,0.0,1.0):float3(1.0,0.0,0.0);
  tangent=normalize(cross(up,normal));
  bitangent=cross(normal,tangent);
}

float3 SampleGGX_VNDF_Isotropic(float3 V_world, float3 N_world, float roughness,float2 u)
{
    float3 T, B;
    GetTangent(N_world, T, B);
    float3 V = normalize(float3(dot(V_world, T), dot(V_world, B), dot(V_world, N_world)));
    float alpha = max(EPS, roughness);
    float3 Vh = normalize(float3(alpha * V.x, alpha * V.y, V.z));
    float3 up = (abs(Vh.z) < 1.0 - EPS) ? float3(0.0, 0.0, 1.0) : float3(1.0, 0.0, 0.0);
    float3 T1 = normalize(cross(up, Vh));
    float3 T2 = cross(Vh, T1);
    float r = sqrt(u.x);
    float phi = 2.0 * PI * u.y;
    float t1 = r * cos(phi);
    float t2 = r * sin(phi);
    float s  = 0.5 * (1.0 + Vh.z);
    t2 = (1.0 - s) * sqrt(max(0.0, 1.0 - t1 * t1)) + s * t2;
    float3 Nh = t1 * T1 + t2 * T2 + sqrt(max(0.0, 1.0 - t1 * t1 - t2 * t2)) * Vh;
    float3 H = normalize(float3(alpha * Nh.x, alpha * Nh.y, max(0.0, Nh.z)));
    float3 H_world = normalize(H.x * T + H.y * B + H.z * N_world);
    return H_world;
}

float3 SampleGGX_Anisotropic(float3 ray, float roughness, float anisotropic, float2 rd, float3 normal, float3 tangent)
{

    float3 bitangent = cross(normal, tangent);

    float3 V_local = normalize(float3(dot(ray, tangent), dot(ray, bitangent), dot(ray, normal)));

    float aspect = sqrt(1.0 - 0.9 * anisotropic);
    float ax = max(eps, roughness / aspect);
    float ay = max(eps, roughness * aspect);
    float3 Vh = normalize(float3(ax * V_local.x, ay * V_local.y, V_local.z));
    float3 up = abs(Vh.z) < 1.0 - eps ? float3(0.0, 0.0, 1.0) : float3(1.0, 0.0, 0.0);
    float3 T1 = normalize(cross(up, Vh));
    float3 T2 = cross(Vh, T1);
    float r = sqrt(rd.x);
    float phi = 2.0 * PI * rd.y;
    float t1 = r * cos(phi);
    float t2 = r * sin(phi);
    float s = 0.5 * (1.0 + Vh.z);
    t2 = (1.0 - s) * sqrt(max(0.0, 1.0 - t1 * t1)) + s * t2;
    float Nh_len = sqrt(max(0.0, 1.0 - t1 * t1 - t2 * t2));
    float3 Nh = t1 * T1 + t2 * T2 + Nh_len * Vh;
    float3 H_local = normalize(float3(ax * Nh.x, ay * Nh.y, max(0.0, Nh.z)));
    float3 H_world = H_local.x * tangent + H_local.y * bitangent + H_local.z * normal;
    return H_world;
}


// --- 新增的透射相关辅助函数 (修正版) ---


float3 SampleGGX_Distribution(float roughness, float2 rd, inout float3 normal, inout float3 tangent, inout float3 bitangent)
{
    float a = sqr(roughness);
    float a2 = sqr(a);
    float phi = 2.0 * PI * rd.y;
    float cos_theta = sqrt((1.0 - rd.x) / (1.0 - rd.x + a2 * rd.x));
    float sin_theta = sqrt(max(0.0, 1.0 - cos_theta * cos_theta));
    float3 H_local = float3(sin_theta * cos(phi), sin_theta * sin(phi), cos_theta);
    float3 H_world = H_local.x * tangent + H_local.y * bitangent + H_local.z * normal;
    return normalize(H_world);
}


bool SampleBSDF(inout Material mat, inout float3 ray, inout float3 normal, out float3 wi, bool is_flip, inout uint seed){

  //lobe权重
  float diffuseweight=(1.0-mat.metallic)*(1.0-mat.transparency);//漫反射
  float specularweight=lerp(0.04, 1.0, mat.metallic)*(1.0-mat.transparency);
  float transmissionweight=mat.transparency*(1.0-mat.metallic);//透射
  float clearcoatweight=0.25*mat.clearcoat*(1.0-mat.transparency);//清漆层
  float sheenweight=mat.sheen*(1.0-mat.metallic)*(1.0-mat.transparency);//光泽层
  //normalization
  float total=diffuseweight+specularweight+transmissionweight+sheenweight+clearcoatweight;
  diffuseweight/=total;
  specularweight/=total;
  transmissionweight/=total;
  sheenweight/=total;
  clearcoatweight/=total;
  float randLobe=random(seed);
  float3 tangent, bitangent;
  GetTangent(normal, tangent, bitangent);  
  if(randLobe<diffuseweight+sheenweight)//漫反射+简化光泽层模型
  {
    wi=SampleHemisphereCos(random2(seed),normal);
  }else if(randLobe<diffuseweight+specularweight+sheenweight)//镜面反射
  {
    float3 H;
    H=SampleGGX_Anisotropic(ray,mat.roughness,mat.anisotropic,random2(seed),normal,tangent);
    wi=reflect(-ray,H);
    if(dot(normal,wi)<=0.0)return 0;
  }else if(randLobe<diffuseweight+specularweight+transmissionweight+sheenweight)//透射
  {
    float eta=(!is_flip)?1.0/mat.ior:mat.ior;
    float3 H=SampleGGX_Distribution(mat.roughness, random2(seed), normal, tangent, bitangent);//后续可以继续改为VNDF
    if(Refract(-ray,H,eta,wi)==false)
    {
        wi=reflect(-ray,H);//全反射
        if(dot(normal,wi)<=0.0)return 0;
    }
  }else{//清漆
    float alpha_c=max(EPS, mat.clearcoat_roughness);
    float3 H=SampleGGX_Distribution(alpha_c,random2(seed),normal,tangent,bitangent);
    wi=reflect(-ray,H);
    if(dot(normal,wi)<=0.0)return 0;
  }
  wi=normalize(wi);
  return 1;
}

float3 EvalBSDF(inout Material mat, inout float3 ray, inout float3 wi, inout float3 normal, bool is_flip, out float pdf)
{

  float3 ret=float3(0.0,0.0,0.0);
  float3 tangent,bitangent;
  float3 F0=lerp(0.08*mat.specular,mat.base_color,mat.metallic);
  GetTangent(normal,tangent,bitangent);
  float Ndotray=dot(normal,ray);
  float Ndotwi=dot(normal,wi);
  bool is_trans=(Ndotray*Ndotwi)<0.0;
  if(Ndotray<=0.0)
  {
    pdf=0.0;
    return float3(1e9,0.0,0.0);
  }
  if(!is_trans&&Ndotwi<=0.0)
  {
    pdf=0.0;
    return float3(0.0,0.0,0.0);
  }
  float alpha=sqr(mat.roughness);
  float transmissionPDF=0.0;
  float specularPDF=0.0;
  float diffusePDF=0.0;
  float sheenPDF=0.0;
  float clearcoatPDF=0.0;
  
  //lobe权重
  float diffuseweight=(1.0-mat.metallic)*(1.0-mat.transparency);//漫反射
  float specularweight=lerp(0.04, 1.0, mat.metallic)*(1.0-mat.transparency);
  float transmissionweight=mat.transparency*(1.0-mat.metallic);//透射
  float sheenweight=mat.sheen*(1.0-mat.metallic)*(1.0-mat.transparency);//光泽层
  float clearcoatweight=0.25*mat.clearcoat*(1.0-mat.transparency);//清漆层  
  //normalization
  float total=diffuseweight+specularweight+transmissionweight+sheenweight+clearcoatweight;
  diffuseweight/=total;
  specularweight/=total;
  transmissionweight/=total;
  sheenweight/=total;
  clearcoatweight/=total;
  if(is_trans)//透射
  {
    float n1 = is_flip ? mat.ior : 1.0; // V 侧 (Camera 侧)
    float n2 = is_flip ? 1.0 : mat.ior; // L 侧 (Light 侧)
    float3 V = ray;
    float3 L = wi;
    float3 H = -(n1 * V + n2 * L);
    if (dot(H, H) < EPS) {
        pdf = 0.0;
        return float3(0.0, 0.0, 0.0);
    }
    H = normalize(H);
    float VdotH = dot(V, H);
    float LdotH = dot(L, H);
    float NdotH = dot(normal, H);
    float NdotV = dot(normal, V);
    float NdotL = dot(normal, L);
    float D = GTR2(NdotH, mat.roughness); // 使用你现有的 GGX D
    float G = SmithG_GGX_Correlated(abs(NdotV), abs(NdotL), mat.roughness); // 你的 G
    float3 F_color = SchlickFresnel(F0, abs(VdotH)); 
    float3 T_color = 1.0 - F_color; // 透射颜色
    float sqrtDenom = n1 * VdotH + n2 * LdotH;
    float commonDenom = sqrtDenom * sqrtDenom + EPS;
    float factor = (abs(VdotH) * abs(LdotH) * n2 * n2) / (abs(NdotV) * abs(NdotL) * commonDenom);
    ret += mat.base_color * T_color * D * G * factor * transmissionweight;
    float jacobian = (n2 * n2 * abs(LdotH)) / commonDenom;
    transmissionPDF = D * abs(NdotH) * jacobian;
  }else{
    float3 H=normalize(ray+wi);
    float NdotH=dot(normal,H);
    float Hdotray=dot(H,ray);
    float3 F=SchlickFresnel(F0,Hdotray);
    float VdotH = dot(ray, H);
    float LdotH = dot(wi, H);
    float NdotV = dot(normal, ray);
    float NdotL = dot(normal, wi);
    //漫反射
    if(mat.metallic<1.0&&mat.transparency<1.0)
    {
      float FL=SchlickWeight(Ndotray);
      float FV=SchlickWeight(Ndotwi);
      float Fd90=0.5+2.0*mat.roughness*sqr(Hdotray);
      float Fd=lerp(1.0,Fd90,FL)*lerp(1.0,Fd90,FV);
      float3 diffuse=mat.base_color*(1.0-mat.metallic)*Fd/PI;
      ret+=diffuseweight*diffuse*(1.0-F);
    }
    diffusePDF=max(Ndotwi,0.0)/PI;

    // 镜面反射
    float ax=max(EPS,alpha/sqrt(1.0-0.9*mat.anisotropic));
    float ay=max(EPS,alpha*sqrt(1.0-0.9*mat.anisotropic));
    float D=GTR2_Anisotropic(NdotH,dot(H,tangent),dot(H,bitangent),ax,ay);
    float Vis=SmithG_GGX_Correlated_Anisotropic(Ndotray,Ndotwi,dot(ray,tangent), dot(ray, bitangent), dot(wi, tangent), dot(wi, bitangent), ax, ay);
    float3 spec=D*Vis*F; 
    if (mat.specular_tint > EPS)
    {
        float3 tint=lerp(float3(1.0, 1.0, 1.0), mat.base_color, mat.specular_tint);
        spec*=tint;
    }
    ret+=spec*specularweight;
    float G1_V=SmithG_GGX_Anisotropic(Ndotray, dot(ray, tangent), dot(ray, bitangent), ax, ay);
    specularPDF=(D*G1_V)/(4.0*Ndotray+EPS);
    // 全反射
    if(transmissionweight > EPS)
    {
        float eta = (!is_flip) ? 1.0/mat.ior : mat.ior;
        float c = dot(ray, H);
        float k = 1.0 - eta * eta * (1.0 - c * c);
        if (k < 0.0) // 发生了全反射
        {
            float D_iso = GTR2(NdotH, mat.roughness);
            float pdf_h_tir = D_iso * abs(NdotH);
            ret+=spec*transmissionweight;
            transmissionPDF = pdf_h_tir / (4.0 * abs(Hdotray) + EPS);
        }
    }

    //光泽层
    if (mat.sheen > EPS && mat.metallic < 1.0 && mat.transparency < 1.0)
    {
        float3 base=mat.base_color;
        float lum = max(EPS, 0.2126 * base.x + 0.7152 * base.y + 0.0722 * base.z);
        float3 tintColor = base / lum; // 提取色相（去亮度）
        float3 sheenColor = lerp(float3(1.0,1.0,1.0), tintColor, mat.sheen_tint);
        float w = SchlickWeight(abs(Hdotray)); // 角度权重，常用 H·V
        float3 sheenTerm = sheenColor * (mat.sheen) * w;
        ret += sheenweight * sheenTerm * (1.0 - F);
        sheenPDF = diffusePDF; // 复用 cosine 采样的 PDF
    }
    //清漆层
    if (mat.clearcoat > EPS && mat.transparency < 1.0)
    {
        float alpha_c = max(EPS, mat.clearcoat_roughness);
        float Dc = GTR1(NdotH, alpha_c);
        float Gc = SmithG_GGX(NdotV, alpha_c) * SmithG_GGX(NdotL, alpha_c);
        float Fc = SchlickFresnelScalar(0.04, abs(Hdotray));
        float coat = (Dc * Gc * Fc) / (4.0 * abs(NdotV) * abs(NdotL) + EPS);
        ret = (1.0 - mat.clearcoat * Fc) * ret + clearcoatweight * coat;
        clearcoatPDF = (Dc * abs(NdotH)) / (4.0 * abs(Hdotray) + EPS);
    }
  }
  pdf=diffusePDF*diffuseweight+specularPDF*specularweight+transmissionPDF*transmissionweight+sheenPDF*sheenweight+clearcoatPDF*clearcoatweight;
  return ret;
}
// ====================== 阴影测试 ======================
// void Traceray(inout RayDesc ray, inout RayPayload payload, bool for_shadow_test) {
    
//     float best_t = 1e9;
//     payload.hit = false;
    
//     [unroll]
//     for (int i = 0; i < MAX_MOTION_GROUPS; ++i) {
//         RayDesc local_ray = ray;
//         RayPayload local_payload;
//         local_payload.hit = true;
//         // MotionParams motion = motion_groups[i];

//         // float3x3 R;
        
//         // if (!motion.is_static){
//         //     float angle = length(motion.angular_velocity) * t;
//         //     float3 axis = normalize(motion.angular_velocity);
//         //     float c = cos(angle);
//         //     float s = sin(angle);
//         //     R = float3x3(  // actually inverse here
//         //         c + axis.x*axis.x*(1-c),   axis.x*axis.y*(1-c) - axis.z*s, axis.x*axis.z*(1-c) + axis.y*s,
//         //         axis.y*axis.x*(1-c) + axis.z*s, c + axis.y*axis.y*(1-c),   axis.y*axis.z*(1-c) - axis.x*s,
//         //         axis.z*axis.x*(1-c) - axis.y*s, axis.z*axis.y*(1-c) + axis.x*s, c + axis.z*axis.z*(1-c)
//         //     );
            
//         //     local_ray.Origin = mul(R, ray.Origin - motion.pivot_point) + motion.pivot_point - motion.linear_velocity * t;
//         //     local_ray.Direction = mul(R, ray.Direction);
//         // }

//         if(for_shadow_test){
//             switch (i) {
//                 case 0: TraceRay(as, RAY_FLAG_ACCEPT_FIRST_HIT_AND_END_SEARCH | RAY_FLAG_FORCE_OPAQUE | RAY_FLAG_SKIP_CLOSEST_HIT_SHADER, 0xFF, 0, 1, 0, local_ray, local_payload); break;
//                 // case 1: TraceRay(as1, RAY_FLAG_ACCEPT_FIRST_HIT_AND_END_SEARCH | RAY_FLAG_FORCE_OPAQUE | RAY_FLAG_SKIP_CLOSEST_HIT_SHADER, 0xFF, 0, 1, 0, local_ray, local_payload); break;
//                 // case 2: TraceRay(as2, RAY_FLAG_ACCEPT_FIRST_HIT_AND_END_SEARCH | RAY_FLAG_FORCE_OPAQUE | RAY_FLAG_SKIP_CLOSEST_HIT_SHADER, 0xFF, 0, 1, 0, local_ray, local_payload); break;
//                 // case 3: TraceRay(as3, RAY_FLAG_ACCEPT_FIRST_HIT_AND_END_SEARCH | RAY_FLAG_FORCE_OPAQUE | RAY_FLAG_SKIP_CLOSEST_HIT_SHADER, 0xFF, 0, 1, 0, local_ray, local_payload); break;
//             }
//         } else {
//             switch (i) {
//                 case 0: TraceRay(as, RAY_FLAG_NONE, 0xFF, 0, 1, 0, local_ray, local_payload); break;
//                 // case 1: TraceRay(as1, RAY_FLAG_NONE, 0xFF, 0, 1, 0, local_ray, local_payload); break;
//                 // case 2: TraceRay(as2, RAY_FLAG_NONE, 0xFF, 0, 1, 0, local_ray, local_payload); break;
//                 // case 3: TraceRay(as3, RAY_FLAG_NONE, 0xFF, 0, 1, 0, local_ray, local_payload); break;
//             }
//         }
            
//         // if (local_payload.hit && !motion.is_static) {
//         //     R = transpose(R);
//         //     local_payload.hit_point = mul(R, local_payload.hit_point - motion.pivot_point) + motion.pivot_point + motion.linear_velocity * t;
//         //     local_payload.normal = mul(R, local_payload.normal);
//         // }
        
//         if (local_payload.hit) {
//             float t_hit = length(local_payload.hit_point - ray.Origin);
//             if (for_shadow_test) {
//                 payload = local_payload;
//                 return; 
//             } else if (t_hit < best_t && t_hit > ray.TMin) {
//                 best_t = t_hit;
//                 payload = local_payload;
//             }
//         }
//     }
// }

bool test_shadow(inout float3 hit_point, inout float3 normal, inout float3 light_dir, inout float max_distance) {
    RayDesc shadow_ray;
    shadow_ray.Origin = hit_point;
    shadow_ray.Direction = light_dir;
    shadow_ray.TMin = t_min;
    shadow_ray.TMax = max_distance - eps * 3.0; // prevent hit the light
    
    RayPayload shadow_payload;
    
    // Traceray(shadow_ray, shadow_payload, true);
    shadow_payload.hit = true;
    TraceRay(as, 
          RAY_FLAG_ACCEPT_FIRST_HIT_AND_END_SEARCH | 
          RAY_FLAG_FORCE_OPAQUE | 
          RAY_FLAG_SKIP_CLOSEST_HIT_SHADER,
          0xFF, 0, 1, 0, shadow_ray, shadow_payload);
    
    return shadow_payload.hit;
}

// ====================== MIS直接光照计算 ======================
float3 mis_direct_lighting(inout uint light_count, inout float3 hit_point, inout float3 normal, inout Material mat, 
                           inout float3 wo, bool is_flip, inout uint seed) {
    float3 total_light = float3(0, 0, 0);

    if (!is_enable_lights(light_count) || mat.metallic > 0.9) return total_light;

    uint sample_times = min(4, light_count + 1);

    for (int i = 0; i < sample_times; ++i) {
        // 按功率重要性采样光源
        int light_idx = sample_light_by_power(light_count, seed);
        if (light_idx < 0) continue;
        
        Light light;
        if(light_idx < light_count) light = lights[light_idx];
        else light.type = 4;

        LightSample light_sample;

        switch (light.type) {
            case 0: sample_point_light(light, hit_point, seed, light_sample); break;
            case 1: sample_area_light(light, hit_point, seed, light_sample); break;
            case 2: sample_spot_light(light, hit_point, seed, light_sample); break;
            case 3: sample_sphere_light(light, hit_point, seed, light_sample); break;
            case 4: sample_hdr(hit_point, seed, light_sample); break;
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
        float ndotl = max(EPS, abs(dot(normal, wi)));
        float pdf;
        float3 bsdf_val = EvalBSDF(mat, wo, wi, normal, is_flip, pdf);
        // bsdf_val = max( min(bsdf_val, 50.0 * float3(1 / (ndotl + EPS), 1 / (ndotl + EPS), 1 / (ndotl + EPS))), float3(0.0, 0.0, 0.0));
        // pdf = max(pdf, 0.0);

        float3 light_contrib = light_sample.radiance * bsdf_val * ndotl;
        
        if (any(light_contrib > 0.0)) {
            // 计算光源采样PDF
            float light_select_pdf = light_power_weights[light_idx];
            float light_pdf = light_sample.pdf * light_select_pdf;

            if (light.type == 0 || light.type == 2) {
                // 点光源和聚光灯：直接加（delta光源）
                total_light += light_contrib / light_pdf;
            }
            else {
                // 面光源、球光源、HDR
                float mis_weight = mis_balance_weight(light_pdf, pdf);
                total_light += light_contrib * mis_weight / light_pdf;
            }
        }
    }
    return total_light / sample_times;
}

float IntersectRaySphere(inout RayDesc ray, inout float3 sphere_center, float sphere_radius) {
    float3 oc = ray.Origin - sphere_center;
    float a = dot(ray.Direction, ray.Direction);
    float b = dot(oc, ray.Direction);
    float c = dot(oc, oc) - sphere_radius * sphere_radius;
    
    float delta = b * b - a * c;  // divide 2
    
    if (delta < 0.0) {
        return -1.0;
    }

    delta = sqrt(delta);
    float t = (-b - delta) / a;
    
    if (t < ray.TMin) {
        t = (-b + delta) / a;
    }

    return (t >= ray.TMin && t <= ray.TMax) ? t : -1.0;
    
}

void CheckHitSphereLight(inout uint light_count, inout RayDesc ray, inout RayPayload payload, inout uint closest_light_idx) {
    float closest_distance = ray.TMax;
    closest_light_idx = 0xFFFFFFFF;
    
    if (payload.hit) {
        closest_distance = length(payload.hit_point - ray.Origin);
    }
    
    for (int i = 0; i < light_count; ++i) {
        Light light = lights[i];
        
        if (light.type != 3 || !light.visible || !light.enabled) {
            continue;
        }
        
        float t = IntersectRaySphere(ray, light.position, light.radius);
        
        if (t > 0.0 && t < closest_distance) {
            closest_distance = t;
            closest_light_idx = i;
            
            payload.hit = true;
            payload.hit_point = ray.Origin + t * ray.Direction;
        }
    }
}
// ====================== 景深辅助函数 ====================

float2 concentric_sample_disk(inout uint seed) {
    float2 u = random2(seed) * 2.0 - 1.0;
    
    if (u.x == 0.0 && u.y == 0.0) {
        return float2(0.0, 0.0);
    }
    
    float theta, r;
    if (abs(u.x) > abs(u.y)) {
        r = u.x;
        theta = (PI / 4.0) * (u.y / u.x);
    } else {
        r = u.y;
        theta = (PI / 2.0) - (PI / 4.0) * (u.x / u.y);
    }
    
    return r * float2(cos(theta), sin(theta));
}

float3 compute_focal_point(float3 ray_origin, float3 ray_dir, float focal_distance) {
    // 计算光线与焦平面的交点
    // 假设焦平面垂直于相机前向方向
    float3 camera_forward = float3(0.0, 0.0, -1.0); // 相机空间前向
    float3 world_forward = mul((float3x3)camera_info.camera_to_world, camera_forward);
    world_forward = normalize(world_forward);
    
    // 焦平面方程: dot(p - (origin + world_forward * focal_distance), world_forward) = 0
    float3 focal_plane_origin = ray_origin + world_forward * focal_distance;
    float denom = dot(ray_dir, world_forward);
    
    if (abs(denom) > 1e-6) {
        float t = dot(focal_plane_origin - ray_origin, world_forward) / denom;
        return ray_origin + ray_dir * t;
    }
    
    // 平行于焦平面，使用近似值
    return ray_origin + ray_dir * focal_distance;
}

// ====================== 主渲染逻辑 ======================
[shader("raygeneration")]
void RayGenMain() {
    uint2 pixel_coords = DispatchRaysIndex().xy;
    uint seed = generate_seed(pixel_coords, render_setting.frame_count);

    // 生成相机射线（相机空间）
    float2 pixel_center = (float2)DispatchRaysIndex() + 
                         (render_setting.enable_accumulation ? random2(seed) : float2(0.5, 0.5));
    float2 uv = pixel_center / float2(DispatchRaysDimensions().xy);
    uv.y = 1.0 - uv.y;
    
    float2 d = uv * 2.0 - 1.0;
    
    float4 target_camera = mul(camera_info.screen_to_camera, float4(d, 1, 1));
    float3 ray_dir_camera = normalize(target_camera.xyz);
    float3 ray_origin_camera = float3(0, 0, 0); 
    
    // 景深效果：薄透镜相机模型
    if (camera_info.enable_depth_of_field && camera_info.lens_radius > 0.0) {
        
        float3 focal_point_camera = ray_dir_camera * (camera_info.focal_distance / max(-ray_dir_camera.z, EPS));
        float2 lens_sample = concentric_sample_disk(seed) * camera_info.lens_radius;
        
        ray_origin_camera = float3(lens_sample.x, lens_sample.y, 0.0f);
        ray_dir_camera = normalize(focal_point_camera - ray_origin_camera);
    }

    float time_offset = 0.0;
    if (camera_info.enable_motion_blur) {
        time_offset = random(seed) * camera_info.exposure_time;
    }
    
    float angular_offset = camera_info.camera_angular_velocity * time_offset;
    
    float cos_theta = cos(angular_offset);
    float sin_theta = sin(angular_offset);
    float3x3 rotation_matrix = float3x3(
        cos_theta, -sin_theta, 0.0,
        sin_theta,  cos_theta, 0.0,
        0.0,       0.0,       1.0
    );

    ray_origin_camera = mul(rotation_matrix, ray_origin_camera);
    ray_dir_camera = mul(rotation_matrix, ray_dir_camera);

    float4 origin = mul(camera_info.camera_to_world, float4(ray_origin_camera, 1.0));
    float4 direction = mul(camera_info.camera_to_world, float4(ray_dir_camera, 0.0));
    origin.xyz += camera_info.camera_linear_velocity * time_offset;
    
    float3 color = float3(0.0, 0.0, 0.0);
    float3 throughput = float3(1.0, 1.0, 1.0);
    RayDesc ray;
    ray.Origin = origin.xyz;
    ray.Direction = normalize(direction.xyz);
    ray.TMin = t_min;
    ray.TMax = t_max;

    entity_id_output[pixel_coords] = -1;

    float3 prev_hit_point;
    float prev_bsdf_pdf;
    bool prev_is_specular = false;

    uint light_count, light_idx;
    light_count = render_setting.light_count;

    for (int depth = 0; depth < min(render_setting.max_depth, MAX_DEPTH); ++depth) {

        RayPayload payload;
        payload.attribute=float4(-1.0,-1.0,-1.0,-1.0);
        TraceRay(as, RAY_FLAG_NONE, 0xFF, 0, 1, 0, ray, payload);
        // Traceray(ray, payload, 0);

        CheckHitSphereLight(light_count, ray, payload, light_idx);

        Material mat;

        if(light_idx == 0xFFFFFFFF){
            if (!payload.hit) {
                if(render_setting.skybox_texture_id_ >= 0){
                    light_idx = light_count;
                }
                else{ // no hdr
                    color += payload.color * throughput;
                    break;
                }
            }
            else{
                // 记录首次命中的实体ID
                if (depth == 0) {
                    entity_id_output[pixel_coords] = (int)payload.instance_id;
                }
                mat = materials[payload.instance_id];
                light_idx = mat.light_index;
            }
            mat.base_color = payload.color;
            if(payload.attribute.r>=-EPS)
            {
                mat.roughness = payload.attribute.r;
                mat.specular = payload.attribute.g;
                mat.metallic = payload.attribute.b;
                mat.roughness = payload.attribute.a;
            }
        }

        float3 wo = -ray.Direction;
        
        // 处理光源命中
        if (light_idx != 0xFFFFFFFF) {
            Light light;
            if(light_idx < light_count) light = lights[light_idx];
            else{
                light.enabled = true;
                light.type = 4;
                light.color = payload.color;
                light.intensity = 1.0;
            }

            if(light.enabled && (light.type != 1 || dot(wo, light.direction) > 0) ){ // 单向面光源

                if (depth == 0 || prev_is_specular || !render_setting.enable_accumulation) {
                    // 第一次直接击中光源或镜面反射击中光源
                    color += light.color * light.intensity * throughput;
                } else {
                    // BSDF采样击中光源，需要MIS
                    // if (light.type != 1 && light.type != 3) {
                    //     break;
                    // }
                                        
                    // 计算光源采样PDF
                    float light_pdf = 0.0;

                    if(light.type == 4){
                        float luminance = dot(payload.color, float3(0.2126, 0.7152, 0.0722));
                        light_pdf = luminance / hdr_cdf[2];

                    } else{
                        float3 to_light = payload.hit_point - prev_hit_point;
                        float distance = length(to_light);
                        float3 light_dir = to_light / distance;
                        float cos_theta_l = max(EPS, abs(dot(light_dir, light.direction)));
                        
                        if (light.type == 1) { // 面光源
                            float area = light.size.x * light.size.y;
                            light_pdf = (distance * distance) / (cos_theta_l * area);
                        } else if (light.type == 3) { // 球光源
                            float surface_area = 4.0 * PI * light.radius * light.radius;
                            light_pdf = (distance * distance) / (cos_theta_l * surface_area);
                        }
                    }
                    
                    float light_select_pdf = light_power_weights[light_idx];
                    light_pdf *= light_select_pdf;

                    // 计算MIS权重
                    float mis_weight = mis_balance_weight(prev_bsdf_pdf, light_pdf);
                    
                    // 应用MIS
                    color += light.color * light.intensity * throughput * mis_weight; // Here throughut equals to pre_throughput * prev_eval_brdf / prev_bsdf_pdf
                }
                break;
            }
        }

        prev_hit_point = payload.hit_point;

        // MIS直接光照
        if(render_setting.enable_accumulation){
            float3 direct_light = mis_direct_lighting(light_count, payload.hit_point, payload.normal, mat, wo, payload.is_filp, seed);
            color += direct_light * throughput;
        }
        
        // 处理自发光材质
        if (any(mat.emission > 0.0)) {
            color += mat.emission * throughput;
        }
        
        // 俄罗斯轮盘赌终止
        if (depth > 4) {
            float p_survive = min(f3_max(throughput), RR_THRESHOLD);
            if (random(seed) > p_survive) break;
            throughput /= p_survive;
        }
        
        // 采样下一跳方向（BSDF采样）
        float3 wi = float3(0.0, 0.0, 0.0);
        if(!SampleBSDF(mat, wo, payload.normal, wi, payload.is_filp, seed)){
            break;
        }
        //wi = SampleHemisphereCos(random2(seed),payload.normal);

        float pdf;
        float3 bsdf_val = EvalBSDF(mat, wo, wi, payload.normal, payload.is_filp, pdf);

        if(isinf(pdf) || isnan(pdf)){
          color = float3(1e9, 0.0, isnan(pdf)? 1e9: 0);
          break;
        }
        
        // 计算余弦项
        float cos_theta = dot(payload.normal, wi);
        if (cos_theta <= 0.0) {
            payload.normal = - payload.normal;
            cos_theta = - cos_theta;
        }

        bsdf_val *= cos_theta / pdf;

        // prevent extreme cases
        if (pdf < 1e-5 || f3_max(bsdf_val) > 1e3) {
            break;
        } 
        // if (pdf < 2e-6 || f3_max(bsdf_val) > 5e3) {
        //     break;
        // } 
        
        // 更新吞吐量
        throughput *= bsdf_val;
        // throughput = max(throughput, float3(0.0, 0.0, 0.0));

        
        // 记录用于MIS的数据
        prev_bsdf_pdf = pdf;
        
        prev_is_specular = (mat.metallic > 0.9 || mat.clearcoat > 0.0); // simplified
        
        // 准备下一次光线
        ray.Origin = payload.hit_point + payload.normal * eps;
        ray.Direction = normalize(wi);
        ray.TMin = t_min;
        ray.TMax = t_max;
    }

    // color = max(color, float3(0.0, 0.0, 0.0));
    
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
// [shader("miss")]
// void MissMain(inout RayPayload payload) {
//     // 简化的天空渐变
//     float t = 0.5 * (normalize(WorldRayDirection()).y + 1.0);
//     payload.color = lerp(float3(1.0, 1.0, 1.0), float3(0.5, 0.7, 1.0), t) * 0.5;
//     payload.hit = false;
//     payload.instance_id = 0xFFFFFFFF;
// }
[shader("miss")]
void MissMain(inout RayPayload payload) {
    payload.hit = false;
    payload.instance_id = 0xFFFFFFFF;
    
    if (render_setting.skybox_texture_id_ >= 0) {
        // 采样HDR天空盒
        float3 normalizedDir = normalize(WorldRayDirection());
        
        float phi = atan2(normalizedDir.z, normalizedDir.x);
        float theta = acos(clamp(normalizedDir.y, -1.0, 1.0));
        
        // 将球面坐标转换为UV坐标
        float u = (phi + PI) / (2.0 * PI);
        float v = theta / PI;
        
        // 确保UV在[0,1]范围内
        u = frac(u);
        v = clamp(v, 0.0, 1.0);
        
        // 采样HDR纹理（索引0是天空盒纹理）
        payload.color = g_Textures[render_setting.skybox_texture_id_].SampleLevel(g_Sampler, float2(u, v), 0).rgb;
        
    } else {
        // 简化的天空渐变（原有的）
        float t = 0.5 * (normalize(WorldRayDirection()).y + 1.0);
        payload.color = lerp(float3(0.5, 0.7, 1.0), float3(1.0, 1.0, 1.0), t) * 0.8;
    }
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
    float w0 = attr.barycentrics.x;
    float w1 = attr.barycentrics.y;
    float2 object_texcoord = w0 * v1.texcoord + w1 * v2.texcoord + (1.0 - w0 - w1) * v0.texcoord;
    int normal_id = materials[InstanceID()].normal_tex_id;
    int attribute_id = materials[InstanceID()].attribute_tex_id;
    if(normal_id >= 0) {
        float3 norm = g_Textures[normal_id].SampleLevel(g_Sampler, object_texcoord, 0).rgb;
        object_space_normal = normalize(norm * 2.0 - 1.0);
    } else {
    if (length(v0.normal) < 1e-6) {
        // 使用几何法线
        object_space_normal = cross(v1.position - v0.position, v2.position - v0.position);
    } else {
        // 插值顶点法线
        object_space_normal = w0 * v1.normal + w1 * v2.normal + (1.0 - w0 - w1) * v0.normal;
    }
    }
    if(attribute_id >= 0) {
        payload.attribute = g_Textures[attribute_id].SampleLevel(g_Sampler, object_texcoord, 0).rgba;
    }
    object_space_normal = normalize(object_space_normal);
    
    // 转换到世界空间
    float3x3 normal_matrix = (float3x3)transpose(WorldToObject4x3());
    payload.normal = normalize(mul(normal_matrix, object_space_normal));
    
    int texture_id = materials[InstanceID()].texture_id;
    if(texture_id >= 0){
        payload.color = g_Textures[texture_id].SampleLevel(g_Sampler, object_texcoord, 0).rgb;
    }
    else payload.color = materials[InstanceID()].base_color;
    
    // 确保法线朝向射线方向
    payload.is_filp = false;
    if (dot(payload.normal, ray_direction) > 0) {
        payload.normal = -payload.normal;
        payload.is_filp = true;
    }
}