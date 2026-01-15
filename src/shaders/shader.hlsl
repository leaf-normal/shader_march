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
    int enable_dispersion;         // 是否启用光色散
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

    float A, B, C;      // 折射率系数
    int medium_id;
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

    float2 resolution;                 // width * height
    
    int global_medium_enabled;
    float3 global_sigma_a;
    float3 global_sigma_s;
    int offset;
    float3 global_Le;
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

Texture2D<float4> g_MipAtlas[2048] : register(t0, space21);

struct MipInfo{
    uint start;
    uint levels;
};

StructuredBuffer<MipInfo> mip_infos : register(t0, space22);


struct RayDifferential {
    float3 rxOrigin;    // origin for ray at pixel + (1,0)
    float3 ryOrigin;    // origin for ray at pixel + (0,1)
    float3 rxDirection; // direction for ray at pixel + (1,0)
    float3 ryDirection; // direction for ray at pixel + (0,1)
    bool hasDifferentials;
};
struct RayPayload {
  float3 color;
  bool hit;
  uint instance_id;
  float3 normal;
  float3 hit_point;
  bool is_filp;
  float4 attribute;

  RayDifferential diffs;

  float dudx;
  float dudy;
  float dvdx;
  float dvdy;

  float3 dpdx;
  float3 dpdy;
  float3 dndx;
  float3 dndy;

  float3x3 dirCov;  // Cov(L)
  bool hasDirCov; 
};
// ====================== 局部介质 ======================
struct MediumParams{
    float3 sigma_a; //吸收系数
    float3 sigma_s; //散射系数
    float3 Le;      //发光密度
    int enabled;
};
StructuredBuffer<MediumParams> media : register(t0, space23);


// ====================== 常量定义 ======================
#define MAX_DEPTH 8
#define RR_THRESHOLD 0.95f
#define t_min 2e-3
#define t_max 10000.0
#define eps 5e-4 // used for geometry
#define EPS 1e-6 // used for division
#define PI 3.14159265359
#define TWO_PI 6.28318530718
#define INV_PI 0.31830988618
#define HDR_DIS 150
#define MIPMAP_INTENS 1.0

// from 380 nm to 780 nm, step = 5 nm
#define SPECTRAL_SAMPLE_COUNT 81
static const float3 SPECTRAL_TABLE[81] = {  // normalized
    {0.00006401, 0.00000182, 0.00030180},
    {0.00010463, 0.00000299, 0.00049365},
    {0.00019854, 0.00000561, 0.00093816},
    {0.00035795, 0.00001015, 0.00169431},
    {0.00066958, 0.00001853, 0.00317478},
    {0.00108509, 0.00002995, 0.00515639},
    {0.00203589, 0.00005662, 0.00970449},
    {0.00363240, 0.00010201, 0.01737357},
    {0.00628781, 0.00018717, 0.03020840},
    {0.01004935, 0.00034158, 0.04861840},
    {0.01328403, 0.00054278, 0.06483389},
    {0.01537092, 0.00078797, 0.07594025},
    {0.01629645, 0.00107621, 0.08174703},
    {0.01628616, 0.00139439, 0.08340999},
    {0.01573121, 0.00177808, 0.08291915},
    {0.01491237, 0.00224600, 0.08160853},
    {0.01360689, 0.00280750, 0.07810387},
    {0.01174928, 0.00345790, 0.07150163},
    {0.00914114, 0.00425711, 0.06025022},
    {0.00664903, 0.00526874, 0.04875175},
    {0.00447511, 0.00650498, 0.03803891},
    {0.00271155, 0.00792183, 0.02883274},
    {0.00149779, 0.00973360, 0.02176633},
    {0.00068783, 0.01210032, 0.01653133},
    {0.00022928, 0.01511371, 0.01272721},
    {0.00011230, 0.01905824, 0.00993377},
    {0.00043516, 0.02353621, 0.00740237},
    {0.00136162, 0.02845869, 0.00522658},
    {0.00296048, 0.03322208, 0.00366141},
    {0.00512832, 0.03711515, 0.00267880},
    {0.00774395, 0.04033441, 0.00197272},
    {0.01056312, 0.04280736, 0.00139625},
    {0.01358817, 0.04463925, 0.00094986},
    {0.01683081, 0.04586987, 0.00062700},
    {0.02028165, 0.04655537, 0.00040942},
    {0.02395946, 0.04679166, 0.00026905},
    {0.02781739, 0.04655771, 0.00018249},
    {0.03174317, 0.04579032, 0.00012868},
    {0.03565960, 0.04454566, 0.00009826},
    {0.03942161, 0.04283309, 0.00008422},
    {0.04287481, 0.04070875, 0.00007721},
    {0.04578990, 0.03819604, 0.00006551},
    {0.04802184, 0.03542129, 0.00005147},
    {0.04944430, 0.03251553, 0.00004679},
    {0.04970165, 0.02952554, 0.00003743},
    {0.04892491, 0.02652152, 0.00002807},
    {0.04691289, 0.02353621, 0.00001591},
    {0.04390889, 0.02064448, 0.00001123},
    {0.03998076, 0.01782762, 0.00000889},
    {0.03515893, 0.01502012, 0.00000468},
    {0.03005869, 0.01239979, 0.00000234},
    {0.02535617, 0.01015379, 0.00000140},
    {0.02095779, 0.00818854, 0.00000094},
    {0.01688228, 0.00646661, 0.00000047},
    {0.01326531, 0.00500671, 0.00000000},
    {0.01023324, 0.00381820, 0.00000000},
    {0.00771587, 0.00285429, 0.00000000},
    {0.00567110, 0.00208597, 0.00000000},
    {0.00408955, 0.00149733, 0.00000000},
    {0.00297592, 0.00108557, 0.00000000},
    {0.00218843, 0.00079546, 0.00000000},
    {0.00153943, 0.00055776, 0.00000000},
    {0.00106216, 0.00038416, 0.00000000},
    {0.00074117, 0.00026779, 0.00000000},
    {0.00053151, 0.00019194, 0.00000000},
    {0.00037952, 0.00013705, 0.00000000},
    {0.00027094, 0.00009784, 0.00000000},
    {0.00019229, 0.00006944, 0.00000000},
    {0.00013566, 0.00004899, 0.00000000},
    {0.00009588, 0.00003463, 0.00000000},
    {0.00006738, 0.00002433, 0.00000000},
    {0.00004679, 0.00001690, 0.00000000},
    {0.00003229, 0.00001166, 0.00000000},
    {0.00002227, 0.00000804, 0.00000000},
    {0.00001555, 0.00000561, 0.00000000},
    {0.00001099, 0.00000397, 0.00000000},
    {0.00000777, 0.00000281, 0.00000000},
    {0.00000549, 0.00000198, 0.00000000},
    {0.00000389, 0.00000140, 0.00000000},
    {0.00000275, 0.00000099, 0.00000000},
    {0.00000194, 0.00000070, 0.00000000}
};
static const float WAVELENGTH_CDF[81] = {0.00012255, 0.00032297, 0.00070374, 0.00139121, 0.00267884, 0.00476932, 0.00870165, 0.01573765, 0.02796544, 0.04763521, 0.07385545, 0.10455516, 0.13759506, 0.17129190, 0.20476805, 0.23769035, 0.26919644, 0.29809938, 0.32264886, 0.34287204, 0.35921171, 0.37236708, 0.38336632, 0.39313948, 0.40249621, 0.41219765, 0.42265556, 0.43433786, 0.44761918, 0.46259327, 0.47927696, 0.49753254, 0.51725830, 0.53836752, 0.56078301, 0.58445640, 0.60930892, 0.63519631, 0.66196416, 0.68941046, 0.71729738, 0.74531453, 0.77314607, 0.80048160, 0.82690314, 0.85206131, 0.87554965, 0.89707118, 0.91634361, 0.93307152, 0.94722513, 0.95906225, 0.96877800, 0.97656112, 0.98265180, 0.98733561, 0.99085900, 0.99344469, 0.99530698, 0.99666081, 0.99765544, 0.99835450, 0.99883661, 0.99917293, 0.99941408, 0.99958627, 0.99970920, 0.99979644, 0.99985799, 0.99990149, 0.99993206, 0.99995329, 0.99996794, 0.99997805, 0.99998510, 0.99999009, 0.99999362, 0.99999611, 0.99999787, 0.99999912, 1.00000000 };
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

int sample_light_by_power(inout uint light_count, inout uint seed) {
    
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
    
    sample.radiance = light.color * light.intensity / (distance * distance + 1e-6);
    
    sample.pdf = 1;
    sample.light_index = -1;
    sample.valid = true;
}

void sample_area_light(inout Light light, inout float3 hit_point, inout uint seed, out LightSample sample) {
    sample = (LightSample)0;
    
    if (!light.enabled || light.size.x <= 0.0 || light.size.y <= 0.0) return;
    
    float3 bitangent = normalize(cross(light.direction, light.tangent));

    float3 local_pos = float3((random(seed) - 0.5) * light.size.x, 
                              (random(seed) - 0.5) * light.size.y, 
                              0.0);
    
    sample.position = light.position + 
                     local_pos.x * light.tangent + 
                     local_pos.y * bitangent;
    
    float3 to_light = sample.position - hit_point;
    float distance = length(to_light);
    sample.direction = to_light / distance;
    
    float cos_theta_l = dot(-sample.direction, light.direction);
    if (cos_theta_l <= 0.0) {
        sample.valid = 0;
        return;
    }
    
    float area = light.size.x * light.size.y;
    
    sample.radiance = light.color * light.intensity;
    
    float solid_angle_pdf = (distance * distance) / (cos_theta_l * area);
    sample.pdf = solid_angle_pdf;
    
    sample.light_index = -1;
    sample.valid = (sample.pdf > 0.0);
}

void sample_sphere_light(inout Light light, inout float3 hit_point, inout uint seed, out LightSample sample) {
    sample = (LightSample)0;
    
    if (!light.enabled || light.radius <= 0.0) return;
    
    float z = 2.0 * random(seed) - 1.0;
    float phi = TWO_PI * random(seed);
    float r = sqrt(max(0.0, 1.0 - z * z));
    
    float3 sphere_dir = float3(r * cos(phi), r * sin(phi), z);
    
    sample.position = light.position + light.radius * sphere_dir;
    
    float3 to_light = sample.position - hit_point;
    float distance = length(to_light);
    sample.direction = to_light / distance;
    
    float3 sphere_normal = sphere_dir;
    float cos_theta_l = max(0.0, dot(-sample.direction, sphere_normal));
    if (cos_theta_l <= 0.0) {
        return;
    }
    
    float surface_area = 4.0 * PI * light.radius * light.radius;
    
    sample.radiance = light.color * light.intensity;
    
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
    
    float cos_angle = dot(-sample.direction, light.direction);
    float cos_cutoff = cos(radians(light.cone_angle * 0.5));

    if (cos_angle < cos_cutoff) {
        return; 
    }
    
    float epsilon = 0.1 * (1.0 - cos_cutoff);
    float spot_factor = smoothstep(cos_cutoff, cos_cutoff + epsilon, cos_angle);
    
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
        uint mid = (left + right) >> 1; 
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
        uint width = (uint)hdr_cdf[0];
    uint height = (uint)hdr_cdf[1];
    float hdr_total_luminance = hdr_cdf[2];
    
    uint cdf_offset = 3; 
    
    uint marginal_cdf_start = cdf_offset + (uint)(width * height);
    uint row = binary_search_cdf(random(seed), marginal_cdf_start, height);
    
    uint conditional_cdf_start = cdf_offset + row * (uint)width;
    uint col = binary_search_cdf(random(seed), conditional_cdf_start, width);
    
    float u = (float(col) + 0.5) / width;
    float v = (float(row) + 0.5) / height;
    
    float phi = u * TWO_PI - PI;
    float theta = v * PI;
    
    float sin_theta = sin(theta);
    float cos_theta = cos(theta);
    float sin_phi = sin(phi);
    float cos_phi = cos(phi);
    
    float3 direction = float3(sin_theta * cos_phi, cos_theta, sin_theta * sin_phi);
    
    float3 hdr_color = g_Textures[render_setting.skybox_texture_id_].SampleLevel(g_Sampler, float2(u, v), 0).rgb;
    float luminance = dot(hdr_color, float3(0.2126, 0.7152, 0.0722));
    
    float solid_angle_pdf = luminance / hdr_total_luminance;
    
    sample.position = hit_point + direction * 10000.0; 
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

// ====================== BSDF辅助函数 ======================

float f3_max(float3 u){
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
    return G / (4.0 * NdotV * NdotL + EPS);
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

void GetTangent(in float3 normal, out float3 tangent, out float3 bitangent)
{
  float3 up=abs(normal.z)<0.999?float3(0.0,0.0,1.0):float3(1.0,0.0,0.0);
  tangent=normalize(cross(up,normal));
  bitangent=cross(normal,tangent);
}

float2 SampleDisk(float2 u) {
    float r = sqrt(u.x);
    float theta = 2.0 * PI * u.y;
    return float2(r * cos(theta), r * sin(theta));
}
float3 SampleGGX_VNDF_TangentSpace(float3 Ve, float roughness, float2 u) {
    float3 Vh = normalize(float3(roughness * Ve.x, roughness * Ve.y, Ve.z));    
    float lensq = Vh.x * Vh.x + Vh.y * Vh.y;
    float3 T1 = lensq > 0.0 ? float3(-Vh.y, Vh.x, 0.0) * rsqrt(lensq) : float3(1.0, 0.0, 0.0);
    float3 T2 = cross(Vh, T1);
    float r = sqrt(u.x);
    float phi = 2.0 * PI * u.y;
    float t1 = r * cos(phi);
    float t2 = r * sin(phi);
    float s = 0.5 * (1.0 + Vh.z);
    t2 = (1.0 - s) * sqrt(1.0 - t1 * t1) + s * t2;
    float3 Nh = t1 * T1 + t2 * T2 + sqrt(max(0.0, 1.0 - t1 * t1 - t2 * t2)) * Vh;
    float3 Ne = normalize(float3(roughness * Nh.x, roughness * Nh.y, max(0.0, Nh.z)));
    return Ne;
}

float3 SampleGGX_VNDF(float3 V, float roughness, float2 u, float3 N, float3 T, float3 B) {
    float3 Ve = float3(dot(V, T), dot(V, B), dot(V, N));
    float3 H_t = SampleGGX_VNDF_TangentSpace(Ve, roughness, u);
    return normalize(H_t.x * T + H_t.y * B + H_t.z * N);
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

// ====================== Adaptive mipmap 辅助函数 =============== 
float4 SampleTexture2D_Lod(int texId, float2 uv, float lodf)
{
    if (texId < 0) return float4(0,0,0,0);

    MipInfo info = mip_infos[texId];
    int levels = int(info.levels);

    int lod0 = clamp(int(floor(lodf)), 0, levels - 1);
    int lod1 = clamp(lod0 + 1, 0, levels - 1);
    float t = saturate(lodf - float(lod0));

    uint idx0 = info.start + lod0;
    uint idx1 = info.start + lod1;
    float4 c0 = g_MipAtlas[idx0].SampleLevel(g_Sampler, uv, 0);
    float4 c1 = g_MipAtlas[idx1].SampleLevel(g_Sampler, uv, 0);

    return lerp(c0, c1, t);
}

bool Solve2x2(float a00, float a01, float a10, float a11, float2 b, out float2 out_x)
{
    float det = a00 * a11 - a01 * a10;
    if (abs(det) < EPS) {
        a00 += EPS; a11 += EPS;
        det = a00 * a11 - a01 * a10;
        if (abs(det) < EPS) {
            out_x = float2(0.0, 0.0);
            return false;
        }
    }
    float invDet = 1.0 / det;
    out_x.x = ( a11 * b.x - a01 * b.y) * invDet;
    out_x.y = (-a10 * b.x + a00 * b.y) * invDet;
    return true;
}

float2 SolveForCoeffs3x2(float3 e1, float3 e2, float3 b)
{
    float a00 = dot(e1, e1);
    float a01 = dot(e1, e2);
    float a10 = a01;
    float a11 = dot(e2, e2);
    float2 rhs = float2(dot(e1, b), dot(e2, b));
    float2 sol;
    if (!Solve2x2(a00, a01, a10, a11, rhs, sol)) {
        if (a00 >= a11 && a00 > 0.0) {
            sol.x = rhs.x / (a00 + 1e-8);
            sol.y = 0.0;
        } else if (a11 > 0.0) {
            sol.x = 0.0;
            sol.y = rhs.y / (a11 + 1e-8);
        } else {
            sol = float2(0.0, 0.0);
        }
    }
    return sol;
}

// ====================== Cov 矩阵计算 ==============================
float3x3 ReflectionJacobian_wrt_Normal(float3 d, float3 n_m)
{
    float nd = dot(n_m, d);
    float3x3 nm_dt = float3x3(
        n_m.x * d.x, n_m.x * d.y, n_m.x * d.z,
        n_m.y * d.x, n_m.y * d.y, n_m.y * d.z,
        n_m.z * d.x, n_m.z * d.y, n_m.z * d.z
    );
    float3x3 I = float3x3(1,0,0, 0,1,0, 0,0,1);
    float3x3 J = -2.0 * (nm_dt + nd * I);
    return J;
}

float3x3 ReflectionJacobian_wrt_Dir(float3 v, float3 n)
{
    // 外积 n n^T
    float3x3 nnT = float3x3(
        n.x * n.x, n.x * n.y, n.x * n.z,
        n.y * n.x, n.y * n.y, n.y * n.z,
        n.z * n.x, n.z * n.y, n.z * n.z
    );

    float3x3 I = float3x3(1,0,0, 0,1,0, 0,0,1);

    float3x3 J = I - 2.0 * nnT;
    return J;
}

float3x3 RefractionJacobian_wrt_Normal(float3 v, float3 n_m, float eta)
{
    float c = dot(v, n_m);
    float eta2 = eta * eta;
    float k = 1.0 - eta2 * (1.0 - c * c);

    if (k <= 0.0) {
        return float3x3(0,0,0, 0,0,0, 0,0,0);
    }

    float s = sqrt(k);

    float coeff = eta + (eta2 * c) / max(s, 1e-8);
    float3 dA_dn = coeff * v;

    float A = eta * c + s;

    float3x3 nm_dA = float3x3(
        n_m.x * dA_dn.x, n_m.x * dA_dn.y, n_m.x * dA_dn.z,
        n_m.y * dA_dn.x, n_m.y * dA_dn.y, n_m.y * dA_dn.z,
        n_m.z * dA_dn.x, n_m.z * dA_dn.y, n_m.z * dA_dn.z
    );

    float3x3 I = float3x3(1,0,0, 0,1,0, 0,0,1);

    float3x3 J = -(nm_dA + A * I);
    return J;
}

float3x3 RefractionJacobian_wrt_Dir(float3 v, float3 n, float eta)
{
    float c    = dot(v, n);
    float eta2 = eta * eta;
    float k    = 1.0 - eta2 * (1.0 - c * c);

    if (k <= 0.0) {
        return float3x3(
            1.0, 0.0, 0.0,
            0.0, 1.0, 0.0,
            0.0, 0.0, 1.0
        );
    }

    float s = sqrt(k);

    // coeff = η + η² c / sqrt(k)
    float coeff = eta + (eta2 * c) / max(s, 1e-8);

    float3x3 nnT = float3x3(
        n.x * n.x, n.x * n.y, n.x * n.z,
        n.y * n.x, n.y * n.y, n.y * n.z,
        n.z * n.x, n.z * n.y, n.z * n.z
    );

    float3x3 I = float3x3(1,0,0, 0,1,0, 0,0,1);

    // J_T_v = η I - coeff * (n n^T)
    float3x3 J = eta * I - coeff * nnT;
    return J;
}


float3x3 MicroNormalCovariance(float3 n, float sigma_n)
{
    float3x3 I = float3x3(1,0,0, 0,1,0, 0,0,1);
    // n n^T
    float3x3 nnT = float3x3(
        n.x*n.x, n.x*n.y, n.x*n.z,
        n.y*n.x, n.y*n.y, n.y*n.z,
        n.z*n.x, n.z*n.y, n.z*n.z
    );
    float3x3 cov = sigma_n * sigma_n * (I - nnT);
    return cov;
}

float3x3 MicroNormalCovariance_Aniso(float3 T, float3 B, float alpha_x, float alpha_y)
{
    float3x3 TT = float3x3(
        T.x*T.x, T.x*T.y, T.x*T.z,
        T.y*T.x, T.y*T.y, T.y*T.z,
        T.z*T.x, T.z*T.y, T.z*T.z
    );
    float3x3 BB = float3x3(
        B.x*B.x, B.x*B.y, B.x*B.z,
        B.y*B.x, B.y*B.y, B.y*B.z,
        B.z*B.x, B.z*B.y, B.z*B.z
    );
    float3x3 cov = (alpha_x*alpha_x) * TT + (alpha_y*alpha_y) * BB;
    cov[0][0] += EPS; cov[1][1] += EPS; cov[2][2] += EPS;
    return cov;
}


// Cov_L = J * Cov_n * J^T
float3x3 PropagateNormalCovToDirectionCov(float3x3 J_L_n, float3x3 Cov_n)
{
    // temp = J * Cov_n
    float3x3 temp;
    // matrix multiply 3x3
    [unroll] for (int r = 0; r < 3; ++r) {
        [unroll] for (int c = 0; c < 3; ++c) {
            float s = 0.0;
            [unroll] for (int k = 0; k < 3; ++k) {
                s += J_L_n[r][k] * Cov_n[k][c];
            }
            temp[r][c] = s;
        }
    }
    // Cov_L = temp * J^T
    float3x3 CovL;
    [unroll] for (int r = 0; r < 3; ++r) {
        [unroll] for (int c = 0; c < 3; ++c) {
            float s = 0.0;
            [unroll] for (int k = 0; k < 3; ++k) {
                s += temp[r][k] * J_L_n[c][k]; // J^T: swap indices
            }
            CovL[r][c] = s;
        }
    }
    return CovL;
}

void DirectionCovToLatLongUVCov(float3 L, float3x3 CovL, out float2x2 CovUV)
{
    float lx = L.x;
    float ly = L.y;
    float lz = L.z;
    float denom_xz = lx*lx + lz*lz;
    denom_xz = max(denom_xz, 1e-12);

    // ∂u/∂L = 1/(2π) * [ -lz/(lx^2+lz^2), 0, lx/(lx^2+lz^2) ]
    float inv2pi = 1.0 / (2.0 * PI);
    float3 du_dL = inv2pi * float3(-lz / denom_xz, 0.0, lx / denom_xz);

    // ∂v/∂L = -1/(π * sqrt(1 - ly^2)) * [0, 1, 0]
    float denom_v = sqrt(max(1.0 - ly*ly, 1e-12));
    float dv_dy = -1.0 / (PI * denom_v);
    float3 dv_dL = float3(0.0, dv_dy, 0.0);

    // Cov_uv = [du; dv] * CovL * [du; dv]^T
    // compute intermediate: a = CovL * du_dL, b = CovL * dv_dL
    float3 a = float3(
        CovL[0][0]*du_dL.x + CovL[0][1]*du_dL.y + CovL[0][2]*du_dL.z,
        CovL[1][0]*du_dL.x + CovL[1][1]*du_dL.y + CovL[1][2]*du_dL.z,
        CovL[2][0]*du_dL.x + CovL[2][1]*du_dL.y + CovL[2][2]*du_dL.z
    );
    float3 b = float3(
        CovL[0][0]*dv_dL.x + CovL[0][1]*dv_dL.y + CovL[0][2]*dv_dL.z,
        CovL[1][0]*dv_dL.x + CovL[1][1]*dv_dL.y + CovL[1][2]*dv_dL.z,
        CovL[2][0]*dv_dL.x + CovL[2][1]*dv_dL.y + CovL[2][2]*dv_dL.z
    );

    float c00 = dot(du_dL, a); // var(u)
    float c01 = dot(du_dL, b); // cov(u,v)
    float c11 = dot(dv_dL, b); // var(v)

    // pack into 2x2
    CovUV = float2x2(c00, c01, c01, c11);
}

float ComputeEnvmapLOD_FromCovUV(int texture_id, float2x2 CovUV, out float2 axis_lengths, out float2x2 cov_texel)
{
    uint w, h;
    g_Textures[texture_id].GetDimensions(w, h);
    float2 texSize = float2(max(1u,w), max(1u,h));

    // Convert uv covariance to texel covariance: cov_texel = diag(texSize) * CovUV * diag(texSize)
    float2x2 D = float2x2(texSize.x, 0.0, 0.0, texSize.y);
    // cov_texel = D * CovUV * D
    float2x2 temp = float2x2(
        D[0][0]*CovUV[0][0] + D[0][1]*CovUV[1][0], D[0][0]*CovUV[0][1] + D[0][1]*CovUV[1][1],
        D[1][0]*CovUV[0][0] + D[1][1]*CovUV[1][0], D[1][0]*CovUV[0][1] + D[1][1]*CovUV[1][1]
    );
    cov_texel = float2x2(
        temp[0][0]*D[0][0] + temp[0][1]*D[1][0], temp[0][0]*D[0][1] + temp[0][1]*D[1][1],
        temp[1][0]*D[0][0] + temp[1][1]*D[1][0], temp[1][0]*D[0][1] + temp[1][1]*D[1][1]
    );

    // eigenvalues of cov_texel (2x2)
    float c00 = cov_texel[0][0];
    float c01 = cov_texel[0][1];
    float c11 = cov_texel[1][1];
    float trace = c00 + c11;
    float det = c00 * c11 - c01 * c01;
    float disc = max(0.0, trace * trace * 0.25 - det);
    float sqrt_disc = sqrt(disc);
    float lambda1 = trace * 0.5 + sqrt_disc;
    float lambda2 = trace * 0.5 - sqrt_disc;
    lambda1 = max(lambda1, 0.0);
    lambda2 = max(lambda2, 0.0);
    float r1 = sqrt(lambda1);
    float r2 = sqrt(lambda2);
    axis_lengths = float2(r1, r2);

    float rho = max(r1, r2);
    rho = max(rho, 1e-6);
    float lod = log2(rho);
    return lod;
}

// ========================== BSDF 主函数 =============================================

bool SampleBSDF(inout Material mat, inout float3 ray, inout float3 normal, out float3 wi, bool is_flip, inout uint seed){

  //lobe权重
  float diffuseweight=(1.0-mat.metallic)*(1.0-mat.transparency);//漫反射
  float specularweight=lerp(0.04, 1.0, mat.metallic)*(1.0-mat.transparency);
  float transmissionweight=mat.transparency*(1.0-mat.metallic);//透射
  float clearcoatweight=0.25*mat.clearcoat*(1.0-mat.transparency);//清漆层
  float sheenweight=mat.sheen*(1.0-mat.metallic)*(1.0-mat.transparency);//光泽层
  //normalization
  float total=diffuseweight+specularweight+transmissionweight+sheenweight;//+clearcoatweight;
  diffuseweight/=total;
  specularweight/=total;
  transmissionweight/=total;
  sheenweight/=total;
  //clearcoatweight/=total;
  float randLobe=random(seed)*(1+clearcoatweight);
  float3 tangent, bitangent;
  GetTangent(normal, tangent, bitangent);  
  if(randLobe<diffuseweight+sheenweight)//漫反射+光泽
  {
    wi=SampleHemisphereCos(random2(seed),normal);
  }else if(randLobe<diffuseweight+specularweight+sheenweight)//镜面反射
  {
    float3 H;
    H=SampleGGX_Anisotropic(ray,mat.roughness,mat.anisotropic,random2(seed),normal,tangent); // replace
    wi=reflect(-ray,H);
    if(dot(normal,wi)<=0.0)return 0;
  }else if(randLobe<diffuseweight+specularweight+transmissionweight+sheenweight)//透射
  {
    float3 H;
    float3 temp_wi;
    bool valid_sample = false;
    for(int i = 0; i < 3; i++) {
        H = SampleGGX_VNDF(ray, mat.roughness, random2(seed), normal, tangent, bitangent);
        float eta = (!is_flip) ? 1.0/mat.ior : mat.ior;        
        if (Refract(-ray, H, eta, temp_wi)) {
            valid_sample = true;
            wi = temp_wi;
            break; 
        } else {
            temp_wi = reflect(-ray, H);
            if (dot(normal, temp_wi) > 0.0) {
                valid_sample = true;
                wi = temp_wi;
                break;
            }
        }
    }
    if (!valid_sample) return 0;
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
//   float3 F0=lerp(0.08*mat.specular,mat.base_color,mat.metallic);
  float F0_ior = pow((mat.ior - 1.0) / (mat.ior + 1.0), 2.0);
  float3 F0_dieletric = float3(F0_ior, F0_ior, F0_ior);
  float3 F0 = lerp(F0_dieletric, mat.base_color, mat.metallic);

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
  float ax=max(EPS,alpha/sqrt(1.0-0.9*mat.anisotropic));
  float ay=max(EPS,alpha*sqrt(1.0-0.9*mat.anisotropic));
  float G1_V=SmithG_GGX_Anisotropic(Ndotray, dot(ray, tangent), dot(ray, bitangent), ax, ay);
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
  float total=diffuseweight+specularweight+transmissionweight+sheenweight;//+clearcoatweight;
  diffuseweight/=total;
  specularweight/=total;
  transmissionweight/=total;
  sheenweight/=total;
  //clearcoatweight/=total;
  if(is_trans)//透射
  {
    float n1 = is_flip ? mat.ior : 1.0; 
    float n2 = is_flip ? 1.0 : mat.ior;
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
    float D = GTR2(NdotH, mat.roughness);
    float G1V = SmithG_GGX(abs(NdotV), mat.roughness);
    float G1L = SmithG_GGX(abs(NdotL), mat.roughness);
    float G = G1V * G1L;
    float3 F_color = SchlickFresnel(F0, abs(VdotH)); 
    float3 T_color = 1.0 - F_color;
    float sqrtDenom = n1 * VdotH + n2 * LdotH;
    float commonDenom = sqrtDenom * sqrtDenom + EPS;
    float factor = (abs(VdotH) * abs(LdotH) * n2 * n2) / (abs(NdotV) * abs(NdotL) * commonDenom);
    // float shit = 0.85 / (0.6 / (exp( (- min(Ndotray,0.5) + 0.18) * 12) + 1) + 0.18);
    ret += mat.base_color * T_color * D * G * factor * transmissionweight ;
    // ret += mat.base_color * T_color * D * G * factor * transmissionweight;  // 正常版本
    float jacobian = (n2 * n2 * abs(LdotH)) / commonDenom;
    float pdf_h_vndf = (D * G1V * abs(VdotH)) / (abs(NdotV) + EPS);
    transmissionPDF = pdf_h_vndf * jacobian;
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
    float D=GTR2_Anisotropic(NdotH,dot(H,tangent),dot(H,bitangent),ax,ay);
    float Vis=SmithG_GGX_Correlated_Anisotropic(Ndotray,Ndotwi,dot(ray,tangent), dot(ray, bitangent), dot(wi, tangent), dot(wi, bitangent), ax, ay);
    float3 spec=D*Vis*F; 
    if (mat.specular_tint > EPS)
    {
        float3 tint=lerp(float3(1.0, 1.0, 1.0), mat.base_color, mat.specular_tint);
        spec*=tint;
    }
    ret+=spec*specularweight;
    float aspect_eff = sqrt(1.0 - 0.9 * mat.anisotropic);
    specularPDF=(D*G1_V)/(4.0*Ndotray+EPS);
    // 全反射
    if(transmissionweight > EPS)
    {
        float eta = (!is_flip) ? 1.0/mat.ior : mat.ior;
        float c = dot(ray, H);
        float k = 1.0 - eta * eta * (1.0 - c * c);
        if (k < 0.0)
        {
            float3 tir_term = D * Vis * float3(1.0, 1.0, 1.0);
            ret+=tir_term * transmissionweight;
            transmissionPDF = specularPDF;
        }
    }
    //光泽层
    if (mat.sheen > EPS && mat.metallic < 1.0 && mat.transparency < 1.0)
    {
        float3 base = mat.base_color;
        float lum = max(0.001, 0.2126*base.x + 0.7152*base.y + 0.0722*base.z);
        float3 tintColor = normalize(base / lum);
        float3 sheenColor = lerp(float3(1.0,1.0,1.0), tintColor, mat.sheen_tint);
        float w = pow(1.0 - abs(Ndotray), 5.0);
        float3 sheenTerm = sheenColor * mat.sheen * w;
        float F_avg = (F.x + F.y + F.z) / 3.0;
        ret += sheenweight * sheenTerm * (1.0 - F_avg) * 0.5;
        sheenPDF = diffusePDF;
    }

    //清漆层
    if (mat.clearcoat > EPS && mat.transparency < 1.0)
    {
        float alpha_c = max(EPS, mat.clearcoat_roughness);
        float Dc = GTR2(NdotH, alpha_c);
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

// ===================== 全局均匀介质 =====================
struct MediumState {
    bool  enabled;
    float3 sigma_a;
    float3 sigma_s;
    float3 sigma_t;  // sigma_a + sigma_s
    float3 Le;
};

MediumState GetCurrentMedium()
{
    MediumState m;
    if (render_setting.global_medium_enabled != 0) {
        m.enabled = true;
        m.sigma_a = max(render_setting.global_sigma_a, EPS);
        m.sigma_s = max(render_setting.global_sigma_s, EPS);
        m.sigma_t = m.sigma_a + m.sigma_s;
        m.Le      = render_setting.global_Le;
    } else {
        m.enabled = false;
        m.sigma_a = 0.0;
        m.sigma_s = 0.0;
        m.sigma_t = 0.0;
        m.Le      = 0.0;
    }
    return m;
}

// ====================== 阴影测试 ======================
bool test_shadow(float3 hit_point, float3 normal, float3 light_dir, float max_distance, out float3 transmittance) {
    RayDesc shadow_ray;
    shadow_ray.Origin = hit_point;
    shadow_ray.Direction = light_dir;
    shadow_ray.TMin = t_min;
    shadow_ray.TMax = max_distance - eps * 3.0; // prevent hit the light
    
    RayPayload shadow_payload;
    
    shadow_payload.hit = true;
    TraceRay(as, 
          RAY_FLAG_ACCEPT_FIRST_HIT_AND_END_SEARCH | 
          RAY_FLAG_FORCE_OPAQUE | 
          RAY_FLAG_SKIP_CLOSEST_HIT_SHADER,
          0xFF, 0, 1, 0, shadow_ray, shadow_payload);
    
    MediumState medium = GetCurrentMedium();

    float3 sigma_t = medium.sigma_t;
    float3 T = exp(-sigma_t * min(max_distance, HDR_DIS));
    transmittance = T;

    return shadow_payload.hit;
}

// ====================== MIS直接光照计算 ======================
float3 mis_direct_lighting(uint light_count, float3 hit_point, float3 normal, inout Material mat, float3 wo, bool is_flip, inout uint seed) {
    float3 total_light = float3(0, 0, 0);

    if (!is_enable_lights(light_count) || (length(normal) > EPS && (mat.roughness < EPS))){
        return total_light;
    }

    uint sample_times = min(4, light_count + 1);

    for (int i = 0; i < sample_times; ++i) {
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

        float light_distance = length(light_sample.position - hit_point);
        
        float3 transmittance;
        if (test_shadow(hit_point, normal, light_sample.direction, light_distance, transmittance)) {
            continue;
        }
        
        float3 wi = light_sample.direction, light_contrib;
        float pdf;
        if(length(normal) < EPS){
            pdf = 1 / (4 * PI);
            light_contrib = light_sample.radiance * transmittance * pdf;
        }
        else{
            float ndotl = max(EPS, abs(dot(normal, wi)));
            float3 bsdf_val = EvalBSDF(mat, wo, wi, normal, is_flip, pdf);
            light_contrib = light_sample.radiance * bsdf_val * ndotl * transmittance;
        }
       
        if (any(light_contrib > 0.0)) {
            float light_select_pdf = light_power_weights[light_idx];
            float light_pdf = light_sample.pdf * light_select_pdf;
            

            if (light.type == 0 || light.type == 2) {
                total_light += light_contrib / light_pdf;
            }
            else {
                float mis_weight = mis_balance_weight(light_pdf, pdf);
                total_light += light_contrib * mis_weight / light_pdf;
            }
        }
    }
    return total_light / sample_times;
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

// ====================== 光线微分生成 ======================
void GenerateCameraRaysWithDifferentials(
    uint2 pixel, 
    float2 pixel_sample_offset, 
    float2 resolution,
    float2 lens_sample,
    out RayDesc outRay, 
    out RayDifferential outDiffs)
{
    float2 invRes = 1.0 / resolution;
    float2 p_center = (float2(pixel) + pixel_sample_offset) * invRes;
    float2 p_rx_raw = (float2(pixel + uint2(1, 0)) + pixel_sample_offset) * invRes;
    float2 p_ry_raw = (float2(pixel + uint2(0, 1)) + pixel_sample_offset) * invRes;

    float2 p_rx = p_center + (p_rx_raw - p_center) * MIPMAP_INTENS;
    float2 p_ry = p_center + (p_ry_raw - p_center) * MIPMAP_INTENS;
    
    p_center = float2(p_center.x * 2.0 - 1.0, 1.0 - p_center.y * 2.0);
    p_rx = float2(p_rx.x * 2.0 - 1.0, 1.0 - p_rx.y * 2.0);
    p_ry = float2(p_ry.x * 2.0 - 1.0, 1.0 - p_ry.y * 2.0);
    
    float4 cam_center = mul(camera_info.screen_to_camera, float4(p_center, 0.0, 1.0));
    float4 cam_rx = mul(camera_info.screen_to_camera, float4(p_rx, 0.0, 1.0));
    float4 cam_ry = mul(camera_info.screen_to_camera, float4(p_ry, 0.0, 1.0));
    
    cam_center.xyz /= cam_center.w;
    cam_rx.xyz /= cam_rx.w;
    cam_ry.xyz /= cam_ry.w;
    
    float3 ray_origin_camera = float3(0.0, 0.0, 0.0);
    float3 rx_origin_camera = float3(0.0, 0.0, 0.0);
    float3 ry_origin_camera = float3(0.0, 0.0, 0.0);
    
    float3 ray_dir_camera = normalize(cam_center.xyz);
    float3 rx_dir_camera = normalize(cam_rx.xyz);
    float3 ry_dir_camera = normalize(cam_ry.xyz);
    
    if (camera_info.enable_depth_of_field && camera_info.lens_radius > 0.0) {
        float t_focus = camera_info.focal_distance / max(-ray_dir_camera.z, EPS);
        float3 focal_point_camera = ray_dir_camera * t_focus;
        
        float t_focus_rx = camera_info.focal_distance / max(-rx_dir_camera.z, EPS);
        float t_focus_ry = camera_info.focal_distance / max(-ry_dir_camera.z, EPS);
        float3 focal_point_rx = rx_dir_camera * t_focus_rx;
        float3 focal_point_ry = ry_dir_camera * t_focus_ry;
        
        float2 lens_offset = lens_sample * camera_info.lens_radius;
        ray_origin_camera = float3(lens_offset, 0.0);
        rx_origin_camera = float3(lens_offset, 0.0);
        ry_origin_camera = float3(lens_offset, 0.0);
        
        ray_dir_camera = normalize(focal_point_camera - ray_origin_camera);
        rx_dir_camera = normalize(focal_point_rx - rx_origin_camera);
        ry_dir_camera = normalize(focal_point_ry - ry_origin_camera);
    } else {
        ray_origin_camera = float3(0.0, 0.0, 0.0);
        rx_origin_camera = float3(0.0, 0.0, 0.0);
        ry_origin_camera = float3(0.0, 0.0, 0.0);
    }
    
    float4 world_origin = mul(camera_info.camera_to_world, float4(ray_origin_camera, 1.0));
    float4 world_rx_origin = mul(camera_info.camera_to_world, float4(rx_origin_camera, 1.0));
    float4 world_ry_origin = mul(camera_info.camera_to_world, float4(ry_origin_camera, 1.0));
    
    float4 world_dir = mul(camera_info.camera_to_world, float4(ray_dir_camera, 0.0));
    float4 world_rx_dir = mul(camera_info.camera_to_world, float4(rx_dir_camera, 0.0));
    float4 world_ry_dir = mul(camera_info.camera_to_world, float4(ry_dir_camera, 0.0));
    
    outRay.Origin = world_origin.xyz;
    outRay.Direction = normalize(world_dir.xyz);
    outRay.TMin = t_min;
    outRay.TMax = t_max;
    
    outDiffs.hasDifferentials = true;
    outDiffs.rxOrigin = world_rx_origin.xyz;
    outDiffs.ryOrigin = world_ry_origin.xyz;
    outDiffs.rxDirection = normalize(world_rx_dir.xyz);
    outDiffs.ryDirection = normalize(world_ry_dir.xyz);
}

// ===================== 色散 ========================

static const float3x3 INV_SS_T = float3x3(
    82.482198365f, -59.275206773f, -11.463075523f,
    -59.275206773f, 72.376400305f, 64.337024116f,
    -11.463075523f, 64.337024116f, 17.969320974f
);

float f3_min(float3 u){
    return min(u[0], min(u[1], u[2]));
}

float GetSpectralAlbedo(float3 rgb, float3 w_spectral)
{
    float min_c = min(min(rgb.r, rgb.g), rgb.b);
    float max_c = max(max(rgb.r, rgb.g), rgb.b);
    float saturation = max_c - min_c;

    const float SATURATION_THRESHOLD = 0.05;

    if (saturation < SATURATION_THRESHOLD * min_c) {
        return (rgb.r + rgb.g + rgb.b) * 0.333333;
    } else {
        
        float r_lambda = dot(mul(w_spectral, INV_SS_T), rgb);

        return clamp(r_lambda, 0.0, 1.0);
    }
}

float GetCauchyIOR(float wavelength_nm, float A, float B, float C)
{
    float lambda_um = wavelength_nm / 1000.0;
    float lambda2 = lambda_um * lambda_um;
    float lambda4 = lambda2 * lambda2;
    
    return A + B / lambda2 + C / lambda4;
}

uint binary_search_wavelength_cdf(float rand_value)
{
    uint left = 0;
    uint right = SPECTRAL_SAMPLE_COUNT - 1;
    uint result = 0;
    
    if (rand_value <= WAVELENGTH_CDF[0]) {
        return 0;
    }
    if (rand_value >= WAVELENGTH_CDF[SPECTRAL_SAMPLE_COUNT - 1]) {
        return SPECTRAL_SAMPLE_COUNT - 1;
    }
    
    while (left <= right) {
        uint mid = (left + right) >> 1;
        float cdf_value = WAVELENGTH_CDF[mid];
        
        if (rand_value <= cdf_value) {
            result = mid;
            if (mid == 0 || rand_value > WAVELENGTH_CDF[mid - 1]) {
                break;
            }
            right = mid - 1;
        } else {
            left = mid + 1;
        }
    }
    return result;
}

// ====================== 主渲染逻辑 ======================
[shader("raygeneration")]
void RayGenMain() {
    uint2 pixel_coords = DispatchRaysIndex().xy;
    uint seed = generate_seed(pixel_coords, render_setting.frame_count);
    float2 lens_sample = float2(0.0, 0.0);
    if (camera_info.enable_depth_of_field && camera_info.lens_radius > 0.0) {
        lens_sample = concentric_sample_disk(seed);
    }
    
    float2 pixel_sample_offset = render_setting.enable_accumulation ? 
                                 random2(seed) : float2(0.5, 0.5);

    RayDesc ray;
    RayDifferential diffs;
    GenerateCameraRaysWithDifferentials(
        pixel_coords,
        pixel_sample_offset,
        render_setting.resolution,
        lens_sample,
        ray,
        diffs
    );
    
    if (camera_info.enable_motion_blur) {
        float time_offset = random(seed) * camera_info.exposure_time;

        float3 linear_move = camera_info.camera_linear_velocity * time_offset;
        ray.Origin += linear_move;
        diffs.rxOrigin += linear_move;
        diffs.ryOrigin += linear_move;

        float angular_offset = camera_info.camera_angular_velocity * time_offset;
        if (abs(angular_offset) > 1e-8) {
            float cos_theta = cos(angular_offset);
            float sin_theta = sin(angular_offset);
            float3x3 rotation_matrix = float3x3(
                cos_theta, -sin_theta, 0.0,
                sin_theta,  cos_theta, 0.0,
                0.0,        0.0,       1.0
            );

            ray.Direction = normalize(mul(rotation_matrix, ray.Direction));
            diffs.rxDirection = normalize(mul(rotation_matrix, diffs.rxDirection));
            diffs.ryDirection = normalize(mul(rotation_matrix, diffs.ryDirection));

            float3 cam_world_pos = mul(camera_info.camera_to_world, float4(0.0, 0.0, 0.0, 1.0)).xyz;
            ray.Origin = cam_world_pos + mul(rotation_matrix, ray.Origin - cam_world_pos);
            diffs.rxOrigin = cam_world_pos + mul(rotation_matrix, diffs.rxOrigin - cam_world_pos);
            diffs.ryOrigin = cam_world_pos + mul(rotation_matrix, diffs.ryOrigin - cam_world_pos);
        }

        diffs.hasDifferentials = false;
    }    

    float3x3 dirCov = (float3x3)0;
    bool hasDirCov = diffs.hasDifferentials;
    if (hasDirCov) {
        float3 v      = ray.Direction;
        float3 dv_dx  = diffs.rxDirection - v;
        float3 dv_dy  = diffs.ryDirection - v;

        dirCov[0][0] = dv_dx.x * dv_dx.x + dv_dy.x * dv_dy.x;
        dirCov[0][1] = dv_dx.x * dv_dx.y + dv_dy.x * dv_dy.y;
        dirCov[0][2] = dv_dx.x * dv_dx.z + dv_dy.x * dv_dy.z;

        dirCov[1][0] = dv_dx.y * dv_dx.x + dv_dy.y * dv_dy.x;
        dirCov[1][1] = dv_dx.y * dv_dx.y + dv_dy.y * dv_dy.y;
        dirCov[1][2] = dv_dx.y * dv_dx.z + dv_dy.y * dv_dy.z;

        dirCov[2][0] = dv_dx.z * dv_dx.x + dv_dy.z * dv_dy.x;
        dirCov[2][1] = dv_dx.z * dv_dx.y + dv_dy.z * dv_dy.y;
        dirCov[2][2] = dv_dx.z * dv_dx.z + dv_dy.z * dv_dy.z;
    }

    // diffs.hasDifferentials = false; // disable
    // hasDirCov = false;

    float3 color = float3(0.0, 0.0, 0.0);
    float3 throughput = float3(1.0, 1.0, 1.0);

    entity_id_output[pixel_coords] = -1;

    float3 prev_hit_point;
    float prev_bsdf_pdf;
    bool prev_is_specular = false;

    uint light_count, light_idx;
    light_count = render_setting.light_count;

    bool enable_dispersion = render_setting.enable_accumulation & camera_info.enable_dispersion;
    float wavelength_nm = 0.0;
    float3 wave_weight = float3(1, 1, 1);

    if (enable_dispersion) {
        float rand_w = random(seed);
        uint idx = binary_search_wavelength_cdf(rand_w);
        
        wave_weight = SPECTRAL_TABLE[idx];
        
        wavelength_nm = 380.0 + 5.0 * idx;
    }
    for (int depth = 0; depth < min(render_setting.max_depth, MAX_DEPTH); ++depth) {

        RayPayload payload;
        payload.attribute = float4(-1.0,-1.0,-1.0,-1.0);
        payload.diffs = diffs;
        payload.dirCov   = dirCov;
        payload.hasDirCov = hasDirCov;

        TraceRay(as, RAY_FLAG_NONE, 0xFF, 0, 1, 0, ray, payload);


        float3 wo = -ray.Direction;

        Material mat;
        MediumState medium = GetCurrentMedium();
        float surface_dist = HDR_DIS;

        if (payload.hit) {
            surface_dist = min(surface_dist, length(payload.hit_point - ray.Origin));
        }
        bool medium_has_ext = (render_setting.enable_accumulation && medium.enabled && (medium.sigma_t.x > 0.0 || medium.sigma_t.y > 0.0 || medium.sigma_t.z > 0.0));

        if (medium_has_ext) {
            bool scatter_in_medium = false;
            float t_medium = 0.0f;

            diffs.hasDifferentials = false;
            hasDirCov = false;

            int channel_idx = min((int)(random(seed) * 3.0), 2); // 0, 1, or 2

            t_medium = -log(max(random(seed), 1e-6)) / medium.sigma_t[channel_idx];
            
            if (t_medium < surface_dist) {
                scatter_in_medium = true;
            }
            if (scatter_in_medium) {
                float3 scatter_point = ray.Origin + ray.Direction * t_medium;

                float3 T = exp(-medium.sigma_t * t_medium);

                float pdf_t = max(dot(medium.sigma_t, T) / 3.0, eps);

                color += medium.Le * (1.0f - T) / medium.sigma_t * throughput / pdf_t;
            
                throughput *= T * medium.sigma_s / pdf_t;
                float3 vol_direct = mis_direct_lighting(light_count, scatter_point, float3(0.0f, 0.0f, 0.0f), mat, float3(0.0f, 0.0f, 0.0f), payload.is_filp, seed);

                color += throughput * vol_direct;

                if (depth > 4) {
                    float p_survive = min(f3_max(throughput), RR_THRESHOLD);
                    if (random(seed) > p_survive) break;
                    throughput /= p_survive;
                }

                float2 u_phase = random2(seed);
                float z = 1.0 - 2.0 * u_phase.x;
                float r = sqrt(max(0.0, 1.0 - z*z));
                float phi = TWO_PI * u_phase.y;
                float3 new_dir = float3(r * cos(phi), r * sin(phi), z);

                ray.Origin = scatter_point + new_dir * eps;
                ray.Direction = new_dir;
                ray.TMin = t_min;
                ray.TMax = t_max;

                prev_bsdf_pdf = 1.0 / (4.0 * PI);
                continue;
            }else{
                float3 T_surf = exp(-medium.sigma_t * surface_dist);
                float pdf_t = max((T_surf.r + T_surf.g + T_surf.b) / 3.0, eps);

                color += medium.Le * (1.0 - T_surf) / medium.sigma_t * (throughput / pdf_t);
                
                throughput *= T_surf / pdf_t;
            }
        }

        light_idx = 0xFFFFFFFF;
        if (!payload.hit) {
            if(render_setting.skybox_texture_id_ >= 0){
                light_idx = light_count;
            }
            else{ // no hdr
                if(enable_dispersion){
                    payload.color = GetSpectralAlbedo(payload.color, wave_weight);
                }
                color += payload.color * throughput;
                break;
            }
        }
        else{
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
            mat.transparency = 1.0 - payload.attribute.a;
        }

        if(enable_dispersion){
            mat.ior = GetCauchyIOR(wavelength_nm, mat.A, mat.B, mat.C);
            mat.base_color = GetSpectralAlbedo(mat.base_color, wave_weight);
        }
        
        if (light_idx != 0xFFFFFFFF) {
            Light light;
            if(light_idx < light_count) light = lights[light_idx];
            else{
                light.enabled = true;
                light.type = 4;
                light.color = payload.color;
                light.intensity = 1.0;
            }
            if(enable_dispersion){
                light.color = GetSpectralAlbedo(payload.color, wave_weight);
            }

            if(light.enabled && (light.type != 1 || dot(wo, light.direction) > 0) ){

                if (depth == 0 || prev_is_specular || !render_setting.enable_accumulation) {
                    color += light.color * light.intensity * throughput;
                } else {
                    float light_pdf = 0.0;

                    if(light.type == 4){
                        float luminance = dot(payload.color, float3(0.2126, 0.7152, 0.0722));
                        light_pdf = luminance / hdr_cdf[2];

                    } else{
                        float3 to_light = payload.hit_point - prev_hit_point;
                        float distance = length(to_light);
                        float3 light_dir = to_light / distance;
                        float cos_theta_l = max(EPS, abs(dot(light_dir, light.direction)));
                        
                        if (light.type == 1) { // area
                            float area = light.size.x * light.size.y;
                            light_pdf = (distance * distance) / (cos_theta_l * area);
                        } else if (light.type == 3) { // sphere
                            float surface_area = 4.0 * PI * light.radius * light.radius;
                            light_pdf = (distance * distance) / (cos_theta_l * surface_area);
                        }
                    }
                    
                    float light_select_pdf = light_power_weights[light_idx];
                    light_pdf *= light_select_pdf;
                    float mis_weight = mis_balance_weight(prev_bsdf_pdf, light_pdf);
                    
                    color += light.color * light.intensity * throughput * mis_weight; 
                }
                break;
            }
        }

        prev_hit_point = payload.hit_point;

        // MIS direct lighting
        if(render_setting.enable_accumulation){
            float3 direct_light = mis_direct_lighting(light_count, payload.hit_point, payload.normal, mat, wo, payload.is_filp, seed);
            if(enable_dispersion){
                direct_light = GetSpectralAlbedo(direct_light, wave_weight);
            }
            color += direct_light * throughput;
        }
        
        // emission
        if (any(mat.emission > 0.0)) {
            if(enable_dispersion){
                mat.emission = GetSpectralAlbedo(mat.emission, wave_weight);
            }
            color += mat.emission * throughput * (0.8 + abs(dot(payload.normal, ray.Direction)) * 0.3);
        }
        
        if (depth > 4) {
            float p_survive = min(f3_max(throughput), RR_THRESHOLD);
            if (random(seed) > p_survive) break;
            throughput /= p_survive;
        }
        
        float3 wi = float3(0.0, 0.0, 0.0);
        float pdf;
        float3 bsdf_val;
        if (mat.roughness < EPS) {
            float F0_ior = pow((mat.ior - 1.0) / (mat.ior + 1.0), 2.0);
            float3 F0 = lerp(float3(F0_ior, F0_ior, F0_ior), mat.base_color, mat.metallic);
            float NdotV = dot(payload.normal, wo);
            float3 F_color = SchlickFresnel(F0, abs(NdotV));
            float F_avg = (F_color.x + F_color.y + F_color.z) / 3.0;

            bool is_glass = (mat.transparency > 0.5);
            bool do_reflect = true;

            if (is_glass && mat.metallic < 1.0) {
                if (random(seed) < F_avg) {
                    do_reflect = true;
                } else {
                    do_reflect = false;
                }
            } else {
                do_reflect = true;
            }

            if (do_reflect) {
                wi = reflect(-wo, payload.normal);
                float cos_theta = dot(payload.normal, wi);
                bsdf_val = F_color / (abs(cos_theta) + EPS); 
            } else {
                float eta = payload.is_filp ? mat.ior : 1.0 / mat.ior;
                if (Refract(-wo, payload.normal, eta, wi) == false) {
                    wi = reflect(-wo, payload.normal);
                    float cos_theta = dot(payload.normal, wi);
                    bsdf_val = F_color / (abs(cos_theta) + EPS);
                } else {
                    float cos_theta = dot(payload.normal, wi);
                    bsdf_val = ((1.0 - F_color) * mat.base_color) / (abs(cos_theta) + EPS);
                }
            }
            
            pdf = 1.0;
            wi = normalize(wi);

        } else {
            if (!SampleBSDF(mat, wo, payload.normal, wi, payload.is_filp, seed)){
                break;
            }

            bsdf_val = EvalBSDF(mat, wo, wi, payload.normal, payload.is_filp, pdf);

            if(isinf(pdf) || isnan(pdf)){
              break;
            }
        }
        
        float cos_theta = dot(payload.normal, wi);
        if (cos_theta <= 0.0) {
            payload.normal = - payload.normal;
            cos_theta = - cos_theta;
        }

        bsdf_val *= cos_theta / max(pdf, EPS);

        throughput *= bsdf_val;

        if (mat.roughness > 0.6 && mat.metallic < 0.2 && mat.transparency < 0.2) {
            diffs.hasDifferentials = false;
            hasDirCov = false;
        }

        if (depth > 4) {
            diffs.hasDifferentials = false;
            hasDirCov = false;            
        }

        if (diffs.hasDifferentials)
        {
            float3 v_in = ray.Direction;

            float3 dv_dx = diffs.rxDirection - v_in;
            float3 dv_dy = diffs.ryDirection - v_in;

            float3 dn_dx = payload.dndx;
            float3 dn_dy = payload.dndy;

            float NdotV = dot(payload.normal, wo);
            float NdotL = dot(payload.normal, wi);
            bool is_trans = (NdotV * NdotL) < 0.0;

            float3 dwo_dx = 0.0;
            float3 dwo_dy = 0.0;


            float3 T, B;
            GetTangent(payload.normal, T, B);

            float alpha = mat.roughness * mat.roughness;
            float aspect = sqrt(max(1.0 - 0.9 * mat.anisotropic, EPS));
            float ax = max(EPS, alpha / aspect);
            float ay = max(EPS, alpha * aspect);

            float3x3 Cov_in = payload.dirCov;
            float3x3 Cov_h = MicroNormalCovariance_Aniso(
                T, B, ax, ay
            );

            float3x3 J_L_v;
            float3x3 J_L_n;

            if (!is_trans)
            {
                // 反射
                J_L_v = ReflectionJacobian_wrt_Dir(v_in, payload.normal);
                J_L_n = ReflectionJacobian_wrt_Normal(v_in, payload.normal);

                dwo_dx = float3(
                    dot(J_L_v[0], dv_dx) + dot(J_L_n[0], dn_dx),
                    dot(J_L_v[1], dv_dx) + dot(J_L_n[1], dn_dx),
                    dot(J_L_v[2], dv_dx) + dot(J_L_n[2], dn_dx)
                );
                dwo_dy = float3(
                    dot(J_L_v[0], dv_dy) + dot(J_L_n[0], dn_dy),
                    dot(J_L_v[1], dv_dy) + dot(J_L_n[1], dn_dy),
                    dot(J_L_v[2], dv_dy) + dot(J_L_n[2], dn_dy)
                );
            }
            else
            {
                // 折射
                float eta = (!payload.is_filp) ? (1.0 / mat.ior) : mat.ior;

                J_L_v = RefractionJacobian_wrt_Dir(v_in, payload.normal, eta);
                J_L_n = RefractionJacobian_wrt_Normal(v_in, payload.normal, eta);

                dwo_dx = float3(
                    dot(J_L_v[0], dv_dx) + dot(J_L_n[0], dn_dx),
                    dot(J_L_v[1], dv_dx) + dot(J_L_n[1], dn_dx),
                    dot(J_L_v[2], dv_dx) + dot(J_L_n[2], dn_dx)
                );
                dwo_dy = float3(
                    dot(J_L_v[0], dv_dy) + dot(J_L_n[0], dn_dy),
                    dot(J_L_v[1], dv_dy) + dot(J_L_n[1], dn_dy),
                    dot(J_L_v[2], dv_dy) + dot(J_L_n[2], dn_dy)
                );
            }

            if (payload.hasDirCov)
            {
                float3x3 Cov_geom  = PropagateNormalCovToDirectionCov(J_L_v, Cov_in);
                float3x3 Cov_micro = PropagateNormalCovToDirectionCov(J_L_n, Cov_h);

                dirCov = Cov_geom + Cov_micro;
                float trace = dirCov[0][0] + dirCov[1][1] + dirCov[2][2];
                float maxTrace = 10.0;
                if (trace > maxTrace) {
                    float scale = maxTrace / max(trace, EPS);
                    dirCov *= scale;
                }
                hasDirCov = true;
            }


            diffs.rxDirection = normalize(wi + dwo_dx);
            diffs.ryDirection = normalize(wi + dwo_dy);

            diffs.rxOrigin = payload.hit_point + payload.dpdx;
            diffs.ryOrigin = payload.hit_point + payload.dpdy;
        }
        
        prev_bsdf_pdf = pdf;
        
        prev_is_specular = (mat.roughness < EPS);

        ray.Origin = payload.hit_point + payload.normal * eps;
        ray.Direction = wi;
        ray.TMin = t_min;
        ray.TMax = t_max;
    }

    output[pixel_coords] = float4(color, 1);
    
    if (render_setting.enable_accumulation) {
        float4 prev_color = accumulated_color[pixel_coords];
        int prev_samples = accumulated_samples[pixel_coords];
        accumulated_samples[pixel_coords] = prev_samples + 1;
        if(enable_dispersion){
            color = color.x * wave_weight * 3.0 / (wave_weight.x + wave_weight.y + wave_weight.z); // consider pdf
        }
        if(prev_samples >= 1){ 
            color = min(color, max(prev_color.xyz / prev_samples * 1000, 500.0));
        }

        accumulated_color[pixel_coords] = prev_color + float4(color, 1);
    }
}

// // ====================== Miss ======================
[shader("miss")]
void MissMain(inout RayPayload payload) {
    payload.hit = false;
    payload.instance_id = 0xFFFFFFFF;
    
    if (render_setting.skybox_texture_id_ >= 0) {
        float3 normalizedDir = normalize(WorldRayDirection());
        
        float phi = atan2(normalizedDir.z, normalizedDir.x);
        float theta = acos(clamp(normalizedDir.y, -1.0, 1.0));
        
        float u = (phi + PI) / (2.0 * PI);
        float v = theta / PI;
        
        u = frac(u);
        v = clamp(v, 0.0, 1.0);
        
        float lod = 0.0;
        if (payload.hasDirCov) {
            float2x2 CovUV;
            DirectionCovToLatLongUVCov(normalizedDir, payload.dirCov, CovUV);

            float2 axis_len;
            float2x2 cov_texel;
            lod = ComputeEnvmapLOD_FromCovUV(
                render_setting.skybox_texture_id_, CovUV,
                axis_len, cov_texel
            );

            lod = clamp(lod, 0.0, 10.0);
        }

        payload.color = SampleTexture2D_Lod(render_setting.skybox_texture_id_, float2(u,v), lod).rgb;
    } else {
        float t = 0.5 * (normalize(WorldRayDirection()).y + 1.0);
        payload.color = lerp(float3(0.0, 0.7, 0.8), float3(1.0, 0.75, 0.8), t * 1.2) * 1.0;
    }
}


// ========================== Adaptive mipmap ====================================

float ComputeTextureLOD(Texture2D<float4> tex,
                        float dudx, float dudy, float dvdx, float dvdy)
{
    uint w, h;
    tex.GetDimensions(w, h);
    float2 dx = float2(dudx * (float)w, dvdx * (float)h);
    float2 dy = float2(dudy * (float)w, dvdy * (float)h);
    float rho2 = max(dot(dx, dx), dot(dy, dy));
    rho2 = max(rho2, 1e-8);
    float lod = 0.5 * log2(rho2);
    if (lod < 0.0) lod = 0.0;
    return lod;
}

void ComputeSmoothNormalDerivatives_Triangle(
    float3 n0, float3 n1, float3 n2,  
    float3 E1, float3 E2,             
    float2 bary,                      
    float3 dpdx, float3 dpdy,          
    out float3 dndx, out float3 dndy,  
    out float3 n_shading)              
{
    float u = bary.x;
    float v = bary.y;
    float w = 1.0 - u - v;
    float3 n_interp = u * n1 + v * n2 + w * n0;
    float lenN = length(n_interp);
    if (lenN < EPS) {
        n_shading = normalize(cross(E1, E2));
        if (length(n_shading) < EPS) {
            n_shading = float3(0.0, 0.0, 1.0);
        }
    } else {
        n_shading = normalize(n_interp);
    }

    float3 dn_du = n1 - n0;
    float3 dn_dv = n2 - n0;

    float2 ab_dx = SolveForCoeffs3x2(E1, E2, dpdx); // returns (a_dx, b_dx)
    float2 ab_dy = SolveForCoeffs3x2(E1, E2, dpdy); // returns (a_dy, b_dy)

    dndx = dn_du * ab_dx.x + dn_dv * ab_dx.y;
    dndy = dn_du * ab_dy.x + dn_dv * ab_dy.y;

}

void ComputeNormalMapDerivatives_WithCustomLOD(
    int id,
    float2 uv,
    float dudx, float dudy, float dvdx, float dvdy,
    float custom_lod,
    out float3 n_t, out float3 dntdx, out float3 dntdy)
{
    float4 c = SampleTexture2D_Lod(id, uv, custom_lod);
    float3 sC = c.xyz;
    float3 nc = sC * 2.0 - 1.0;
    n_t = normalize(nc);

    uint w, h;
    g_Textures[id].GetDimensions(w, h);
    
    float mip_scale = exp2(custom_lod);
    float2 texel = 1.0 / float2(max(1u, w), max(1u, h)) * mip_scale;
    
    float4 su = SampleTexture2D_Lod(id, uv + float2(texel.x, 0.0), custom_lod);
    float4 su_m = SampleTexture2D_Lod(id, uv - float2(texel.x, 0.0), custom_lod);
    float4 sv = SampleTexture2D_Lod(id, uv + float2(0.0, texel.y), custom_lod);
    float4 sv_m = SampleTexture2D_Lod(id, uv - float2(0.0, texel.y), custom_lod);

    float3 dn_du = (su.xyz - su_m.xyz) / texel.x;
    float3 dn_dv = (sv.xyz - sv_m.xyz) / texel.y;

    dntdx = dn_du * dudx + dn_dv * dvdx;
    dntdy = dn_du * dudy + dn_dv * dvdy;
}

void UpdateNormalDerivativesAndNormalMap(
    float3 v0, float3 v1, float3 v2,
    float3 n0, float3 n1, float3 n2,
    float2 uv0, float2 uv1, float2 uv2,
    float2 uv_hit,
    float2 bary,
    int normal_tex_id,
    float custom_lod,   
    inout RayPayload payload)
{
    float3 E1 = v1 - v0;
    float3 E2 = v2 - v0;

    float3 dndx_vert = float3(0,0,0);
    float3 dndy_vert = float3(0,0,0);
    float3 n_shading = float3(0,0,1);

    ComputeSmoothNormalDerivatives_Triangle(n0, n1, n2, E1, E2, bary, payload.dpdx, payload.dpdy, dndx_vert, dndy_vert, n_shading);

    float3 triT, triB;
    
    {
        float2 deltaUV1 = uv1 - uv0;
        float2 deltaUV2 = uv2 - uv0;
        
        float det = (deltaUV1.x * deltaUV2.y - deltaUV1.y * deltaUV2.x);
        
        if (abs(det) < 1e-6) {
            GetTangent(n_shading, triT, triB);
        } else {
            float invDet = 1.0 / det;

            triT = invDet * (deltaUV2.y * E1 - deltaUV1.y * E2);
            triB = invDet * (-deltaUV2.x * E1 + deltaUV1.x * E2);

            triT = normalize(triT);
            triB = normalize(triB);
        }
    }

    float3 dndx_map = float3(0,0,0);
    float3 dndy_map = float3(0,0,0);
    float3 final_world_normal = normalize(n_shading);

    if (normal_tex_id >= 0) {
        
        float3 n_t; float3 dntdx_t; float3 dntdy_t;
        ComputeNormalMapDerivatives_WithCustomLOD(
            normal_tex_id, uv_hit,
            payload.dudx, payload.dudy, payload.dvdx, payload.dvdy,
            custom_lod,
            n_t, dntdx_t, dntdy_t);

        final_world_normal = normalize(triT * n_t.x + triB * n_t.y + n_shading * n_t.z);

        dndx_map = triT * dntdx_t.x + triB * dntdx_t.y + n_shading * dntdx_t.z;
        dndy_map = triT * dntdy_t.x + triB * dntdy_t.y + n_shading * dntdy_t.z;

    }

    payload.dndx = dndx_vert + dndx_map;
    payload.dndy = dndy_vert + dndy_map;

    payload.normal = normalize(final_world_normal);
}

[shader("closesthit")]
void ClosestHitMain(inout RayPayload payload, in BuiltInTriangleIntersectionAttributes attr) {
    payload.hit = true;
    payload.instance_id = InstanceID();
    uint primitive_id = PrimitiveIndex();

    float3 ray_origin = WorldRayOrigin();
    float3 ray_direction = WorldRayDirection();
    float hit_distance = RayTCurrent();
    payload.hit_point = ray_origin + hit_distance * ray_direction;

    GeometryDescriptor geo_desc = geometry_descriptors[payload.instance_id];
    uint index_offset = geo_desc.index_offset + primitive_id * 3;
    uint i0 = indices[index_offset];
    uint i1 = indices[index_offset + 1];
    uint i2 = indices[index_offset + 2];

    VertexInfo v0 = vertices[geo_desc.vertex_offset + i0];
    VertexInfo v1 = vertices[geo_desc.vertex_offset + i1];
    VertexInfo v2 = vertices[geo_desc.vertex_offset + i2];

    // triangle barycentrics
    float w0 = attr.barycentrics.x; // weight for v1
    float w1 = attr.barycentrics.y; // weight for v2
    float w2 = 1.0 - w0 - w1;       // weight for v0

    // uv interpolation
    float2 uv_hit = w0 * v1.texcoord + w1 * v2.texcoord + w2 * v0.texcoord;

    // transform positions & normals to world space
    float3 p0_w = mul(float4(v0.position, 1.0), ObjectToWorld4x3());
    float3 p1_w = mul(float4(v1.position, 1.0), ObjectToWorld4x3());
    float3 p2_w = mul(float4(v2.position, 1.0), ObjectToWorld4x3());

    // object -> world
    float3x3 normal_matrix = (float3x3)transpose(WorldToObject4x3());
    float3 n0_w = normalize(mul(normal_matrix, v0.normal));
    float3 n1_w = normalize(mul(normal_matrix, v1.normal));
    float3 n2_w = normalize(mul(normal_matrix, v2.normal));
    

    float3 E1_w = p1_w - p0_w; // v1 - v0
    float3 E2_w = p2_w - p0_w; // v2 - v0
    float3 geo_normal = normalize(cross(E1_w, E2_w));

    if (length(v0.normal) < 1e-6f) {
        n0_w = geo_normal;
        n1_w = geo_normal;
        n2_w = geo_normal;
    } else {
        n0_w = normalize(n0_w);
        n1_w = normalize(n1_w);
        n2_w = normalize(n2_w);
    }

    // dpdx / dpdy
    if (payload.diffs.hasDifferentials) {

        float plane_dist = dot(p0_w, geo_normal); // d = N dot P

        // 2. t = (d - N dot O) / (N dot D)
        float denom_rx = dot(payload.diffs.rxDirection, geo_normal);
        float denom_ry = dot(payload.diffs.ryDirection, geo_normal);

        // prevent divide by zero
        denom_rx = abs(denom_rx) < EPS ? EPS : denom_rx;
        denom_ry = abs(denom_ry) < EPS ? EPS : denom_ry;

        float t_rx = (plane_dist - dot(payload.diffs.rxOrigin, geo_normal)) / denom_rx;
        float t_ry = (plane_dist - dot(payload.diffs.ryOrigin, geo_normal)) / denom_ry;

        // 3. hit point
        float3 hit_rx = payload.diffs.rxOrigin + t_rx * payload.diffs.rxDirection;
        float3 hit_ry = payload.diffs.ryOrigin + t_ry * payload.diffs.ryDirection;

        // 4. dpdx / dpdy
        payload.dpdx = hit_rx - payload.hit_point;
        payload.dpdy = hit_ry - payload.hit_point;
    } else {
        payload.dpdx = float3(0.0, 0.0, 0.0);
        payload.dpdy = float3(0.0, 0.0, 0.0);
    }

    float2 ab_dx = SolveForCoeffs3x2(E1_w, E2_w, payload.dpdx); // a_dx = ∂u/∂x, b_dx = ∂v/∂x
    float2 ab_dy = SolveForCoeffs3x2(E1_w, E2_w, payload.dpdy); // a_dy = ∂u/∂y, b_dy = ∂v/∂y

    // uv derivatives
    float2 uv10 = v1.texcoord - v0.texcoord;
    float2 uv20 = v2.texcoord - v0.texcoord;
    payload.dudx = uv10.x * ab_dx.x + uv20.x * ab_dx.y;
    payload.dvdx = uv10.y * ab_dx.x + uv20.y * ab_dx.y;
    payload.dudy = uv10.x * ab_dy.x + uv20.x * ab_dy.y;
    payload.dvdy = uv10.y * ab_dy.x + uv20.y * ab_dy.y;

    // attribute & base color
    Material mat = materials[InstanceID()];
    int attribute_id = mat.attribute_tex_id;
    int texture_id = mat.texture_id;
    int normal_id = mat.normal_tex_id;
    

    if (attribute_id >= 0) {
        float lod_attr = ComputeTextureLOD(g_Textures[attribute_id], payload.dudx, payload.dudy, payload.dvdx, payload.dvdy);
        payload.attribute = SampleTexture2D_Lod(attribute_id, uv_hit, lod_attr).rgba;
    } else {
        payload.attribute = float4(-1.0, -1.0, -1.0, -1.0);
    }

    if (texture_id >= 0) {
        float lod_col = ComputeTextureLOD(g_Textures[texture_id], payload.dudx, payload.dudy, payload.dvdx, payload.dvdy);
        payload.color = SampleTexture2D_Lod(texture_id, uv_hit, lod_col).rgb;
    } else {
        payload.color = mat.base_color;
    }

    float lod_normal = 0.0;

    if (payload.diffs.hasDifferentials && normal_id >= 0)
    {
        lod_normal = ComputeTextureLOD(
            g_Textures[normal_id],
            payload.dudx, payload.dudy, payload.dvdx, payload.dvdy
        );
    }
    float2 bary = float2(w0, w1);
    UpdateNormalDerivativesAndNormalMap(
        p0_w, p1_w, p2_w,
        n0_w, n1_w, n2_w,
        v0.texcoord, v1.texcoord, v2.texcoord,
        uv_hit,
        bary,
        normal_id,
        lod_normal,
        payload);

    payload.is_filp = false;
    if (dot(payload.normal, ray_direction) > 0.0) {
        payload.normal = -payload.normal;
        payload.is_filp = true;
    }
}