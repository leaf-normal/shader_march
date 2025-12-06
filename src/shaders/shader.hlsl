struct CameraInfo {
  float4x4 screen_to_camera;
  float4x4 camera_to_world;
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
  
  float subsurface;     //次表面散射
  float specular;      //镜面反射强度
  float specular_tint; //镜面反射
  float anisotropic;   //各向异性[-1,1]
  float sheen;         //光泽层
  float sheen_tint;    //光泽层染色
  float clearcoat;     //清漆层强度
  float clearcoat_roughness; //清漆层粗糙度
  float specular_transmission; //镜面透射
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
// float3 evaluate_brdf(float3 wi, float3 wo, float3 normal, Material mat) {
//     // 漫反射分量
//     float3 diffuse = mat.base_color / PI;
    
//     // 镜面分量（简化版）
//     float3 h = normalize(wi + wo);
//     float ndoth = max(0.0, dot(normal, h));
//     float specular = pow(ndoth, 32.0 * (1.0 - mat.roughness));
    
//     float3 specular_color = lerp(float3(0.04, 0.04, 0.04), mat.base_color, mat.metallic);
    
//     return diffuse + specular_color * specular;  
// }

// float bsdf_pdf(float3 wi, float3 wo, float3 normal, Material mat) {
//     if (mat.metallic < 0.5) {
//         // 漫反射：余弦加权半球采样
//         return max(0.0, dot(wi, normal)) / PI;
//     } else {
//         // 镜面反射：理想镜面
//         return 1.0;
//     }
// }
// ====================== BSDF系统 ======================
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

float SchlickWeight(float cosTheta)
{
  return pow(clamp(1.0-cosTheta,0.0,1.0),5.0);
}
float3 SchlickFresnel(float3 F0,float cosTheta)
{
  return F0+(1.0-F0)*SchlickWeight(cosTheta);
}

float GTR1(float NdotH,float a)
{
  if(a>=1.0)return 1.0/PI;
  float a2=sqr(a);
  float t=1.0+(a2-1.0)*sqr(NdotH);
  return (a2-1.0)/(PI*log(a2)*t);
}
float GTR2(float NdotH,float a)
{
    float a2=a*a;
    float t=1.0+(a2-1.0)*sqr(NdotH);
    return a2/(PI*sqr(t));
}
float GTR2_Anisotropic(float NdotH,float HdotX,float HdotY,float ax,float ay)
{
    return 1.0/(PI*ax*ay*sqr(sqr(HdotX/ax)+sqr(HdotY/ay)+NdotH*NdotH));
}

float SmithG_GGX(float NdotV,float alphaG) {
    float a=alphaG*alphaG;
    float b=sqr(NdotV);
    return 1.0/(NdotV+sqrt(a+b-a*b));
}
float SmithG_GGX_Anisotropic(float NdotV,float VdotX,float VdotY,float ax,float ay)
{
    return 1.0/(NdotV+sqrt(sqr(VdotX*ax)+sqr(VdotY*ay)+sqr(NdotV)));
}
float3 SampleHemisphereCos(float2 randd,float3 normal)
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

float3 SampleGGX_VNDF(float3 ray,float roughness,float2 rd,float3 normal)
{
  float3 V=ray;
  float alpha=sqr(roughness);
  float3 Vh=normalize(float3(alpha*V.x,alpha*V.y,V.z));
  float3 up=abs(Vh.z)<1-eps?float3(0.0,0.0,1.0):float3(1.0,0.0,0.0);
  float3 T1=normalize(cross(up,Vh));
  float3 T2=cross(Vh,T1);
  float r=sqrt(rd.x);
  float phi=2.0*PI*rd.y;
  float t1=r*cos(phi);
  float t2=r*sin(phi);
  float s=0.5*(1.0+Vh.z);
  t2=(1.0-s)*sqrt(1.0-sqr(t1))+s*t2;
  float3 Nh=t1*T1+t2*T2+sqrt(max(0.0,1.0-sqr(t1)-sqr(t2)))*Vh;
  float3 H=normalize(float3(alpha*Nh.x,alpha*Nh.y,max(0.0,Nh.z)));
  return H;
}


float3 SampleGGX_Anisotropic(float3 ray,float roughness,float anisotropic,float2 rd,float3 normal,float3 tangent)
{
    // 各向异性采样
  float aspect=sqrt(1.0-0.9*anisotropic);
  float ax=max(eps,roughness/aspect);
  float ay=max(eps,roughness*aspect);
  float3 bitangent=cross(normal,tangent);
  float3 V=ray;
  float3 Vh=normalize(float3(ax*V.x,ay*V.y,V.z));
  float3 up=abs(Vh.z)<1-eps?float3(0.0,0.0,1.0):float3(1.0,0.0,0.0);
  float3 T1=normalize(cross(up,Vh));
  float3 T2=cross(Vh,T1);  
  float r=sqrt(rd.x);
  float phi=2.0*PI*rd.y;
  float t1=r*cos(phi);
  float t2=r*sin(phi);
  float s=0.5*(1.0+Vh.z);
  t2=(1.0-s)*sqrt(1.0-sqr(t1))+s*t2;
  float3 Nh=t1*T1+t2*T2+sqrt(max(0.0,1.0-sqr(t1)-sqr(t2)))*Vh;
  float3 H=normalize(float3(ax*Nh.x,ay*Nh.y,max(0.0,Nh.z)));
  return H;
}

void GetTangent(float3 normal,out float3 tangent,out float3 bitangent)
{
  float3 up=abs(normal.z)<0.999?float3(0.0,0.0,1.0):float3(1.0,0.0,0.0);
  tangent=normalize(cross(up,normal));
  bitangent=cross(normal,tangent);
}

void SampleBSDF(Material material, float3 ray, float3 normal, out float3 wi, inout uint seed){
  // do something
  float3 tangent,bitangent; //固定切线for各向异性
  GetTangent(normal,tangent,bitangent);
  float3 F0=lerp(0.08*material.specular,material.base_color,material.metallic);
  //lobe权重
  float diffuseweight=(1.0-material.metallic)*(1.0-material.transparency);//漫反射
  float specularweight=1.0; //镜面反射
  float transmissionweight=material.transparency*material.specular_transmission;//透射
  float clearcoatweight=material.clearcoat;//清漆层
  float sheenweight=material.sheen*(1.0-material.metallic);//光泽层
  //normalization
  float total=diffuseweight+specularweight+transmissionweight+clearcoatweight+sheenweight;
  diffuseweight/=total;
  specularweight/=total;
  transmissionweight/=total;
  clearcoatweight/=total;
  sheenweight/=total;
  
  float randLobe=random(seed);
  float3 H;
  if(randLobe<diffuseweight+sheenweight)//漫反射+简化光泽层模型
  {
    wi=SampleHemisphereCos(float2(random(seed),random(seed)),normal);
  }else if(randLobe<diffuseweight+sheenweight+specularweight)//镜面反射
  {
    if(abs(material.anisotropic)>0.001)
      H=SampleGGX_Anisotropic(ray,material.roughness,material.anisotropic,float2(random(seed),random(seed)),normal,tangent);
    else
      H=SampleGGX_VNDF(ray,material.roughness,float2(random(seed),random(seed)),normal);
    wi=reflect(-ray,H);
  }else if(randLobe<1-clearcoatweight)//透射
  {
    float eta=dot(ray,normal)>0.0?1.0/material.ior:material.ior;
    H=SampleGGX_VNDF(ray,material.roughness,float2(random(seed),random(seed)),normal);
    wi=refract(-ray,H,eta);
    if(length(wi)<eps)wi=reflect(-ray,H);//全反射
  }else{//清漆
    H=SampleGGX_VNDF(ray,material.clearcoat_roughness,float2(random(seed),random(seed)),normal);
    wi=reflect(-ray,H);
  }
  if(dot(normal,wi)<0.0&&material.transparency<eps)wi=reflect(wi,normal);
}

float3 EvalBSDF(Material material,float3 ray,float3 wi,float3 normal,out float pdf)
{
  float3 ret=float3(0.0,0.0,0.0);
  float3 tangent,bitangent;
  GetTangent(normal,tangent,bitangent);
  float Ndotray=dot(normal,ray);
  float Ndotwi=dot(normal,wi);
  bool is_trans=Ndotray*Ndotwi<0.0;
  if(Ndotray<=0.0)
  {
    pdf=0.0;
    return float3(0.0,0.0,0.0);
  }
  if(!is_trans&&Ndotwi<=0.0)
  {
    pdf=0.0;
    return float3(0.0,0.0,0.0);
  }
  float alpha=sqr(material.roughness);
  float transmissionPDF=0.0;
  float specularPDF=0.0;
  float diffusePDF=0.0;
  float sheenPDF=0.0;
  float clearcoatPDF=0.0;
  if(is_trans)//透射
  {
    float eta=Ndotray>0.0?1.0/material.ior:material.ior;
    float3 H=normalize(ray+wi*eta);
    float NdotH=dot(normal,H);
    float Hdotray=dot(H,ray);
    float Hdotwi=dot(H,wi);
    float D=GTR2(NdotH,alpha);
    float G=SmithG_GGX(Ndotray,material.roughness)*SmithG_GGX(abs(dot(normal,wi)),material.roughness);
    float3 F0=lerp(0.08*material.specular,material.base_color,material.metallic);
    float3 F=SchlickFresnel(F0,Hdotray);
    float denom=sqr(Hdotray+eta*Hdotwi);
    float3 transmission=(material.base_color*(1.0-F)*D*G*abs(Hdotwi)*abs(Hdotray))/(abs(Ndotray)*abs(Ndotwi)*denom+eps);
    ret+=transmission*material.transparency*material.specular_transmission;
    float jacobian=(eta*eta*abs(Hdotwi))/denom;
    transmissionPDF=D*abs(NdotH)*jacobian;
  }else{
    float3 H=normalize(ray+wi);
    float NdotH=dot(normal,H);
    float Hdotray=dot(H,ray);
    float3 F0=lerp(0.08*material.specular,material.base_color,material.metallic);
    float3 F=SchlickFresnel(F0,Hdotray);
    //漫反射
    if(material.metallic<1.0&&material.transparency<1.0)
    {
      float FL=SchlickWeight(Ndotray);
      float FV=SchlickWeight(Ndotwi);
      float Fd90=0.5+2.0*material.roughness*sqr(Hdotray);
      float Fd=lerp(1.0,Fd90,FL)*lerp(1.0,Fd90,FV);
      float3 diffuse=material.base_color*(1.0-material.metallic)*Fd/PI;
      ret+=diffuse*(1.0-F);
    }
    diffusePDF=max(Ndotwi,0.0)/PI;
    //镜面反射
    float D,G;
    if(abs(material.anisotropic)>eps)//各向异性镜面
    {
      float aspect=sqrt(1.0-0.9*material.anisotropic);
      float ax=max(eps,alpha/aspect);
      float ay=max(eps,alpha*aspect);
      float HdotX=dot(H,tangent);
      float HdotY=dot(H,bitangent);
      D=GTR2_Anisotropic(NdotH,HdotX,HdotY,ax,ay);
      float VdotX=dot(ray,tangent);
      float VdotY=dot(ray,bitangent);
      float LdotX=dot(wi,tangent);
      float LdotY=dot(wi,bitangent);
      G=SmithG_GGX_Anisotropic(Ndotray,VdotX,VdotY,ax,ay)*SmithG_GGX_Anisotropic(Ndotwi,LdotX,LdotY,ax,ay);
      specularPDF=GTR2_Anisotropic(NdotH,dot(H,tangent),dot(H,bitangent),ax,ay)*NdotH/(4.0*Hdotray);
    }else{//各向同性
      D=GTR2(NdotH,alpha);
      G=SmithG_GGX(Ndotray,material.roughness)*SmithG_GGX(Ndotwi,material.roughness);
      specularPDF=GTR2(NdotH,alpha)*NdotH/(4.0*Hdotray);
    }
    float3 spec=(D*G*F)/(4.0*Ndotray*Ndotwi);
    if(material.specular_tint>0.0)
    {
      float3 tint=lerp(float3(1.0,1.0,1.0),material.base_color,material.specular_tint);
      spec*=tint;
    }
    ret+=spec;//光泽层
    if(material.sheen>0.0&&material.metallic<1.0)
    {
      float3 sheen_color=lerp(float3(1.0,1.0,1.0),material.base_color,material.sheen_tint);
      float sheen_intensity=material.sheen*(1.0-material.metallic);
      float3 sheen_F=SchlickFresnel(sheen_color,Hdotray);
      ret+=sheen_F*sheen_intensity*(1.0-F);
    }
    //清漆层
    if(material.clearcoat>0.0)
    {
      float clearcoat_alpha=material.clearcoat_roughness*material.clearcoat_roughness;
      float D_clearcoat=GTR1(NdotH,clearcoat_alpha);
      float G_clearcoat=SmithG_GGX(Ndotray,material.clearcoat_roughness)*SmithG_GGX(Ndotwi,material.clearcoat_roughness);
      float3 F_clearcoat=SchlickFresnel(float3(0.04,0.04,0.04),Hdotray); 
      float3 clearcoat=(D_clearcoat*G_clearcoat*F_clearcoat)/(4.0 * Ndotray * Ndotwi);
      ret=ret*(1.0-F_clearcoat*material.clearcoat)+clearcoat*material.clearcoat;
      clearcoatPDF=GTR1(NdotH,clearcoat_alpha)*NdotH/(4.0*Hdotray);
    }
    
    sheenPDF=diffusePDF;
  }
  //lobe权重
  float diffuseweight=(1.0-material.metallic)*(1.0-material.transparency);//漫反射
  float specularweight=1.0; //镜面反射
  float transmissionweight=material.transparency*material.specular_transmission;//透射
  float clearcoatweight=material.clearcoat;//清漆层
  float sheenweight=material.sheen*(1.0-material.metallic);//光泽层
  //normalization
  float total=diffuseweight+specularweight+transmissionweight+clearcoatweight+sheenweight;
  diffuseweight/=total;
  specularweight/=total;
  transmissionweight/=total;
  clearcoatweight/=total;
  sheenweight/=total;
  pdf=diffusePDF*diffuseweight+specularPDF*specularweight+transmissionweight*transmissionPDF+clearcoatweight*clearcoatPDF+sheenPDF*sheenweight;
  return ret;
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
        float pdf;
        float3 bsdf_val = EvalBSDF(mat, wo, wi, normal, pdf);
        float ndotl = max(0.0, dot(normal, wi));
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
                // 面光源和球光源：使用MIS
                float mis_weight = mis_balance_weight(light_pdf, pdf);
                total_light += light_contrib * mis_weight / light_pdf;
            }
        }
    }
    return total_light;
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

    float3 prev_hit_point;
    float3 prev_eval_bsdf;
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
        
        // 处理光源命中
        if (mat.light_index != 0xFFFFFFFF) {
            if (depth == 0 || prev_is_specular) {
                // 第一次直接击中光源或镜面反射击中光源
                Light light = lights[mat.light_index];
                if (light.enabled) {
                    color += light.color * light.intensity * throughput;
                }
            } else {
                // BSDF采样击中光源，需要MIS
                Light light = lights[mat.light_index];
                if (!light.enabled || (light.type != 1 && light.type != 3)) {
                    break;
                }
                
                float3 to_light = payload.hit_point - prev_hit_point;
                float distance = length(to_light);
                float3 light_dir = to_light / distance;
                
                // 计算光源采样PDF
                float light_pdf = 0.0;
                float cos_theta_l = max(0.0, dot(-light_dir, payload.normal));
                
                if (light.type == 1) { // 面光源
                    float area = light.size.x * light.size.y;
                    light_pdf = (distance * distance) / (cos_theta_l * area);
                } else if (light.type == 3) { // 球光源
                    float surface_area = 4.0 * PI * light.radius * light.radius;
                    light_pdf = (distance * distance) / (cos_theta_l * surface_area);
                }
                
                float light_select_pdf = light_power_weights[mat.light_index];
                light_pdf *= light_select_pdf;

                // 计算MIS权重
                float mis_weight = mis_balance_weight(prev_bsdf_pdf, light_pdf);
                
                // 应用MIS
                color += light.color * light.intensity * throughput * prev_eval_bsdf * mis_weight / prev_bsdf_pdf;
            }
            break;
        }

        prev_hit_point = payload.hit_point;

        // MIS直接光照
        float3 direct_light = mis_direct_lighting(light_count, payload.hit_point, payload.normal, mat, wo, seed);
        color += direct_light * throughput;
        
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
        float3 wi;
        float pdf;
        SampleBSDF(mat, wo, payload.normal, wi, seed);
        float3 bsdf_val = EvalBSDF(mat, wo, wi, payload.normal, pdf);
        
        if (pdf <= 0.0) {
            break;
        }
        
        // 计算余弦项
        float cos_theta = max(0.0, dot(payload.normal, wi));
        if (cos_theta <= 0.0) {
            // 检查是否为透射
            float Ndotray = dot(payload.normal, wo);
            float Ndotwi = dot(payload.normal, wi);
            bool is_trans = Ndotray * Ndotwi < 0.0;
            if (!is_trans) break; // 非透射且余弦为负，终止
        }
        
        // 更新吞吐量
        throughput *= bsdf_val * abs(cos_theta) / pdf;
        
        // 记录用于MIS的数据
        prev_bsdf_pdf = pdf;
        prev_eval_bsdf = bsdf_val * abs(cos_theta);
        
        // 判断是否为镜面反射（简化：检查材质是否为金属或清漆）
        prev_is_specular = (mat.metallic > 0.5 || mat.clearcoat > 0.0);
        
        // 准备下一次光线
        ray.Origin = payload.hit_point + payload.normal * eps;
        ray.Direction = normalize(wi);
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