#include <cmath>
#include <cfloat>
#include <iostream>
#include <random>
#include <vector>
#include <iomanip>
#include <chrono>
using namespace std;
// 常量定义
#define MAX_DEPTH 4
#define RR_THRESHOLD 0.95f
#define t_min 1e-3
#define t_max 10000.0
#define eps 5e-4f // used for geometry
#define EPS 1e-6f // used for division
#define PI 3.14159265359
#define TWO_PI 6.28318530718
#define INV_PI 0.31830988618

// 向量结构
struct float3 {
    float x, y, z;
    
    float3() : x(0), y(0), z(0) {}
    float3(float v) : x(v), y(v), z(v) {}
    float3(float x_, float y_, float z_) : x(x_), y(y_), z(z_) {}

    float3 operator-() const { return float3(-x , -y , -z); }
    
    float3 operator+(const float3& b) const { return float3(x + b.x, y + b.y, z + b.z); }
    float3 operator-(const float3& b) const { return float3(x - b.x, y - b.y, z - b.z); }
    float3 operator*(const float3& b) const { return float3(x * b.x, y * b.y, z * b.z); }
    float3 operator/(const float3& b) const { return float3(x / b.x, y / b.y, z / b.z); }
    
    float3 operator*(float b) const { return float3(x * b, y * b, z * b); }
    float3 operator/(float b) const { return float3(x / b, y / b, z / b); }
    
    float3& operator+=(const float3& b) { x += b.x; y += b.y; z += b.z; return *this; }
    float3& operator-=(const float3& b) { x -= b.x; y -= b.y; z -= b.z; return *this; }
    float3& operator*=(const float3& b) { x *= b.x; y *= b.y; z *= b.z; return *this; }
    float3& operator/=(const float3& b) { x /= b.x; y /= b.y; z /= b.z; return *this; }
    
    float operator[](int i) const { return i == 0 ? x : (i == 1 ? y : z); }
    float& operator[](int i) { return i == 0 ? x : (i == 1 ? y : z); }
};

float3 operator*(float a, const float3 b) { 
    return float3(a * b.x, a * b.y, a * b.z); 
}

struct float2 {
    float x, y;
    float2(float x_, float y_) : x(x_), y(y_) {}
};

// 材料结构
struct Material {
    float3 base_color;
    float roughness;
    float metallic;
    uint32_t light_index;

    float3 emission;        // 自发光颜色
    float ior;              // 折射率
    float transparency;     // 透明度
    int texture_id;       
  
    float subsurface;       // 次表面散射
    float specular;         // 镜面反射强度
    float specular_tint;    // 镜面反射
    float anisotropic;      // 各向异性[-1,1]
    float sheen;            // 光泽层
    float sheen_tint;       // 光泽层染色
    float clearcoat;        // 清漆层强度
    float clearcoat_roughness; // 清漆层粗糙度

    uint32_t group_id;

    Material() : 
        base_color(0.8f, 0.8f, 0.8f), 
        roughness(0.5f), 
        metallic(0.0f), 
        light_index(0xFFFFFFFF),
        emission(float3(0.0f, 0.0f, 0.0f)),
        ior(1.0f),
        transparency(0.0f),
        texture_id(-1),  // -1 表示没有纹理
        subsurface(0.0f),
        specular(0.0f),
        specular_tint(0.0f),
        anisotropic(0.0f),
        sheen(0.0f),
        sheen_tint(0.0f),
        clearcoat(0.0f),
        clearcoat_roughness(0.0f),
        group_id(0) {}
    
    
    Material(const float3& color, float rough = 0.5f, float metal = 0.0f, 
             uint32_t index = 0xFFFFFFFF, const float3& emit = float3(0.0f, 0.0f, 0.0f),
             float refractive_index = 1.0f, float trans = 0.0f, int tex_id = -1,
             float sub = 0.0f, float spec = 0.0f, float spec_tint = 0.0f,
             float aniso = 0.0f, float sh = 0.0f, float sh_tint = 0.0f,
             float coat = 0.0f, float coat_rough = 0.0f) : 
        base_color(color), 
        roughness(rough), 
        metallic(metal), 
        light_index(index),
        emission(emit),
        ior(refractive_index),
        transparency(trans),
        texture_id(tex_id),
        subsurface(sub),
        specular(spec),
        specular_tint(spec_tint),
        anisotropic(aniso),
        sheen(sh),
        sheen_tint(sh_tint),
        clearcoat(coat),
        clearcoat_roughness(coat_rough),
        group_id(0) {}
};

uint32_t pcg_hash(uint32_t& state) {
    state = state * 747796405u + 2891336453u;
    uint32_t word = ((state >> ((state >> 28u) + 4u)) ^ state) * 277803737u;
    return (word >> 22u) ^ word;
}

float random(uint32_t& seed) {
    return (float)pcg_hash(seed) / 4294967296.0;
}

float2 random2(uint32_t& seed) {
    return float2(random(seed), random(seed));
}

float dot(const float3& a, const float3& b) {
    return a.x * b.x + a.y * b.y + a.z * b.z;
}

float max(float a, float b) { return a > b ? a : b; }
float min(float a, float b) { return a < b ? a : b; }

float clamp(float x, float min_val, float max_val) {
    return min(max(x, min_val), max_val);
}

float3 clamp(const float3& v, float min_val, float max_val) {
    return float3(
        clamp(v.x, min_val, max_val),
        clamp(v.y, min_val, max_val),
        clamp(v.z, min_val, max_val)
    );
}


float3 cross(const float3& a, const float3& b) {
    return float3(
        a.y * b.z - a.z * b.y,
        a.z * b.x - a.x * b.z,
        a.x * b.y - a.y * b.x
    );
}

float length(const float3& v) {
    return sqrt(v.x * v.x + v.y * v.y + v.z * v.z);
}

float3 normalize(const float3& v) {
    float len = length(v);
    if (len > 0.0f) return v / len;
    return v;
}

float lerp(float a, float b, float t) {
    return a + (b - a) * t;
}

float3 lerp(const float3& a, const float3& b, float t) {
    return a + (b - a) * t;
}

float3 reflect(const float3& I, const float3& N) {
    return I - N * 2.0f * dot(I, N);
}


// 随机数生成器
float f3_max( float3 u){
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
bool Refract(float3 v, float3 n, float eta, float3& t) {
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
  return F0 + (float3(1.0,1.0,1.0) - F0) * SchlickWeight(cosTheta);
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
float3 SampleHemisphereCos(float2 randd,  float3 normal)
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

float3 SampleGGX_VNDF( float3 ray, float roughness, float2 rd,  float3 normal)
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
    float3 H_local = normalize(float3(ax * Nh.x, ay * Nh.y, max(0.0f, Nh.z)));
    float3 H_world = H_local.x * tangent + H_local.y * bitangent + H_local.z * normal;
    return H_world;
}

void GetTangent( float3 normal, float3& tangent, float3& bitangent)
{
  float3 up=abs(normal.z)<0.999?float3(0.0,0.0,1.0):float3(1.0,0.0,0.0);
  tangent=normalize(cross(up,normal));
  bitangent=cross(normal,tangent);
}

// --- 新增的透射相关辅助函数 (修正版) ---


float3 SampleGGX_Distribution(float roughness, float2 rd, float3& normal, float3& tangent, float3& bitangent)
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


bool SampleBSDF(Material& mat, float3& ray, float3& normal, float3& wi, bool is_flip, uint32_t& seed){

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

float3 EvalBSDF( Material mat,  float3 ray,  float3 wi,  float3 normal, bool is_flip, float& pdf)
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
    float3 H = n1 * V + n2 * L;
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
    float3 T_color = float3(1.0, 1.0, 1.0) - F_color; // 透射颜色
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
      ret+=diffuseweight*diffuse*(float3(1.0, 1.0, 1.0)-F);
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
            spec=D*Vis*float3(1.0,1.0,1.0);
            if (mat.specular_tint > EPS)spec*=lerp(float3(1.0, 1.0, 1.0), mat.base_color, mat.specular_tint);
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
        ret += sheenweight * sheenTerm * (float3(1.0,1.0,1.0) - F);
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

void rand(float3& u, uint32_t& sd){
    u = float3(random(sd), random(sd), random(sd));
    u = normalize(u);
}


void auto_test(Material mat) {

    mt19937 rd(time(NULL));

    uint32_t seed = rd();

    int num = 0;

    float3 normal, wi, ray;
    float ratio, max = 0;

    while(1){
        seed = ++num * rd();
        rand(normal, seed);

        rand(ray, seed);

        rand(wi, seed);

        ray = normalize(cross(ray, normal));

        wi = normalize(cross(wi, normal));

        if(rd() & 1){
            ray += random(seed) / ((rd()&1)? 1.0: 100.0) * normal;
        }
        else{
            ray = normal + random(seed) / ((rd()&1)? 1.0: 100.0) * ray;
        }

        if(rd() & 1){
            wi += random(seed) / ((rd()&1)? 1.0: 100.0) * normal;
        }
        else{
            wi = normal + random(seed) / ((rd()&1)? 1.0: 100.0) * wi;
        }
        bool is_flip = rd()&1;
        float pdf;
        ray = normalize(ray);
        wi = normalize(wi);
        float3 result = EvalBSDF(mat, ray, wi, normal, is_flip, pdf);
        if(pdf > 0.0) ratio = f3_max(result) * abs(dot(wi, normal)) / pdf;
        if(ratio > max){
            printf("%d\nType1\n", num);
            printf("float3 normal(%f, %f, %f);\n", normal.x, normal.y, normal.z);
            printf("float3 ray(%f, %f, %f);\n", ray.x, ray.y, ray.z);
            printf("float3 wi(%f, %f, %f);\n", wi.x, wi.y, wi.z);
            printf("BSDF:%f %f %f PDF:%f\n",result.x,result.y,result.z,pdf);
            printf("ratio: %f\n\n\n", f3_max(result) * abs(dot(wi, normal)) / pdf);
            max = ratio;

            // break;
        }

        if (SampleBSDF(mat, ray, normal, wi, is_flip, seed)) {
            result = EvalBSDF(mat, ray, wi, normal, is_flip, pdf);
            if(pdf > 0.0) ratio = f3_max(result) * abs(dot(wi, normal)) / pdf;
            if(ratio > max){
                printf("%d\nType2\n", num);
                printf("float3 normal(%f, %f, %f);\n", normal.x, normal.y, normal.z);
                printf("float3 ray(%f, %f, %f);\n", ray.x, ray.y, ray.z);
                printf("float3 wi(%f, %f, %f);\n", wi.x, wi.y, wi.z);
                printf("BSDF:%f %f %f PDF:%f\n",result.x,result.y,result.z,pdf);
                printf("ratio: %f\n\n\n", f3_max(result) * abs(dot(wi, normal)) / pdf);
                max = ratio;

                // break;
            }
        }
    }

}


void rotate_to_z_axis(float3& normal, float3& ray, float3& wi) {
    float3 up = (fabs(normal.z) < 0.999f) ? float3(0, 0, 1) : float3(1, 0, 0);
    float3 tangent = normalize(cross(up, normal));
    float3 bitangent = cross(normal, tangent); 
    float3 nor = normal;

    auto rotate = [&](const float3& v) {
        return float3(
            dot(tangent, v),
            dot(bitangent, v),
            dot(nor, v)
        );
    };

    normal = rotate(normal); 
    ray = rotate(ray);
    wi = rotate(wi);
}


float3 integrateBSDF_hemisphere_grid(Material& mat, float3 ray, float3 normal, bool is_flip, 
                                     int theta_steps = 5000, int phi_steps = 5000) {
    float3 integral = float3(0.0f, 0.0f, 0.0f);
    
    normal = float3(0.0f, 0.0f, 1.0f);
    
    float delta_theta = (PI / 2.0f) / theta_steps;
    float delta_phi = TWO_PI / phi_steps;
    
    for (int i = 0; i < theta_steps; i++) {
        float theta = (i + 0.5f) * delta_theta; 
        float sin_theta = sin(theta);
        float cos_theta = cos(theta);
        
        for (int j = 0; j < phi_steps; j++) {
            float phi = (j + 0.5f) * delta_phi;
            
            float3 wi_local = float3(
                sin_theta * cos(phi),
                sin_theta * sin(phi),
                cos_theta
            );
            
            float3 wi = wi_local;
            wi = normalize(wi);
            
            float pdf;
            float3 bsdf_val = EvalBSDF(mat, ray, wi, normal, is_flip, pdf);
            
            if (pdf > 0.0f) {
                float cos_theta = dot(wi, normal);
                if (cos_theta > 0.0f) {
                    float solid_angle_weight = sin_theta * delta_theta * delta_phi;
                    integral += bsdf_val * cos_theta * solid_angle_weight;
                }
            }
        }
    }
    
    return integral;
}
float3 integrateBSDF_sphere_grid(Material& mat, float3 ray, float3 normal, bool is_flip,
                                int theta_steps = 5000, int phi_steps = 5000) {
    float3 integral = float3(0.0f, 0.0f, 0.0f);
    
    normal = float3(0.0f, 0.0f, 1.0f);
    
    float delta_theta = PI / theta_steps;
    float delta_phi = TWO_PI / phi_steps;
    
    for (int i = 0; i < theta_steps; i++) {
        float theta = (i + 0.5f) * delta_theta;
        float sin_theta = sin(theta);
        float cos_theta = cos(theta);
        
        for (int j = 0; j < phi_steps; j++) {
            float phi = (j + 0.5f) * delta_phi;
            
            float3 wi_local = float3(
                sin_theta * cos(phi),
                sin_theta * sin(phi),
                cos_theta
            );
            
            float3 wi = wi_local;
            wi = normalize(wi);
            
            float pdf;
            float3 bsdf_val = EvalBSDF(mat, ray, wi, normal, is_flip, pdf);
            
            if (pdf > 0.0f) {
                float cos_theta = abs(dot(wi, normal));
                float solid_angle_weight = sin_theta * delta_theta * delta_phi;
                integral += bsdf_val * cos_theta * solid_angle_weight;
            }
        }
    }
    
    return integral;
}

float3 integrateBSDF_importance(Material& mat, float3 ray, float3 normal, bool is_flip, uint32_t seed, int num_samples = 10000000) {
    float3 integral = float3(0.0f, 0.0f, 0.0f);
    int valid_samples = 0;
    
    for (int i = 0; i < num_samples; i++) {
        float3 wi = ray;
        if (SampleBSDF(mat, ray, normal, wi, is_flip, seed)) {
            float pdf;
            float3 bsdf_val = EvalBSDF(mat, ray, wi, normal, is_flip, pdf);
            
            if (pdf > 0.0f) {
                float cos_theta = abs(dot(wi, normal));
                integral += bsdf_val * cos_theta / pdf;
                valid_samples++;
            }
        }
    }
    
    if (valid_samples > 0) {
        return integral / float(valid_samples);
    }
    return float3(0.0f, 0.0f, 0.0f);
}

void verifyPDF_grid(Material& mat, float3 ray, float3 normal, bool is_flip) {
    float pdf_sum = 0.0f;
    int valid_samples = 0;
    
    normal = float3(0.0f, 0.0f, 1.0f);
    
    const int theta_steps = 5000;
    const int phi_steps = 5000;
    
    const float delta_theta = PI / theta_steps;
    const float delta_phi = TWO_PI / phi_steps;
    
    for (int i = 0; i < theta_steps; i++) {
        float theta = (i + 0.5f) * delta_theta;
        float sin_theta = sin(theta);
        
        for (int j = 0; j < phi_steps; j++) {
            float phi = (j + 0.5f) * delta_phi;
            
            float3 wi_local = float3(
                sin_theta * cos(phi),
                sin_theta * sin(phi),
                cos(theta)
            );
            
            float3 wi = wi_local;
            wi = normalize(wi);
            
            float pdf;
            EvalBSDF(mat, ray, wi, normal, is_flip, pdf);
            
            if (pdf > 0.0f) {
                float solid_angle = sin_theta * delta_theta * delta_phi;
                pdf_sum += pdf * solid_angle;
                valid_samples++;
            }
        }
    }
    
    printf("PDF验证（网格采样 5000x5000）:\n");
    printf("  - 总采样点: %d\n", theta_steps * phi_steps);
    printf("  - 有效采样点: %d\n", valid_samples);
    printf("  - PDF积分值: %.6f (期望: 1.000000)\n", pdf_sum);
    printf("  - 偏差: %.6f\n", abs(pdf_sum - 1.0f));
    
}

void testEnergyConservation(Material& mat, float3 ray, float3 normal, bool is_flip) {
    printf("==============================================================\n");
    printf("                 BSDF 网格采样积分测试                         \n");
    printf("==============================================================\n\n");
    
    printf("【材质与场景参数】\n");
    printf("--------------------------------------------------------------\n");
    printf("材质颜色: (%.6f, %.6f, %.6f)\n", mat.base_color.x, mat.base_color.y, mat.base_color.z);
    printf("粗糙度: %.6f\n", mat.roughness);
    printf("金属度: %.6f\n", mat.metallic);
    printf("透明度: %.6f\n", mat.transparency);
    printf("折射率 (IOR): %.6f\n", mat.ior);
    printf("镜面强度: %.6f\n", mat.specular);
    printf("清漆层强度: %.6f, 清漆粗糙度: %.6f\n", mat.clearcoat, mat.clearcoat_roughness);
    printf("光泽层强度: %.6f, 光泽层染色: %.6f\n", mat.sheen, mat.sheen_tint);
    printf("各向异性: %.6f\n", mat.anisotropic);
    printf("入射方向: (%.6f, %.6f, %.6f)\n", ray.x, ray.y, ray.z);
    printf("法线方向: (0.000000, 0.000000, 1.000000) [固定]\n");
    printf("法线朝向: %s\n", is_flip ? "翻转" : "正常");
    printf("入射角 cosθ: %.6f\n", dot(ray, normal));
    printf("--------------------------------------------------------------\n\n");
    
    printf("【半球网格积分（反射方向）】\n");
    printf("--------------------------------------------------------------\n");
    printf("网格分辨率: 5000 × 5000\n");
    
    auto start = chrono::high_resolution_clock::now();
    float3 integral_hemisphere = integrateBSDF_hemisphere_grid(mat, ray, normal, is_flip);
    auto end = chrono::high_resolution_clock::now();
    chrono::duration<double> elapsed = end - start;
    
    float luminance_hemisphere = 0.2126f * integral_hemisphere.x + 
                                 0.7152f * integral_hemisphere.y + 
                                 0.0722f * integral_hemisphere.z;
    
    printf("积分结果 (RGB): (%.6f, %.6f, %.6f)\n", 
           integral_hemisphere.x, integral_hemisphere.y, integral_hemisphere.z);
    printf("亮度 (Y): %.6f\n", luminance_hemisphere);
    printf("计算时间: %.2f 秒\n", elapsed.count());
    printf("--------------------------------------------------------------\n\n");
    
    printf("【球面网格积分（全方向）】\n");
    printf("--------------------------------------------------------------\n");
    printf("网格分辨率: 5000 × 5000\n");
    
    start = chrono::high_resolution_clock::now();
    float3 integral_sphere = integrateBSDF_sphere_grid(mat, ray, normal, is_flip);
    end = chrono::high_resolution_clock::now();
    elapsed = end - start;
    
    float luminance_sphere = 0.2126f * integral_sphere.x + 
                             0.7152f * integral_sphere.y + 
                             0.0722f * integral_sphere.z;
    
    printf("积分结果 (RGB): (%.6f, %.6f, %.6f)\n", 
           integral_sphere.x, integral_sphere.y, integral_sphere.z);
    printf("亮度 (Y): %.6f\n", luminance_sphere);
    printf("计算时间: %.2f 秒\n", elapsed.count());
    printf("--------------------------------------------------------------\n\n");
    
    printf("【能量分布】\n");
    printf("--------------------------------------------------------------\n");
    
    if (mat.transparency > 0.0f) {
        float reflectance = luminance_hemisphere;
        float transmittance = max(0.0f, luminance_sphere - luminance_hemisphere);
        
        printf("透明材质:\n");
        printf("总能量输出 (球面积分): %.6f\n", luminance_sphere);
        printf("反射能量 (半球积分): %.6f\n", reflectance);
        printf("透射能量: %.6f\n", transmittance);
        printf("反射率: %.6f\n", reflectance);
        printf("透射率: %.6f\n", transmittance);
        printf("R + T = %.6f\n", reflectance + transmittance);
    } else {
        float reflectance = luminance_hemisphere;
        
        printf("非透明材质:\n");
        printf("反射能量: %.6f\n", reflectance);
        printf("反射率: %.6f\n", reflectance);
    }
    printf("--------------------------------------------------------------\n\n");
    
    printf("【PDF归一化验证】\n");
    printf("--------------------------------------------------------------\n");
    
    start = chrono::high_resolution_clock::now();
    verifyPDF_grid(mat, ray, normal, is_flip);
    end = chrono::high_resolution_clock::now();
    elapsed = end - start;
    
    printf("验证时间: %.2f 秒\n", elapsed.count());
    printf("--------------------------------------------------------------\n");
    
    printf("\n测试完成。\n");
}


void test(Material mat) {

    uint32_t seed = 1;

    
    // float3 normal(0.0, 0.0, 1.0);
    // float3 ray(0.0f, 1.0f, 0.00001f); 
    // float3 wi(0.0f, 1.0f, -0.001f);  

    bool is_flip = false;
    float pdf;

    // float3 normal(0.000000, -0.000000, 1.000000);
    // float3 ray(0.513823, -0.857724, 0.017180);
    // float3 wi(-0.537973, 0.842871, -0.012399);
    // float3 normal(-0.000000, 0.000000, 1.000000);
    // float3 ray(0.837933, -0.545768, 0.002084);
    // float3 wi(0.353021, -0.935612, -0.002188);
    float3 normal(-0.000000, 0.000000, 1.000000);
    // float3 ray(0.837933, -0.545768, 0.002084);
    float3 ray(0.0, 1.0, 0.1);
    float3 wi(0.0, -1.0, 0.1);

    ray = normalize(ray);
    wi = normalize(wi);


    // rotate_to_z_axis(normal, ray, wi);
    printf("float3 normal(%f, %f, %f);\n", normal.x, normal.y, normal.z);
    printf("float3 ray(%f, %f, %f);\n", ray.x, ray.y, ray.z);
    printf("float3 wi(%f, %f, %f);\n", wi.x, wi.y, wi.z);

    // printf("%f\n", dot(ray, normal));
    // printf("%f\n", dot(wi, normal));

    float3 result;
    float ratio;

    is_flip = true;

    // SampleBSDF(mat, ray, normal, wi, is_flip, seed);
    result = EvalBSDF(mat, ray, wi, normal, is_flip, pdf);
    ratio = f3_max(result) * abs(dot(wi, normal)) / pdf;
    printf("BSDF值: (%f, %f, %f)\n", result.x, result.y, result.z);
    printf("PDF: %f\nratio: %f\n\n", pdf, ratio);
    
    
    printf("=== 测试采样 ===\n");
    for (int i = 0; i < 100; i++) {
        if (SampleBSDF(mat, ray, normal, wi, is_flip, seed)) {
            result = EvalBSDF(mat, ray, wi, normal, is_flip, pdf);
            printf("采样 %d: wi = (%f, %f, %f)\n", i+1, wi.x, wi.y, wi.z);
            printf("BSDF:%f %f %f PDF:%f\n",result.x,result.y,result.z,pdf);
            printf("ratio: %f\n\n\n", f3_max(result) * abs(dot(wi, normal)) / pdf);
        }
    }
}

int main() {

    
    Material mat = Material(float3(0.99f, 0.98f, 0.99f), 0.05f, 0.95f,
            0xFFFFFFFF, float3(0, 0, 0), 1.05, 1.0, -1, 0.0, 0.7, 0.1, 0.0);
    // Material mat = Material(float3(0.99f, 0.98f, 0.99f), 0.1f, 0.9f,
    //         0xFFFFFFFF, float3(0, 0, 0), 1.05, 0.0, -1, 0.0, 0.0, 0.0, 0.0);
   
    float3 normal(-0.000000, 0.000000, 1.000000);
    // float3 ray(0.837933, -0.545768, 0.002084);
    float3 ray(0.0, 1.0, 0.1);

    ray = normalize(ray);

    testEnergyConservation(mat, ray, normal, 0);


    // test(mat);

    mat.transparency = 0.0;

    test(mat);

    return 0;
}
