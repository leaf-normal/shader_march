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
    uint normal_offset;
    uint index_offset;
    uint vertex_count;
    uint index_count;
    uint material_index;
};

// *add
StructuredBuffer<GeometryDescriptor> geometry_descriptors : register(t0, space8);
StructuredBuffer<float3> vertex_positions : register(t1, space8);
StructuredBuffer<uint> indices : register(t2, space8);

struct RayPayload {
  float3 color;
  bool hit;
  uint instance_id;
  float3 normal;
  float3 hit_point;
};

#define MAX_DEPTH 16
#define RR_THRESHOLD 0.9f
#define t_min 0.001
#define t_max 10000.0
#define eps 5e-4
#define PI 3.141592654

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
float sqr(float x)
{
  return x*x;
}
float3 sqr(float3 x)
{
  return x*x;
}

float Schlickweight(float cosTheta)
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
  }else if(randLobe<1)//清漆
  {
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
    float3 F=SchlickFresnel(F0,dot(H,ray));
    float denom=sqr(Hdotray+eta*Hdotwi);
    float3 transmission=(material.base_color*(1.0-F)*D*G*abs(Hdotwi)*abs(Hdotray))/(abs(Ndotray)*abs(Ndotwi)*denom);
    ret+=transmission*material.transparency*material.specular_transmission;
    float jacobian=(eta*eta*abs(Hdotwi))/denom;
    transmissionPDF=D*NdotH*jacobian;
  }else{
    float3 H=normalize(ray+wi);
    float NdotH=dot(normal,H);
    float Hdotray=dot(H,ray);
    float3 F0=lerp(0.08*material.specular,material.base_color,material.metallic);
    float3 F=SchlickFresnel(F0,Hdotray);
    //漫反射
    if(material.metallic<1.0&&material.transparency<1.0)
    {
      float FL=Schlickweight(Ndotray);
      float FV=Schlickweight(Ndotwi);
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
    ret+=spec;
    //清漆层
    if(material.clearcoat>0.0)
    {
      float clearcoat_alpha=material.clearcoat_roughness*material.clearcoat_roughness;
      float D_clearcoat=GTR1(NdotH,clearcoat_alpha);
      float G_clearcoat=SmithG_GGX(Ndotray,material.clearcoat_roughness)*SmithG_GGX(Ndotwi,material.clearcoat_roughness);
      float3 F_clearcoat=SchlickFresnel(float3(0.04,0.04,0.04),Hdotray); 
      float3 clearcoat=(D_clearcoat*G_clearcoat*F_clearcoat)/(4.0 * Ndotray * Ndotwi);
      ret+=clearcoat*material.clearcoat;
      clearcoatPDF=GTR1(NdotH,clearcoat_alpha)*NdotH/(4.0*Hdotray);
    }
    //光泽层
    if(material.sheen>0.0&&material.metallic<1.0)
    {
      float3 sheen_color=lerp(float3(1.0,1.0,1.0),material.base_color,material.sheen_tint);
      float sheen_intensity=material.sheen*(1.0-material.metallic);
      float3 sheen_F=SchlickFresnel(sheen_color,Hdotray);
      ret+=sheen_F*sheen_intensity*(1.0-F);
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
    // SampleBSDF(mat,-ray.Direction,payload.normal,wi,seed);
    float3 bsdf=EvalBSDF(material,-ray.Direction,wi,payload.normal,pdf);
    // if(pdf <= 0.0) break;

    float cosTheta = dot(wi, payload.normal);

    throughout *= bsdf * abs(cosTheta) / pdf;

    ray.Origin = payload.hit_point + eps * payload.normal;
    ray.Direction = normalize( reflect(-ray.Direction, payload.normal) );
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
  
  // 1. 获取实例ID和基元ID
  uint instance_id = InstanceID();
  uint primitive_id = PrimitiveIndex();
  payload.instance_id = instance_id;
  
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
  float3 v0 = vertex_positions[geo_desc.vertex_offset + i0];
  float3 v1 = vertex_positions[geo_desc.vertex_offset + i1];
  float3 v2 = vertex_positions[geo_desc.vertex_offset + i2];

  payload.normal = normalize( mul( (float3x3)transpose( WorldToObject4x3() ), cross(v1 - v0, v2 - v0) ) );

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
    float3 ambient = mat.base_color * 0.05;
    
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