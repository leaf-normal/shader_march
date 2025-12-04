struct Material {
  float3 base_color;
  float roughness;
  float metallic;

  float3 emission;        // 自发光颜色
  float ior;                // 折射率
  float transparency;        
  int material_type;        // 材质类型: 0=漫反射, 1=镜面, 2=玻璃, 3=发射
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


#define MAX_DEPTH 8
#define RR_THRESHOLD 0.9f
#define t_min 0.001
#define t_max 10000.0
#define eps 5e-4
#define PI 3.141592654

uint pcg_hash(inout uint state)
{
    state = state * 747796405u + 2891336453u;
    uint word = ((state >> ((state >> 28u) + 4u)) ^ state) * 277803737u;
    return (word >> 22u) ^ word;
}
uint generate_seed(uint2 pixel, uint frame)
{
    return (pixel.x * 1973u + pixel.y * 9277u + frame * 26699u) ^ 0x6f5ca34du;
}

float random(inout uint seed)
{
  return (float)pcg_hash(seed) / 4294967296.0;
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
    float3 F0=lerp(0.08*material.specular,material.base_color,material.metallic);
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
