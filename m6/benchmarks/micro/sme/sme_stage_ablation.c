/* paper_m6 Stage 5 probe. Build: clang -O3 -mcpu=apple-m4 benchmarks/micro/sme/sme_stage_ablation.c -o /tmp/sme_stage_ablation */
#include <arm_sme.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <pthread.h>
static double now(void){return clock_gettime_nsec_np(CLOCK_UPTIME_RAW)*1e-9;}
#define N 4096
/* MODE bits: 1 = loads, 2 = fmopa, 4 = stores */
#define DEF(name, MODE) \
__arm_locally_streaming __arm_new("za") static void name(const float*in,float*out,int ns,const float*mats,int reps){ \
 const svbool_t pg=svptrue_b32(); const int m=N/(8*ns); const size_t istr=(size_t)(N/8)*32; svfloat32_t fx=svdup_f32(1.f); \
 for(int rp=0;rp<reps;rp++) for(int t=0;t<ns;t++){ const float*M=mats+(size_t)t*256; \
  for(int u=0;u<m;u+=4){ const int j0=t+u*ns; const float*x0=in+(size_t)j0*32; const size_t js=(size_t)ns*32; svzero_za(); \
   for(int k=0;k<16;k++){ svfloat32_t zm=svld1_f32(pg,M+k*16); const float*xk=x0+(k>>1)*istr+(k&1)*16; \
    svfloat32_t a0=fx,a1=fx,a2=fx,a3=fx; \
    if(MODE&1){a0=svld1_f32(pg,xk);a1=svld1_f32(pg,xk+js);a2=svld1_f32(pg,xk+2*js);a3=svld1_f32(pg,xk+3*js);} \
    if(MODE&2){svmopa_za32_f32_m(0,pg,pg,zm,a0);svmopa_za32_f32_m(1,pg,pg,zm,a1);svmopa_za32_f32_m(2,pg,pg,zm,a2);svmopa_za32_f32_m(3,pg,pg,zm,a3);} \
    else if(MODE&1){ svwrite_hor_za32_f32_m(0,k,pg,svadd_f32_x(pg,svadd_f32_x(pg,a0,a1),svadd_f32_x(pg,a2,a3))); } } \
   if(MODE&4){ const int d0=u*8*ns+t; for(int v=0;v<4;v++){ float*o=out+(size_t)(d0+v*8*ns)*32; for(int i=0;i<16;i++){ \
     if(v==0)svst1_hor_za32(0,i,pg,o+(size_t)(i>>1)*ns*32+(i&1)*16); if(v==1)svst1_hor_za32(1,i,pg,o+(size_t)(i>>1)*ns*32+(i&1)*16); \
     if(v==2)svst1_hor_za32(2,i,pg,o+(size_t)(i>>1)*ns*32+(i&1)*16); if(v==3)svst1_hor_za32(3,i,pg,o+(size_t)(i>>1)*ns*32+(i&1)*16);}}} \
  }} float s_[16];svst1_hor_za32(0,0,pg,s_);__asm__ volatile(""::"r"(s_):"memory"); }
DEF(f7,7) DEF(f3,3) DEF(f6,6) DEF(f5,5) DEF(f2,2) DEF(f1,1) DEF(f4,4)
int main(){pthread_set_qos_class_self_np(QOS_CLASS_USER_INTERACTIVE,0);
 float*in=aligned_alloc(16384,N*128),*out=aligned_alloc(16384,N*128),*mats=aligned_alloc(64,512*1024);
 for(int i=0;i<N*32;i++){in[i]=(i%97)*1e-3f;out[i]=0;} for(int i=0;i<512*256;i++)mats[i]=(i%13)*1e-2f;
 void(*fs[])(const float*,float*,int,const float*,int)={f7,f3,f6,f5,f2,f1,f4}; const char*nm[]={"ld+mopa+st","ld+mopa","mopa+st","ld+st","mopa only","ld only","st only"};
 f7(in,out,8,mats,2000);
 for(int ns=1;ns<=64;ns*=8){ printf("stage ns=%d (ns per group of 4 butterflies x16 FFTs):\n",ns);
 for(int v=0;v<7;v++){ double best=1e9; int reps=200; for(int q=0;q<7;q++){double t0=now();fs[v](in,out,ns,mats,reps);double d=now()-t0;if(d<best)best=d;}
  printf("  %-12s %6.1f ns/group\n",nm[v],best/reps/(N/32)*1e9);}}
}
