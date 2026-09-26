/* paper_m6 Stage 5 probe. Build: clang -O3 -mcpu=apple-m4 benchmarks/micro/sme/sme_load_variants.c -o /tmp/sme_load_variants */
#include <arm_sme.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <pthread.h>
static double now(void){return clock_gettime_nsec_np(CLOCK_UPTIME_RAW)*1e-9;}
#define PG svptrue_b32()
#define SINK float s_[16];svst1_hor_za32(0,0,PG,s_);__asm__ volatile(""::"r"(s_):"memory");
/* (a) single ld1w into Z, consumed by fmopa (4 tiles) */
__arm_locally_streaming __arm_new("za") static void la(const float*b,size_t nf,int reps){svfloat32_t w=svdup_f32(1);svzero_za();
 for(int r=0;r<reps;r++)for(size_t i=0;i<nf;i+=64){svmopa_za32_f32_m(0,PG,PG,w,svld1_f32(PG,b+i));svmopa_za32_f32_m(1,PG,PG,w,svld1_f32(PG,b+i+16));svmopa_za32_f32_m(2,PG,PG,w,svld1_f32(PG,b+i+32));svmopa_za32_f32_m(3,PG,PG,w,svld1_f32(PG,b+i+48));} SINK}
/* (b) SME2 ld1w x4 consecutive into Z, consumed by fmopa */
__arm_locally_streaming __arm_new("za") static void lb(const float*b,size_t nf,int reps){svfloat32_t w=svdup_f32(1);svzero_za();svcount_t pn=svptrue_c32();
 for(int r=0;r<reps;r++)for(size_t i=0;i<nf;i+=64){svfloat32x4_t x=svld1_f32_x4(pn,b+i);svmopa_za32_f32_m(0,PG,PG,w,svget4(x,0));svmopa_za32_f32_m(1,PG,PG,w,svget4(x,1));svmopa_za32_f32_m(2,PG,PG,w,svget4(x,2));svmopa_za32_f32_m(3,PG,PG,w,svget4(x,3));} SINK}
/* (c) ld1w into ZA horizontal slices */
__arm_locally_streaming __arm_new("za") static void lc(const float*b,size_t nf,int reps){svzero_za();
 for(int r=0;r<reps;r++)for(size_t i=0;i<nf;i+=256)for(int s=0;s<16;s++)svld1_hor_za32(0,s,PG,b+i+s*16); SINK}
/* (d) ld1w x4 only, no fmopa: accumulate nothing, just keep last */
__arm_locally_streaming __arm_new("za") static void ld(const float*b,size_t nf,int reps){svzero_za();svcount_t pn=svptrue_c32();svfloat32_t w=svdup_f32(1);
 for(int r=0;r<reps;r++)for(size_t i=0;i<nf;i+=256){svfloat32x4_t x0=svld1_f32_x4(pn,b+i),x1=svld1_f32_x4(pn,b+i+64),x2=svld1_f32_x4(pn,b+i+128),x3=svld1_f32_x4(pn,b+i+192);
  svmopa_za32_f32_m(0,PG,PG,w,svget4(x0,0));svmopa_za32_f32_m(1,PG,PG,w,svget4(x1,1));svmopa_za32_f32_m(2,PG,PG,w,svget4(x2,2));svmopa_za32_f32_m(3,PG,PG,w,svget4(x3,3));} SINK}
int main(){pthread_set_qos_class_self_np(QOS_CLASS_USER_INTERACTIVE,0);
 printf("%10s %8s %8s %8s %8s  GB/s\n","bytes","ld+mopa","ldx4+mopa","ld->ZA","ldx4 1/4mopa");
 for(size_t by=64<<10;by<=(64<<20);by*=4){size_t nf=by/4;float*b=aligned_alloc(16384,by);memset(b,0,by);int reps=(int)((256<<20)/by);if(reps<2)reps=2;
  double t[4];for(int k=0;k<4;k++){ if(k==0)la(b,nf,1);if(k==1)lb(b,nf,1);if(k==2)lc(b,nf,1);if(k==3)ld(b,nf,1);
   double best=1e9;for(int q=0;q<5;q++){double t0=now();if(k==0)la(b,nf,reps);if(k==1)lb(b,nf,reps);if(k==2)lc(b,nf,reps);if(k==3)ld(b,nf,reps);double d=now()-t0;if(d<best)best=d;}t[k]=by*(double)reps/best*1e-9;}
  printf("%10zu %8.1f %8.1f %8.1f %8.1f\n",by,t[0],t[1],t[2],t[3]);free(b);}
}
