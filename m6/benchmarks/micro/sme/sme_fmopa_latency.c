/* paper_m6 Stage 5 probe. Build: clang -O3 -mcpu=apple-m4 benchmarks/micro/sme/sme_fmopa_latency.c -o /tmp/sme_fmopa_latency */
#include <arm_sme.h>
#include <stdio.h>
#include <time.h>
#include <pthread.h>
static double now(void){return clock_gettime_nsec_np(CLOCK_UPTIME_RAW)*1e-9;}
#define PG svptrue_b32()
__arm_locally_streaming __arm_new("za") static void t1(long n){svfloat32_t a=svdup_f32(1),b=svdup_f32(.5f);svzero_za();for(long i=0;i<n;i++){svmopa_za32_f32_m(0,PG,PG,a,b);svmopa_za32_f32_m(0,PG,PG,b,a);svmopa_za32_f32_m(0,PG,PG,a,a);svmopa_za32_f32_m(0,PG,PG,b,b);}float s[16];svst1_hor_za32(0,0,PG,s);__asm__ volatile(""::"r"(s):"memory");}
__arm_locally_streaming __arm_new("za") static void t2(long n){svfloat32_t a=svdup_f32(1),b=svdup_f32(.5f);svzero_za();for(long i=0;i<n;i++){svmopa_za32_f32_m(0,PG,PG,a,b);svmopa_za32_f32_m(1,PG,PG,b,a);svmopa_za32_f32_m(0,PG,PG,a,a);svmopa_za32_f32_m(1,PG,PG,b,b);}float s[16];svst1_hor_za32(0,0,PG,s);__asm__ volatile(""::"r"(s):"memory");}
__arm_locally_streaming __arm_new("za") static void t4(long n){svfloat32_t a=svdup_f32(1),b=svdup_f32(.5f);svzero_za();for(long i=0;i<n;i++){svmopa_za32_f32_m(0,PG,PG,a,b);svmopa_za32_f32_m(1,PG,PG,b,a);svmopa_za32_f32_m(2,PG,PG,a,a);svmopa_za32_f32_m(3,PG,PG,b,b);}float s[16];svst1_hor_za32(0,0,PG,s);__asm__ volatile(""::"r"(s):"memory");}
/* fp16->fp32 widening fmopa (2-way): 1024 flop */
__arm_locally_streaming __arm_new("za") static void h4(long n){svfloat16_t a=svdup_f16(1),b=svdup_f16(.5f);svzero_za();for(long i=0;i<n;i++){svmopa_za32_f16_m(0,PG,PG,a,b);svmopa_za32_f16_m(1,PG,PG,b,a);svmopa_za32_f16_m(2,PG,PG,a,a);svmopa_za32_f16_m(3,PG,PG,b,b);}float s[16];svst1_hor_za32(0,0,PG,s);__asm__ volatile(""::"r"(s):"memory");}
/* SME2 fmla vgx4 into ZA vector groups: 4 vectors x16 FMA = 128 flop, 8 independent groups */
__arm_locally_streaming __arm_new("za") static void v4(long n){svfloat32x4_t a=svcreate4(svdup_f32(1),svdup_f32(2),svdup_f32(3),svdup_f32(4));svfloat32x4_t b=svcreate4(svdup_f32(.5f),svdup_f32(.25f),svdup_f32(.1f),svdup_f32(.2f));svzero_za();for(long i=0;i<n;i++){for(int g=0;g<16;g++) svmla_za32_f32_vg1x4(g,a,b);}float s[16];svst1_hor_za32(0,0,PG,s);__asm__ volatile(""::"r"(s):"memory");}
int main(){pthread_set_qos_class_self_np(QOS_CLASS_USER_INTERACTIVE,0);long n=20000000;
double t;
t4(40000000);t1(1000);t=now();t1(n);t=now()-t;printf("1 tile : %.0f GF, %.2f ns/fmopa\n",n*4*512/t*1e-9,t/(n*4)*1e9);
t=now();t2(n);t=now()-t;printf("2 tiles: %.0f GF\n",n*4*512/t*1e-9);
t=now();t4(n);t=now()-t;printf("4 tiles: %.0f GF\n",n*4*512/t*1e-9);
t=now();h4(n);t=now()-t;printf("4 tiles f16->f32 widening: %.0f GF\n",n*4*1024/t*1e-9);
t=now();v4(n/4);t=now()-t;printf("fmla vgx4 x16 groups: %.0f GF\n",n/4*16*128/t*1e-9);
}
