/*
 * The out-of-line exact path of the SPU_FLOAT_RZ x86 float helpers (spu_helpers.h,
 * spu__xr_chk2/3), reached through an assembly stub that keeps every register.
 *
 * Why a stub: the fast path of fa/fm/fma is a handful of SSE ops plus a range check, and the
 * lanes that fail it (an extended-range operand, an overflow) are rare. With a plain C call on the
 * failing branch, every xmm register is call-clobbered on x86-64, so the compiler kept every live
 * vector in a stack slot around every float op of a lifted function, though the call almost never
 * runs (LBP2's hottest job function: ~500 stack moves for ~95 SPU instructions). The inline asm of
 * spu__xr_chk2/3 changes only the result and one GPR, so the fast path keeps its registers; the stub
 * pays for that by saving every vector register (the full ymm registers when the OS has enabled
 * AVX, so their upper halves survive whatever the C code here or libm uses; else the xmm
 * registers) and every caller-saved GPR. Some code takes this path often (FMOD's DSP plugins in
 * LBP2: ~5% of the process with XSAVE/XRSTOR here), so it is plain moves, not XSAVE. MXCSR and
 * the x87 state are not saved: nothing below changes them.
 *
 * Contract (see spu__xr_chk2/3 in spu_helpers.h): called from cold code with a 96-byte block at
 * the caller's rsp (below its red zone): r +0 (the fast result), ok +16 (the per-lane ok mask),
 * a +32, b +48, c +64 (fma only), op +80 (u32). Writes the fixed result to block +0; every
 * register as it was.
 */
#ifndef _GNU_SOURCE
#define _GNU_SOURCE   /* dladdr */
#endif
#include "spu_helpers.h"
#include <string.h>

#if SPU__XR_FIXSTUB
#include <cpuid.h>
#include <dlfcn.h>
#include <pthread.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>

/* 1 when the OS has enabled AVX state (XCR0 bits 1-2): the stub then saves the full ymm
 * registers, else the xmm registers. */
__attribute__((visibility("hidden"))) unsigned char spu__xr_ymm_ok;
static unsigned char spu__xr_use_v3;   /* spu__xr_fix_v3 runs on this host */

__attribute__((constructor)) static void spu__xr_fix_init(void)
{
    unsigned a, b, c, d;
    if (__get_cpuid(1, &a, &b, &c, &d) && (c & bit_OSXSAVE) && (c & bit_AVX)) {
        unsigned lo, hi;
        __asm__ volatile("xgetbv" : "=a"(lo), "=d"(hi) : "c"(0));
        if ((lo & 6u) == 6u) spu__xr_ymm_ok = 1;
    }
    __builtin_cpu_init();
    spu__xr_use_v3 = spu__xr_ymm_ok && __builtin_cpu_supports("avx2") && __builtin_cpu_supports("fma")
                  && __builtin_cpu_supports("bmi") && __builtin_cpu_supports("bmi2")
                  && __builtin_cpu_supports("lzcnt");
}

/* area: the caller's block (see the contract above). */
static inline __attribute__((always_inline)) void spu__xr_fix_body(unsigned char* area, uint32_t op)
{
    u128 r, ok, a, b, c;
    memcpy(&r,  area +  0, 16);
    memcpy(&ok, area + 16, 16);
    memcpy(&a,  area + 32, 16);
    memcpy(&b,  area + 48, 16);
    memcpy(&c,  area + 64, 16);
    const uint32_t n1 = (op & SPU__XR_OP_NEG1) ? 0x80000000u : 0;   /* negb (fa) / negc (fma) */
    const uint32_t n2 = (op & SPU__XR_OP_NEG2) ? 0x80000000u : 0;   /* negp (fma) */
    for (int i = 0; i < 4; i++) {
        if (ok._u32[i] >> 31) continue;
        switch (op & SPU__XR_OP_KIND) {
        case SPU__XR_OP_FA:  r._u32[i] = spu__fa_any(a._u32[i], b._u32[i] ^ n1); break;
        case SPU__XR_OP_FM:  r._u32[i] = spu__fm_any(a._u32[i], b._u32[i]); break;
        default:             r._u32[i] = spu__fma_any(a._u32[i], b._u32[i], c._u32[i], n2, n1); break;
        }
    }
    memcpy(area + 0, &r, 16);
}
static __attribute__((noinline)) void spu__xr_fix_base(unsigned char* area, uint32_t op) { spu__xr_fix_body(area, op); }
/* The same with the ISA the game's lifted code is built for (fma inline instead of a libm call,
 * lzcnt): the path is hot in some code, see above. */
static __attribute__((noinline, target("avx2,fma,bmi,bmi2,lzcnt")))
void spu__xr_fix_v3(unsigned char* area, uint32_t op) { spu__xr_fix_body(area, op); }
/* SPU_XR_FIX_STATS=<seconds> (diagnostic, off by default): count the lanes that take this path
 * by call site (the lifted code's return address), op and reason, and print the top entries to
 * stderr every <seconds> s (run.sh ends the process with SIGKILL, so no exit-time report). */
static int spu__xr_stats_secs;
static pthread_mutex_t spu__xr_stats_mu = PTHREAD_MUTEX_INITIALIZER;
typedef struct { uintptr_t ret; uint32_t op, why; uint64_t lanes; uint32_t ex[4]; } spu__xr_stat;
#define SPU__XR_NSTAT 4096
static spu__xr_stat spu__xr_stats[SPU__XR_NSTAT];
static uint64_t spu__xr_stat_calls, spu__xr_stat_lanes;
static time_t spu__xr_stat_last;

__attribute__((constructor)) static void spu__xr_stats_init(void)
{
    const char* e = getenv("SPU_XR_FIX_STATS");
    if (e && *e) { spu__xr_stats_secs = atoi(e); if (spu__xr_stats_secs <= 0) spu__xr_stats_secs = 10; }
}
static int spu__xr_stat_cmp(const void* x, const void* y)
{
    const spu__xr_stat *a = x, *b = y;
    return a->lanes < b->lanes ? 1 : a->lanes > b->lanes ? -1 : 0;
}
/* reason bits: operand biased exponent 255 (a 1, b 2, c 4) or 0 (a 8, b 16, c 32); exact result
 * class in bits 6-7: 0 normal below FLT_MAX, 1 = +-FLT_MAX, 2 = exponent 255, 3 = zero. */
static void spu__xr_stats_note(const unsigned char* area, uint32_t op, uintptr_t ret)
{
    u128 r, ok, a, b, c;
    memcpy(&r, area, 16); memcpy(&ok, area + 16, 16); memcpy(&a, area + 32, 16);
    memcpy(&b, area + 48, 16); memcpy(&c, area + 64, 16);
    pthread_mutex_lock(&spu__xr_stats_mu);
    spu__xr_stat_calls++;
    for (int i = 0; i < 4; i++) {
        if (ok._u32[i] >> 31) continue;
        const uint32_t ea = a._u32[i] >> 23 & 0xFF, eb = b._u32[i] >> 23 & 0xFF, ec = c._u32[i] >> 23 & 0xFF;
        const int fma = (op & SPU__XR_OP_KIND) == SPU__XR_OP_FMA;
        uint32_t why = (ea == 255) | (eb == 255) << 1 | (fma && ec == 255) << 2
                     | (ea == 0) << 3 | (eb == 0) << 4 | (fma && ec == 0) << 5;
        const uint32_t rm = r._u32[i] & 0x7FFFFFFFu;
        why |= (rm == 0 ? 3u : rm == 0x7F7FFFFFu ? 1u : (rm >> 23) == 255 ? 2u : 0u) << 6;
        spu__xr_stat_lanes++;
        uint32_t h = (uint32_t)((ret * 0x9E3779B97F4A7C15ull) >> 40) ^ op * 31 ^ why * 977;
        for (int k = 0; k < SPU__XR_NSTAT; k++, h++) {
            spu__xr_stat* s = &spu__xr_stats[h % SPU__XR_NSTAT];
            if (!s->lanes) { s->ret = ret; s->op = op; s->why = why;
                             s->ex[0] = a._u32[i]; s->ex[1] = b._u32[i]; s->ex[2] = c._u32[i]; s->ex[3] = r._u32[i]; }
            if (s->ret == ret && s->op == op && s->why == why) { s->lanes++; break; }
        }
    }
    const time_t now = time(NULL);
    if (!spu__xr_stat_last) spu__xr_stat_last = now;
    if (now - spu__xr_stat_last >= spu__xr_stats_secs) {
        spu__xr_stat_last = now;
        static spu__xr_stat tmp[SPU__XR_NSTAT];
        int n = 0;
        for (int k = 0; k < SPU__XR_NSTAT; k++) if (spu__xr_stats[k].lanes) tmp[n++] = spu__xr_stats[k];
        qsort(tmp, (size_t)n, sizeof tmp[0], spu__xr_stat_cmp);
        Dl_info di; uintptr_t base = 0;
        if (dladdr((void*)spu__xr_stats_note, &di) && di.dli_fbase) base = (uintptr_t)di.dli_fbase;
        fprintf(stderr, "[xr-fix] t=%ld calls=%llu lanes=%llu sites=%d\n", (long)now,
                (unsigned long long)spu__xr_stat_calls, (unsigned long long)spu__xr_stat_lanes, n);
        static const char* kind[] = { "fa", "fm", "fma", "?" };
        for (int k = 0; k < n && k < 40; k++)
            fprintf(stderr, "[xr-fix]   %10llu  ret=+0x%lx %s%s%s why=0x%02x  ex a=%08x b=%08x c=%08x r=%08x\n",
                    (unsigned long long)tmp[k].lanes, (unsigned long)(tmp[k].ret - base),
                    kind[tmp[k].op & 3], tmp[k].op & SPU__XR_OP_NEG1 ? " n1" : "", tmp[k].op & SPU__XR_OP_NEG2 ? " n2" : "",
                    tmp[k].why, tmp[k].ex[0], tmp[k].ex[1], tmp[k].ex[2], tmp[k].ex[3]);
        fflush(stderr);
    }
    pthread_mutex_unlock(&spu__xr_stats_mu);
}

__attribute__((used, visibility("hidden"))) void spu__xr_fix_c(unsigned char* area, uint32_t op, uintptr_t ret)
{
    if (spu__xr_use_v3) spu__xr_fix_v3(area, op); else spu__xr_fix_base(area, op);
    if (__builtin_expect(spu__xr_stats_secs != 0, 0)) spu__xr_stats_note(area, op, ret);
}

__asm__(
    "\t.text\n"
    "\t.globl spu__xr_fix_stub\n"
    "\t.hidden spu__xr_fix_stub\n"
    "\t.type spu__xr_fix_stub, @function\n"
    "\t.p2align 4\n"
    "spu__xr_fix_stub:\n"
    "\t.cfi_startproc\n"
#if defined(__CET__)
    "\tendbr64\n"
#endif
    "\tpushq %rbp\n"
    "\t.cfi_def_cfa_offset 16\n"
    "\t.cfi_offset %rbp, -16\n"
    "\tmovq %rsp, %rbp\n"
    "\t.cfi_def_cfa_register %rbp\n"
    "\tpushq %rax\n\tpushq %rcx\n\tpushq %rdx\n\tpushq %rsi\n\tpushq %rdi\n"
    "\tpushq %r8\n\tpushq %r9\n\tpushq %r10\n\tpushq %r11\n"
    "\tsubq $512, %rsp\n"
    "\tandq $-32, %rsp\n"
    "\tcmpb $0, spu__xr_ymm_ok(%rip)\n"
    "\tje 1f\n"
    "\tvmovdqu %ymm0, 0(%rsp)\n"
    "\tvmovdqu %ymm1, 32(%rsp)\n"
    "\tvmovdqu %ymm2, 64(%rsp)\n"
    "\tvmovdqu %ymm3, 96(%rsp)\n"
    "\tvmovdqu %ymm4, 128(%rsp)\n"
    "\tvmovdqu %ymm5, 160(%rsp)\n"
    "\tvmovdqu %ymm6, 192(%rsp)\n"
    "\tvmovdqu %ymm7, 224(%rsp)\n"
    "\tvmovdqu %ymm8, 256(%rsp)\n"
    "\tvmovdqu %ymm9, 288(%rsp)\n"
    "\tvmovdqu %ymm10, 320(%rsp)\n"
    "\tvmovdqu %ymm11, 352(%rsp)\n"
    "\tvmovdqu %ymm12, 384(%rsp)\n"
    "\tvmovdqu %ymm13, 416(%rsp)\n"
    "\tvmovdqu %ymm14, 448(%rsp)\n"
    "\tvmovdqu %ymm15, 480(%rsp)\n"
    /* clean upper state for the legacy-SSE C code (a dirty one stalled it badly) */
    "\tvzeroupper\n"
    "\tjmp 2f\n"
    "1:\n"
    "\tmovdqu %xmm0, 0(%rsp)\n"
    "\tmovdqu %xmm1, 32(%rsp)\n"
    "\tmovdqu %xmm2, 64(%rsp)\n"
    "\tmovdqu %xmm3, 96(%rsp)\n"
    "\tmovdqu %xmm4, 128(%rsp)\n"
    "\tmovdqu %xmm5, 160(%rsp)\n"
    "\tmovdqu %xmm6, 192(%rsp)\n"
    "\tmovdqu %xmm7, 224(%rsp)\n"
    "\tmovdqu %xmm8, 256(%rsp)\n"
    "\tmovdqu %xmm9, 288(%rsp)\n"
    "\tmovdqu %xmm10, 320(%rsp)\n"
    "\tmovdqu %xmm11, 352(%rsp)\n"
    "\tmovdqu %xmm12, 384(%rsp)\n"
    "\tmovdqu %xmm13, 416(%rsp)\n"
    "\tmovdqu %xmm14, 448(%rsp)\n"
    "\tmovdqu %xmm15, 480(%rsp)\n"
    "2:\n"
    "\tleaq 16(%rbp), %rdi\n"              /* the block */
    "\tmovl 80(%rdi), %esi\n"              /* op */
    "\tmovq 8(%rbp), %rdx\n"               /* the return address in the lifted code (stats only) */
    "\tcall spu__xr_fix_c\n"
    "\tcmpb $0, spu__xr_ymm_ok(%rip)\n"
    "\tje 3f\n"
    "\tvmovdqu 0(%rsp), %ymm0\n"
    "\tvmovdqu 32(%rsp), %ymm1\n"
    "\tvmovdqu 64(%rsp), %ymm2\n"
    "\tvmovdqu 96(%rsp), %ymm3\n"
    "\tvmovdqu 128(%rsp), %ymm4\n"
    "\tvmovdqu 160(%rsp), %ymm5\n"
    "\tvmovdqu 192(%rsp), %ymm6\n"
    "\tvmovdqu 224(%rsp), %ymm7\n"
    "\tvmovdqu 256(%rsp), %ymm8\n"
    "\tvmovdqu 288(%rsp), %ymm9\n"
    "\tvmovdqu 320(%rsp), %ymm10\n"
    "\tvmovdqu 352(%rsp), %ymm11\n"
    "\tvmovdqu 384(%rsp), %ymm12\n"
    "\tvmovdqu 416(%rsp), %ymm13\n"
    "\tvmovdqu 448(%rsp), %ymm14\n"
    "\tvmovdqu 480(%rsp), %ymm15\n"
    "\tjmp 4f\n"
    "3:\n"
    "\tmovdqu 0(%rsp), %xmm0\n"
    "\tmovdqu 32(%rsp), %xmm1\n"
    "\tmovdqu 64(%rsp), %xmm2\n"
    "\tmovdqu 96(%rsp), %xmm3\n"
    "\tmovdqu 128(%rsp), %xmm4\n"
    "\tmovdqu 160(%rsp), %xmm5\n"
    "\tmovdqu 192(%rsp), %xmm6\n"
    "\tmovdqu 224(%rsp), %xmm7\n"
    "\tmovdqu 256(%rsp), %xmm8\n"
    "\tmovdqu 288(%rsp), %xmm9\n"
    "\tmovdqu 320(%rsp), %xmm10\n"
    "\tmovdqu 352(%rsp), %xmm11\n"
    "\tmovdqu 384(%rsp), %xmm12\n"
    "\tmovdqu 416(%rsp), %xmm13\n"
    "\tmovdqu 448(%rsp), %xmm14\n"
    "\tmovdqu 480(%rsp), %xmm15\n"
    "4:\n"
    "\tleaq -72(%rbp), %rsp\n"
    "\tpopq %r11\n\tpopq %r10\n\tpopq %r9\n\tpopq %r8\n"
    "\tpopq %rdi\n\tpopq %rsi\n\tpopq %rdx\n\tpopq %rcx\n\tpopq %rax\n"
    "\tpopq %rbp\n"
    "\t.cfi_def_cfa %rsp, 8\n"
    "\tret\n"
    "\t.cfi_endproc\n"
    "\t.size spu__xr_fix_stub, .-spu__xr_fix_stub\n"
);
#endif
