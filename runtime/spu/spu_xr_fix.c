/*
 * The out-of-line exact path of the SPU_FLOAT_RZ x86 float helpers (spu_helpers.h,
 * spu__xr_fixcall), reached through an assembly stub that keeps every register.
 *
 * Why a stub: the fast path of fa/fm/fma is a handful of SSE ops plus a range check, and the
 * lanes that fail it (an extended-range operand, an overflow) are rare. With a plain C call on the
 * failing branch, every xmm register is call-clobbered on x86-64, so the compiler kept every live
 * vector in a stack slot around every float op of a lifted function, though the call almost never
 * runs (LBP2's hottest job function: ~500 stack moves for ~95 SPU instructions). The inline asm in
 * spu__xr_fixcall changes only xmm0 and the flags, so the fast path keeps its registers; the stub
 * pays for that by saving every vector register (the full ymm registers when the OS has enabled
 * AVX, so their upper halves survive whatever the C code here or libm uses; else the xmm
 * registers) and every caller-saved GPR. Some code takes this path often (FMOD's DSP plugins in
 * LBP2: ~5% of the process with XSAVE/XRSTOR here), so it is plain moves, not XSAVE. MXCSR and
 * the x87 state are not saved: nothing below changes them.
 *
 * Contract (see spu__xr_fixcall): xmm0 = the fast result, xmm1 = the per-lane ok mask, xmm2 = a,
 * xmm3 = b, xmm4 = c (fma only), eax = op; called with rsp 128 below the caller's red zone. Returns
 * the fixed result in xmm0; every other register as it was.
 */
#include "spu_helpers.h"
#include <string.h>

#if SPU__XR_FIXSTUB
#include <cpuid.h>

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

/* area: the stub's register image; XMM i is at 32*i (the low half of a 32-byte slot). */
static inline __attribute__((always_inline)) void spu__xr_fix_body(unsigned char* area, uint32_t op)
{
    u128 r, ok, a, b, c;
    memcpy(&r,  area +   0, 16);
    memcpy(&ok, area +  32, 16);
    memcpy(&a,  area +  64, 16);
    memcpy(&b,  area +  96, 16);
    memcpy(&c,  area + 128, 16);
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
__attribute__((used, visibility("hidden"))) void spu__xr_fix_c(unsigned char* area, uint32_t op)
{
    if (spu__xr_use_v3) spu__xr_fix_v3(area, op); else spu__xr_fix_base(area, op);
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
    "\tmovl %eax, %esi\n"
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
    "\tmovq %rsp, %rdi\n"
    "\tcall spu__xr_fix_c\n"
    "\tcmpb $0, spu__xr_ymm_ok(%rip)\n"
    "\tje 3f\n"
    "\tvmovdqu 0(%rsp), %xmm0\n"          /* the result: 16 bytes, as written (store forwarding); the asm owns ymm0 */
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
