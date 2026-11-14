/* The x86 vector paths in runtime/ppu/ppu_vmx.h against the scalar loops they replace (name_s).
 *
 * Every helper with a vector path must give the same 16 bytes (and the same CR6 nibble for compares)
 * as its scalar loop, NaN bit patterns included, with VSCR[NJ] clear and set and with the destination
 * aliasing a source. The operands mix random words, a pool of specials (signed zeros, infinities,
 * quiet / signalling NaNs with payloads, denormals, the float limits), denormals, mid-range values and,
 * for the multiply-adds, addends that cancel the product exactly or nearly (the signed-zero edge that
 * fnmadd would get wrong). vmaddfp / vnmsubfp are also checked against libm fmaf called through a
 * pointer, so the reference does not depend on how the header compiled.
 *
 * Built twice: SSE4.1 only (the FMA ops take the scalar path) and with -mavx2 -mfma (all vector paths;
 * that binary skips itself on a host without FMA).
 *
 *   test_vmx_simd [count]     default 4 million vector cases per helper
 */
#include <stdio.h>
#include <stdint.h>
#include <string.h>
#include <math.h>
#include <stdlib.h>
#include "../runtime/ppu/ppu_vmx.h"

static uint64_t st = 0x9E3779B97F4A7C15ull;
static uint32_t rnd(void) { st ^= st << 13; st ^= st >> 7; st ^= st << 17; return (uint32_t)st; }
static float f(uint32_t u) { float x; memcpy(&x, &u, 4); return x; }
static uint32_t U(float x) { uint32_t u; memcpy(&u, &x, 4); return u; }

static const uint32_t specials[] = {
    0x00000000, 0x80000000, 0x7F800000, 0xFF800000, 0x7FC00000, 0xFFC00000, 0x7FC12345, 0xFFF00001,
    0x7F800001, 0xFF812345, 0x7FBFFFFF, 0x00000001, 0x80000001, 0x007FFFFF, 0x807FFFFF, 0x00800000,
    0x80800000, 0x7F7FFFFF, 0xFF7FFFFF, 0x3F800000, 0xBF800000, 0x40000000, 0x3F000000, 0x00400000,
};
#define NSPEC (sizeof specials / sizeof specials[0])

static uint32_t word(void) {
    const uint32_t k = rnd() % 16, w = rnd();
    if (k < 4) return w;                                                   /* anything */
    if (k < 7) return specials[rnd() % NSPEC];
    if (k < 9) return w & 0x807FFFFF;                                      /* denormal / zero */
    if (k < 11) return (w & 0x8000FFFF) | ((0x7D + rnd() % 6) << 23);      /* few bits, near 1: exact products */
    return (w & 0x80FFFFFF) | ((0x60 + rnd() % 0x40) << 23);               /* mid range */
}
static void vec(uint8_t* v) { for (int i = 0; i < 4; i++) vmx_sw(v, i, word()); }

static unsigned long long checks, bad;
static void report(const char* name, const uint8_t* a, const uint8_t* b, const uint8_t* c, uint32_t vscr,
                   const uint8_t* r1, const uint8_t* r2, uint32_t cr1, uint32_t cr2) {
    checks++;
    if (!memcmp(r1, r2, 16) && cr1 == cr2) return;
    if (bad++ < 12) {
        printf("%s vscr %08x:", name, vscr);
        for (int i = 0; i < 4; i++) printf(" [%08x %08x %08x -> %08x vs %08x]", vmx_w(a, i), vmx_w(b, i),
                                           c ? vmx_w(c, i) : 0, vmx_w(r1, i), vmx_w(r2, i));
        if (cr1 != cr2) printf(" cr6 %x vs %x", cr1, cr2);
        printf("\n");
    }
}

/* run vector helper V and scalar S on (a,b[,c]) both into a fresh buffer and into a buffer aliasing a */
#define T2(V, S) do { uint8_t r1[16], r2[16], x1[16], x2[16]; V(r1, a, b, vscr); S(r2, a, b, vscr); \
    report(#V, a, b, NULL, vscr, r1, r2, 0, 0); memcpy(x1, a, 16); memcpy(x2, a, 16); \
    V(x1, x1, b, vscr); S(x2, x2, b, vscr); report(#V " alias", a, b, NULL, vscr, x1, x2, 0, 0); } while (0)
#define T3(V, S) do { uint8_t r1[16], r2[16], x1[16], x2[16]; V(r1, a, b, c, vscr); S(r2, a, b, c, vscr); \
    report(#V, a, b, c, vscr, r1, r2, 0, 0); memcpy(x1, c, 16); memcpy(x2, c, 16); \
    V(x1, a, b, x1, vscr); S(x2, a, b, x2, vscr); report(#V " alias", a, b, c, vscr, x1, x2, 0, 0); } while (0)
#define T1(V, S) do { uint8_t r1[16], r2[16]; V(r1, b, vscr); S(r2, b, vscr); \
    report(#V, a, b, NULL, vscr, r1, r2, 0, 0); } while (0)
#define TC(V, S) do { uint8_t r1[16], r2[16]; uint32_t c1 = V(r1, a, b), c2 = S(r2, a, b); \
    report(#V, a, b, NULL, vscr, r1, r2, c1, c2); } while (0)

int main(int argc, char** argv)
{
#if VMX_X86 && defined(__FMA__)
    __builtin_cpu_init();
    if (!__builtin_cpu_supports("fma") || !__builtin_cpu_supports("avx2")) {
        printf("host has no AVX2/FMA: skipped\n");
        return 0;
    }
    const char* mode = "SSE4.1 + AVX2/FMA";
#elif VMX_X86
    const char* mode = "SSE4.1 (FMA ops scalar)";
#else
    const char* mode = "scalar only";
#endif
    float (*volatile pfmaf)(float, float, float) = fmaf;
    const long count = argc > 1 ? atol(argv[1]) : 4000000L;
    unsigned long long ref_bad = 0;
    for (long n = 0; n < count; n++) {
        uint8_t a[16], b[16], c[16];
        vec(a); vec(b); vec(c);
        const uint32_t vscr = (n & 7) == 7 ? VMX_NJ : 0;
        /* multiply-add addend (b) cancelling the product a*c exactly or within a few ulps */
        const int k = (int)(rnd() % 4);
        if (k) for (int i = 0; i < 4; i++) {
            const uint32_t p = U(f(vmx_w(a, i)) * f(vmx_w(c, i)));
            vmx_sw(b, i, k == 1 ? p : k == 2 ? p ^ 0x80000000u : (p ^ 0x80000000u) + (rnd() % 5) - 2);
        }
        T3(vmx_vmaddfp, vmx_vmaddfp_s);
        T3(vmx_vnmsubfp, vmx_vnmsubfp_s);
        if (!vscr) {   /* the same against libm fmaf through a pointer (no flushes with NJ clear) */
            uint8_t r1[16], r2[16];
            vmx_vmaddfp(r1, a, b, c, 0); vmx_vnmsubfp(r2, a, b, c, 0);
            for (int i = 0; i < 4; i++) {
                const float x = f(vmx_w(a, i)), y = f(vmx_w(b, i)), z = f(vmx_w(c, i));
                const uint32_t e1 = vmx_fix(U(pfmaf(x, z, y))), e2 = vmx_fix(U(pfmaf(x, z, -y)) ^ 0x80000000u);
                if (vmx_w(r1, i) != e1 || vmx_w(r2, i) != e2) {
                    if (ref_bad++ < 8)
                        printf("fmaf ref %08x %08x %08x: madd %08x vs %08x, nmsub %08x vs %08x\n",
                               vmx_w(a, i), vmx_w(b, i), vmx_w(c, i), vmx_w(r1, i), e1, vmx_w(r2, i), e2);
                }
            }
        }
        T2(vmx_vaddfp, vmx_vaddfp_s);
        T2(vmx_vsubfp, vmx_vsubfp_s);
        T2(vmx_vmaxfp, vmx_vmaxfp_s);
        T2(vmx_vminfp, vmx_vminfp_s);
        T1(vmx_vrefp, vmx_vrefp_s);
        T1(vmx_vrsqrtefp, vmx_vrsqrtefp_s);
        if (n & 1) memcpy(b, a, 16);                       /* all-true compares too */
        else if (n & 2) memcpy(b + 4 * (rnd() % 4), a, 4);
        TC(vmx_vcmpequw, vmx_vcmpequw_s);
        TC(vmx_vcmpgtuw, vmx_vcmpgtuw_s);
        TC(vmx_vcmpgtsw, vmx_vcmpgtsw_s);
        TC(vmx_vcmpeqfp, vmx_vcmpeqfp_s);
        TC(vmx_vcmpgefp, vmx_vcmpgefp_s);
        TC(vmx_vcmpgtfp, vmx_vcmpgtfp_s);
    }
    printf("%s: %llu checks, %llu differ; fmaf reference: %llu lanes differ\n", mode, checks, bad, ref_bad);
    return (bad || ref_bad) ? 1 : 0;
}
