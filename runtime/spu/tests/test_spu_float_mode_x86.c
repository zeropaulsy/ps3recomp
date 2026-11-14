/* SPU_FLOAT_RZ on x86: spu_run_with_halt may run SPU code with MXCSR at round toward zero with
 * flush-to-zero and denormals-are-zero (SPU_MXCSR_RZFZ), where fa/fs/fm/fma/fms/fnms take their
 * native form. Every SPU floating-point helper must give the same bits in that mode as with the
 * default MXCSR: this runs each one on the same operands in both modes and compares, lane by lane.
 * The default-mode results are what test_spu_float_exact and the per-op tests check against the
 * exact references, so agreement here carries that over to the RZ mode.
 * Operands: SPU edge words, random normals in a narrow and a wide window, results near 2^-126
 * and 2^128, fma addends close to -a*b, raw random words; for the double ops also IEEE edge
 * doubles (denormals, infinities, NaNs) and raw random doublewords.
 * Build: cc -O2 -msse4.1 [-mavx2 [-mfma]] -ffp-contract=off -I runtime/spu -I include THIS -lm
 *        -o t && ./t [millions] */
#include "spu_helpers.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <xmmintrin.h>

#ifndef SPU_MXCSR_RZFZ
#define SPU_MXCSR_RZFZ 0xFFC0u
#endif

static uint64_t s_rng = 0x9E3779B97F4A7C15ull;
static inline uint32_t rnd32(void)
{
    s_rng ^= s_rng << 13; s_rng ^= s_rng >> 7; s_rng ^= s_rng << 17;
    return (uint32_t)(s_rng >> 16);
}
static const uint32_t k_edge[] = {
    0x00000000u, 0x80000000u, 0x00000001u, 0x807FFFFFu, 0x00400000u, 0x00800000u, 0x80800000u,
    0x00800001u, 0x3F800000u, 0xBF800000u, 0x3F7FFFFFu, 0x3F800001u, 0x7F7FFFFFu, 0xFF7FFFFFu,
    0x7F000000u, 0x7F800000u, 0xFF800000u, 0x7FC00000u, 0x7F800001u, 0x7FFFFFFFu, 0xFFFFFFFFu,
    0x01000000u, 0x00FFFFFFu, 0x4B000000u, 0x4B7FFFFFu, 0x33800000u, 0x34000000u, 0x5F800000u,
    0x1F800000u, 0x20000000u, 0x60000000u, 0x4F000000u, 0xCF000000u, 0x4F800000u,
};
#define NEDGE (sizeof k_edge / sizeof k_edge[0])
static const uint64_t k_dedge[] = {
    0, 0x8000000000000000ull, 1, 0x000FFFFFFFFFFFFFull, 0x0010000000000000ull, 0x8010000000000000ull,
    0x3FF0000000000000ull, 0xBFF0000000000000ull, 0x7FEFFFFFFFFFFFFFull, 0x7FF0000000000000ull,
    0xFFF0000000000000ull, 0x7FF8000000000000ull, 0x7FF0000000000001ull, 0x3FF0000000000001ull,
    0x3CA0000000000000ull, 0x0000000080000000ull, 0x47EFFFFFE0000000ull, 0x47EFFFFFF0000000ull,
    0x3810000000000000ull, 0x380FFFFFFFFFFFFFull, 0x36A0000000000000ull,
};
#define NDEDGE (sizeof k_dedge / sizeof k_dedge[0])
static uint32_t word(uint32_t z)
{
    const uint32_t sign = rnd32() & 0x80000000u;
    switch (rnd32() % 9) {
    case 0: return k_edge[rnd32() % NEDGE] ^ (rnd32() & 0x80000000u);
    case 1: return sign | ((100u + rnd32() % 56u) << 23) | (rnd32() & 0x7FFFFFu);
    case 2: return sign | ((1u + rnd32() % 254u) << 23) | (rnd32() & 0x7FFFFFu);
    case 3: return sign | ((rnd32() % 2u ? 255u : 0u) << 23) | (rnd32() & 0x7FFFFFu);
    case 4: return sign | ((1u + rnd32() % 4u) << 23) | (rnd32() & 0x7FFFFFu);
    case 5: return sign | ((250u + rnd32() % 6u) << 23) | (rnd32() & 0x7FFFFFu);
    case 6: return z ? (z ^ 0x80000000u) + (rnd32() % 5u) - 2u : rnd32();
    case 7: return sign | ((63u + rnd32() % 2u) << 23) | (rnd32() & 0x7FFFFFu);
    default: return rnd32();
    }
}
static void dword(u128* v, int i)
{
    uint64_t d;
    switch (rnd32() % 4) {
    case 0: d = k_dedge[rnd32() % NDEDGE] ^ ((uint64_t)(rnd32() & 1) << 63); break;
    case 1: d = ((uint64_t)(rnd32() & 0x80000FFFu) << 32 | rnd32()) ^ ((uint64_t)(1023 - 30 + rnd32() % 60) << 52); break;
    case 2: d = ((uint64_t)(rnd32() % 3) << 52) | ((uint64_t)rnd32() << 20) | (rnd32() & 0xFFFFF); break; /* tiny */
    default: d = (uint64_t)rnd32() << 32 | rnd32(); break;
    }
    v->_u32[2 * i] = (uint32_t)(d >> 32); v->_u32[2 * i + 1] = (uint32_t)d;
}

enum { OP_FA, OP_FS, OP_FM, OP_FMA, OP_FMS, OP_FNMS, OP_FCEQ, OP_FCGT, OP_FCMEQ, OP_FCMGT, OP_FI,
       OP_FREST, OP_FRSQEST, OP_FESD, OP_CFLTS, OP_CFLTU, OP_CSFLT, OP_CUFLT,
       OP_DFA, OP_DFS, OP_DFM, OP_DFMA, OP_DFMS, OP_DFNMS, OP_DFNMA, OP_DFCEQ, OP_DFCMEQ, OP_DFCGT,
       OP_DFCMGT, OP_FRDS, OP_N };
static const char* const k_name[OP_N] = { "fa", "fs", "fm", "fma", "fms", "fnms", "fceq", "fcgt",
    "fcmeq", "fcmgt", "fi", "frest", "frsqest", "fesd", "cflts", "cfltu", "csflt", "cuflt",
    "dfa", "dfs", "dfm", "dfma", "dfms", "dfnms", "dfnma", "dfceq", "dfcmeq", "dfcgt", "dfcmgt", "frds" };

#define BATCH 4096
typedef struct { u128 a, b, c; int i8; } operands;
static operands s_in[BATCH];
static u128 s_out[2][BATCH];

/* noinline, and called only after the MXCSR write: the compiler cannot move the arithmetic
 * across the mode change. */
__attribute__((noinline)) static void run(int op, u128* out, int n)
{
    for (int k = 0; k < n; k++) {
        const u128 a = s_in[k].a, b = s_in[k].b, c = s_in[k].c; const int i8 = s_in[k].i8;
        u128 r;
        switch (op) {
        case OP_FA: r = spu_fa(a, b); break;          case OP_FS: r = spu_fs(a, b); break;
        case OP_FM: r = spu_fm(a, b); break;          case OP_FMA: r = spu_fma(a, b, c); break;
        case OP_FMS: r = spu_fms(a, b, c); break;     case OP_FNMS: r = spu_fnms(a, b, c); break;
        case OP_FCEQ: r = spu_fceq(a, b); break;      case OP_FCGT: r = spu_fcgt(a, b); break;
        case OP_FCMEQ: r = spu_fcmeq(a, b); break;    case OP_FCMGT: r = spu_fcmgt(a, b); break;
        case OP_FI: r = spu_fi(a, b); break;          case OP_FREST: r = spu_frest(a); break;
        case OP_FRSQEST: r = spu_frsqest(a); break;   case OP_FESD: r = spu_fesd(a); break;
        case OP_CFLTS: r = spu_cflts(a, i8); break;   case OP_CFLTU: r = spu_cfltu(a, i8); break;
        case OP_CSFLT: r = spu_csflt(a, i8); break;   case OP_CUFLT: r = spu_cuflt(a, i8); break;
        case OP_DFA: r = spu_dfa(a, b); break;        case OP_DFS: r = spu_dfs(a, b); break;
        case OP_DFM: r = spu_dfm(a, b); break;        case OP_DFMA: r = spu_dfma(a, b, c); break;
        case OP_DFMS: r = spu_dfms(a, b, c); break;   case OP_DFNMS: r = spu_dfnms(a, b, c); break;
        case OP_DFNMA: r = spu_dfnma(a, b, c); break; case OP_DFCEQ: r = spu_dfceq(a, b); break;
        case OP_DFCMEQ: r = spu_dfcmeq(a, b); break;  case OP_DFCGT: r = spu_dfcgt(a, b); break;
        case OP_DFCMGT: r = spu_dfcmgt(a, b); break;  default: r = spu_frds(a); break;
        }
        out[k] = r;
    }
}

static long s_bad[OP_N], s_lanes[OP_N], s_mode_bad;
static void batch(int op, int n)
{
    const unsigned outer = _mm_getcsr();
    _mm_setcsr(0x1F80u); spu_float_mode_note(0);
    run(op, s_out[0], n);
    _mm_setcsr(SPU_MXCSR_RZFZ); spu_float_mode_note(1);
    run(op, s_out[1], n);
    if ((_mm_getcsr() & ~0x3Fu) != (SPU_MXCSR_RZFZ & ~0x3Fu)) s_mode_bad++;   /* helpers restore it */
    _mm_setcsr(outer); spu_float_mode_note(0);
    for (int k = 0; k < n; k++) {
        s_lanes[op] += 4;
        for (int i = 0; i < 4; i++)
            if (s_out[0][k]._u32[i] != s_out[1][k]._u32[i] && s_bad[op]++ < 6)
                printf("MISMATCH %-7s lane %d a=%08X b=%08X c=%08X i8=%d default=%08X rz=%08X\n",
                       k_name[op], i, s_in[k].a._u32[i], s_in[k].b._u32[i], s_in[k].c._u32[i],
                       s_in[k].i8, s_out[0][k]._u32[i], s_out[1][k]._u32[i]);
    }
}

int main(int argc, char** argv)
{
    const long millions = argc > 1 ? atol(argv[1]) : 4;
    /* edge words against edge words, every lane position */
    int n = 0;
    for (unsigned i = 0; i < NEDGE; i++) for (unsigned j = 0; j < NEDGE; j++) for (unsigned k = 0; k < NEDGE; k += 3) {
        operands* o = &s_in[n];
        for (int l = 0; l < 4; l++) {
            o->a._u32[l] = k_edge[(i + l) % NEDGE]; o->b._u32[l] = k_edge[(j + 3 * l) % NEDGE];
            o->c._u32[l] = k_edge[(k + 5 * l) % NEDGE];
        }
        o->i8 = (int)(rnd32() & 0xFF);
        if (++n == BATCH) { for (int op = 0; op < OP_DFA; op++) batch(op, n); n = 0; }
    }
    if (n) for (int op = 0; op < OP_DFA; op++) batch(op, n);
    const long total = millions * 1000000L;
    for (long done = 0; done < total; done += BATCH) {
        for (int k = 0; k < BATCH; k++) {
            operands* o = &s_in[k];
            for (int l = 0; l < 4; l++) {
                o->a._u32[l] = word(0); o->b._u32[l] = word(0);
                const u128 one = { ._u32 = { o->a._u32[l], 0, 0, 0 } }, two = { ._u32 = { o->b._u32[l], 0, 0, 0 } };
                o->c._u32[l] = word(spu__fm_s(one, two)._u32[0]);
            }
            o->i8 = (int)(rnd32() & 0xFF);
        }
        for (int op = 0; op < OP_DFA; op++) batch(op, BATCH);
        for (int k = 0; k < BATCH; k++)
            for (int i = 0; i < 2; i++) { dword(&s_in[k].a, i); dword(&s_in[k].b, i); dword(&s_in[k].c, i); }
        for (int op = OP_DFA; op < OP_N; op++) batch(op, BATCH);
    }
    long bad = s_mode_bad;
    for (int op = 0; op < OP_N; op++) {
        if (s_bad[op]) printf("  %-7s %ld of %ld lanes differ\n", k_name[op], s_bad[op], s_lanes[op]);
        bad += s_bad[op];
    }
    if (s_mode_bad) printf("  MXCSR not restored after %ld batches\n", s_mode_bad);
    printf("test_spu_float_mode_x86: %s (%s), %ld differences\n", bad ? "FAIL" : "ok",
#if defined(__AVX2__) && defined(__FMA__)
           "AVX2+FMA",
#elif defined(__AVX2__)
           "AVX2",
#else
           "SSE4.1",
#endif
           bad);
    return bad != 0;
}
