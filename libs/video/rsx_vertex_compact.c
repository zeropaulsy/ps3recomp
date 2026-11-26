#include "rsx_vertex_compact.h"
#include "rsx_commands.h"
#include "rsx_vertex_formats.h"

#include <limits.h>
#include <stdlib.h>
#include <string.h>

static float rsx_compact_be_f32(const u8* p)
{
    const u32 w = ((u32)p[0] << 24) | ((u32)p[1] << 16) |
                  ((u32)p[2] << 8) | (u32)p[3];
    float f;
    memcpy(&f, &w, sizeof(f));
    return f;
}

static float rsx_compact_be_f16(const u8* p)
{
    const u16 h = (u16)((p[0] << 8) | p[1]);
    const u32 sign = (u32)(h >> 15) << 31;
    const u32 exp = (h >> 10) & 0x1F;
    const u32 man = h & 0x3FF;
    u32 out;
    if (exp == 0)
        out = sign;
    else if (exp == 31)
        out = sign | 0x7F800000u | (man << 13);
    else
        out = sign | ((exp + 112) << 23) | (man << 13);
    float f;
    memcpy(&f, &out, sizeof(f));
    return f;
}

int rsx_vertex_decode_element(
    u32 type, u32 size, const u8* source, float out[4])
{
    if (!source || !out)
        return 0;
    out[0] = out[1] = out[2] = 0.0f;
    out[3] = 1.0f;
    for (u32 component = 0; component < size && component < 4;
         component++) {
        switch (type) {
        case RSX_VTX_TYPE_FLOAT:
            out[component] = rsx_compact_be_f32(source + component * 4);
            break;
        case RSX_VTX_TYPE_HALF:
            out[component] = rsx_compact_be_f16(source + component * 2);
            break;
        case RSX_VTX_TYPE_UNORM8:
            out[component] = source[component] / 255.0f;
            break;
        case RSX_VTX_TYPE_UINT8:
            out[component] = (float)source[component];
            break;
        case RSX_VTX_TYPE_SNORM16: {
            const s16 value = (s16)(
                (source[component * 2] << 8) | source[component * 2 + 1]);
            out[component] = value / 32767.0f;
            break;
        }
        case RSX_VTX_TYPE_SINT16: {
            const s16 value = (s16)(
                (source[component * 2] << 8) | source[component * 2 + 1]);
            out[component] = (float)value;
            break;
        }
        case RSX_VTX_TYPE_CMP32: {
            const u32 word =
                ((u32)source[0] << 24) | ((u32)source[1] << 16) |
                ((u32)source[2] << 8) | (u32)source[3];
            s32 x = (s32)(word & 0x7FF);
            s32 y = (s32)((word >> 11) & 0x7FF);
            s32 z = (s32)((word >> 22) & 0x3FF);
            if (x & 0x400) x -= 0x800;
            if (y & 0x400) y -= 0x800;
            if (z & 0x200) z -= 0x400;
            out[0] = (float)x / 1023.0f;
            out[1] = (float)y / 1023.0f;
            out[2] = (float)z / 511.0f;
            out[3] = 1.0f;
            return 1;
        }
        default:
            return 0;
        }
    }
    return 1;
}

void rsx_vertex_layout_plan_init(
    rsx_vertex_layout_plan* plan, u32 input_mask)
{
    if (!plan)
        return;
    memset(plan, 0, sizeof(*plan));
    plan->mask = input_mask & 0xFFFFu;
    for (u32 attr = 0; attr < RSX_DSP_NUM_VERTEX_ATTR; attr++)
        plan->slot_by_attr[attr] = -1;
    for (u32 attr = 0; attr < RSX_DSP_NUM_VERTEX_ATTR; attr++) {
        if (!(plan->mask & (1u << attr)))
            continue;
        const u32 slot = plan->count++;
        plan->attrs[slot] = (u8)attr;
        plan->slot_by_attr[attr] = (s8)slot;
    }
    plan->stride = (u32)plan->count * 16u;
}

void rsx_vertex_fetch_default(const rsx_dispatch* rsx, u32 attr, float out[4])
{
    rsx_dsp_vertex_default(rsx, attr, out);
    if (attr != 3)
        return;
    const u32 base = (0x1C00u + attr * 0x10u) >> 2;
    const u32 written = rsx->seen[base] | rsx->seen[base + 1] |
        rsx->seen[base + 2] | rsx->seen[base + 3];
    /* Preserve the legacy white color fallback only for an unwritten
     * attribute. An explicit (0,0,0,1) is shader input, even when it equals
     * the hardware default; a shader may swizzle x into output alpha. */
    if (!written && out[0] == 0.0f && out[1] == 0.0f &&
        out[2] == 0.0f && out[3] == 1.0f)
        out[0] = out[1] = out[2] = 1.0f;
}

void rsx_vertex_fetch_plan_init(
    rsx_vertex_fetch_plan* plan, const rsx_dispatch* rsx,
    const rsx_vertex_layout_plan* layout,
    rsx_vertex_guest_ptr_fn guest_ptr, void* guest_user)
{
    if (!plan || !rsx || !layout)
        return;
    memset(plan, 0, sizeof(*plan));
    plan->layout = *layout;
    plan->base_offset = rsx_dsp_vertex_data_base_offset(rsx);
    plan->divider_mask = rsx_dsp_reg(rsx, 0x1FC0);
    plan->guest_ptr = guest_ptr;
    plan->guest_user = guest_user;
    for (u32 attr = 0; attr < RSX_DSP_NUM_VERTEX_ATTR; attr++) {
        rsx_vertex_fetch_attr* fetch = &plan->attr[attr];
        rsx_dsp_get_vertex_attr(rsx, attr, &fetch->desc);
        fetch->elem_size =
            rsx_vertex_attrib_size(fetch->desc.type, fetch->desc.size);
        fetch->stride =
            fetch->desc.stride ? fetch->desc.stride : fetch->elem_size;
        rsx_vertex_fetch_default(rsx, attr, fetch->default_value);
    }
}

u32 rsx_vertex_inline_layout(
    const rsx_dispatch* rsx, u32 offsets[RSX_DSP_NUM_VERTEX_ATTR])
{
    u32 packed = 0, declared = 0;
    for (u32 attr = 0; attr < RSX_DSP_NUM_VERTEX_ATTR; attr++) {
        rsx_dsp_vertex_attr desc;
        rsx_dsp_get_vertex_attr(rsx, attr, &desc);
        if (offsets)
            offsets[attr] = packed;
        if (!desc.type || !desc.size)
            continue;
        packed += rsx_vertex_attrib_size(desc.type, desc.size);
        if (desc.stride > declared)
            declared = desc.stride;
    }
    return declared ? declared : packed;
}

void rsx_vertex_fetch_plan_set_inline(
    rsx_vertex_fetch_plan* plan, const rsx_dispatch* rsx,
    const u8* data, u32 bytes)
{
    if (!plan)
        return;
    plan->inline_data = NULL;
    plan->inline_bytes = 0;
    if (!data || !bytes || !rsx)
        return;
    plan->inline_stride = rsx_vertex_inline_layout(rsx, plan->inline_off);
    plan->inline_data = data;
    plan->inline_bytes = bytes;
}

void rsx_vertex_fetch_plan_prepare(
    rsx_vertex_fetch_plan* plan, const rsx_vertex_ref* refs, u32 count)
{
    if (!plan)
        return;
    /* Ordinary per-vertex attributes (rsx_vertex_freq_plain) all span the same
     * elements: find them once for the draw instead of once per attribute.
     * The per-attribute walk was 2.2 ms of the RSX thread a frame in
     * inFAMOUS 2's New Marais. */
    int plain_known = 0;
    u32 plain_first = UINT_MAX, plain_last = 0;
    for (u32 slot = 0; slot < plan->layout.count; slot++) {
        const u32 attr = plan->layout.attrs[slot];
        rsx_vertex_fetch_attr* fetch = &plan->attr[attr];
        fetch->stable_base = NULL;
        fetch->stable_first = 0;
        fetch->stable_last = 0;
        if (!refs || !count ||
            !fetch->desc.type || !fetch->desc.size || !fetch->elem_size)
            continue;

        /* INLINE_ARRAY: the stream is already one contiguous, correctly
         * strided array per attribute, which is exactly what stable_base
         * describes -- so it needs no resolver and no bounds fallback, only
         * the right base. */
        if (plan->inline_data) {
            const u32 stride =
                plan->inline_stride ? plan->inline_stride : fetch->stride;
            const u32 off = plan->inline_off[attr];
            if (!stride || off + fetch->elem_size > stride)
                continue;
            const u32 verts = plan->inline_bytes / stride;
            if (!verts)
                continue;
            fetch->stride = stride;
            fetch->stable_base = plan->inline_data + off;
            fetch->stable_first = 0;
            fetch->stable_last = verts - 1;
            continue;
        }

        if (!plan->guest_ptr)
            continue;

        u32 first = UINT_MAX;
        u32 last = 0;
        if (rsx_vertex_freq_plain(fetch->desc.frequency,
                                  (plan->divider_mask >> attr) & 1u)) {
            if (!plain_known) {
                plain_known = 1;
                for (u32 i = 0; i < count; i++) {
                    const u32 element = rsx_vertex_element_index(
                        refs[i].vertex_id, refs[i].base_index, 0, 0);
                    if (element < plain_first) plain_first = element;
                    if (element > plain_last) plain_last = element;
                }
            }
            first = plain_first;
            last = plain_last;
        } else for (u32 i = 0; i < count; i++) {
            const u32 element = rsx_vertex_element_index(
                refs[i].vertex_id, refs[i].base_index,
                fetch->desc.frequency,
                (plan->divider_mask >> attr) & 1u);
            if (element < first) first = element;
            if (element > last) last = element;
        }

        const u64 prefix = rsx_dsp_resolve_vertex_offset(
            plan->base_offset, fetch->desc.offset);
        const u64 start = prefix + (u64)first * fetch->stride;
        const u64 end =
            prefix + (u64)last * fetch->stride + fetch->elem_size;
        if (start > UINT_MAX || end > 0x100000000ull || end <= start ||
            end - start > UINT_MAX)
            continue;

        const u8* base = plan->guest_ptr(
            plan->guest_user, fetch->desc.location, (u32)start,
            (u32)(end - start));
        if (!base)
            continue;
        fetch->stable_base = base;
        fetch->stable_first = first;
        fetch->stable_last = last;
    }
}

/* One attribute of one vertex, exactly as rsx_vertex_fetch_one does it. */
static int rsx_vertex_fetch_attr_value(
    const rsx_vertex_fetch_plan* plan, u32 attr,
    const rsx_vertex_ref* ref, float* value)
{
    const rsx_vertex_fetch_attr* fetch = &plan->attr[attr];
    if (!fetch->desc.type || !fetch->desc.size) {
        memcpy(value, fetch->default_value, 16);
        return 1;
    }

    const u32 element = rsx_vertex_element_index(
        ref->vertex_id, ref->base_index, fetch->desc.frequency,
        (plan->divider_mask >> attr) & 1u);
    const u8* source = NULL;
    if (fetch->stable_base &&
        element >= fetch->stable_first && element <= fetch->stable_last) {
        source = fetch->stable_base +
            (u64)(element - fetch->stable_first) * fetch->stride;
    } else if (plan->inline_data) {
        /* An inline draw has no vertex array to fall back to: the array
         * offsets still hold whatever the last ordinary draw bound, and
         * reading them would feed this draw another mesh's vertices.
         * An attribute the stream does not supply gets its default,
         * which is what the hardware feeds. */
        source = NULL;
    } else if (plan->guest_ptr) {
        const u32 offset =
            rsx_dsp_resolve_vertex_offset(plan->base_offset,
                                          fetch->desc.offset) +
            element * fetch->stride;
        source = plan->guest_ptr(
            plan->guest_user, fetch->desc.location, offset,
            fetch->elem_size);
    }

    if (source && rsx_vertex_decode_element(
            fetch->desc.type, fetch->desc.size, source, value))
        return 1;
    if (attr == 0)
        return 0;
    memcpy(value, fetch->default_value, 16);
    return 1;
}

int rsx_vertex_fetch_one(
    const rsx_vertex_fetch_plan* plan, const rsx_vertex_ref* ref, u8* out)
{
    if (!plan || !ref || (!out && plan->layout.stride))
        return 0;
    for (u32 slot = 0; slot < plan->layout.count; slot++)
        if (!rsx_vertex_fetch_attr_value(plan, plan->layout.attrs[slot], ref,
                                   (float*)(out + slot * 16u)))
            return 0;
    return 1;
}

#if defined(__SSSE3__)
#include <tmmintrin.h>
/* Four-lane decoders for the stable-range loop. Each gives exactly the floats
 * the scalar decoders above give (same IEEE operations, lane by lane): the
 * integer types convert exactly and divide with divps instead of divss, and
 * the half conversion is the same bit construction (zero/denormal -> signed
 * zero, exponent 31 -> inf/NaN with the mantissa kept). Lanes past the
 * attribute's size read as 0 and the w lane as 1, as the scalar loop presets. */
static inline __m128i rsx_v_load(const u8* src, u32 bytes)
{
    /* Direct loads of exactly `bytes` (never past the element: it may end
     * the mapping). A copy through a stack temporary would stall on store
     * forwarding for every vertex. */
    u32 a, b;
    switch (bytes) {
    case 16: return _mm_loadu_si128((const __m128i*)src);
    case 12: memcpy(&a, src + 8, 4);
             return _mm_unpacklo_epi64(_mm_loadl_epi64((const __m128i*)src), _mm_cvtsi32_si128((int)a));
    case 8:  return _mm_loadl_epi64((const __m128i*)src);
    case 6:  memcpy(&a, src, 4); b = (u32)src[4] | ((u32)src[5] << 8);
             return _mm_unpacklo_epi32(_mm_cvtsi32_si128((int)a), _mm_cvtsi32_si128((int)b));
    case 4:  memcpy(&a, src, 4); return _mm_cvtsi32_si128((int)a);
    case 2:  return _mm_cvtsi32_si128((int)((u32)src[0] | ((u32)src[1] << 8)));
    default: break;
    }
    u8 t[16] = { 0 };
    memcpy(t, src, bytes);
    return _mm_loadu_si128((const __m128i*)t);
}
static inline __m128 rsx_v_w1(__m128 v, u32 n)
{
    /* Lane 3 is +0.0 when n < 4: OR in the bits of 1.0f. */
    return n < 4 ? _mm_or_ps(v, _mm_castsi128_ps(_mm_set_epi32(0x3F800000, 0, 0, 0))) : v;
}
static inline __m128 rsx_v_f32(const u8* src, u32 n)
{
    const __m128i sw = _mm_setr_epi8(3, 2, 1, 0, 7, 6, 5, 4, 11, 10, 9, 8, 15, 14, 13, 12);
    return rsx_v_w1(_mm_castsi128_ps(_mm_shuffle_epi8(rsx_v_load(src, n * 4u), sw)), n);
}
/* Big-endian 16-bit lanes to 32-bit lanes: zero-extended, or shifted to the
 * top for a sign-extending shift. */
static inline __m128i rsx_v_u16(const u8* src, u32 n)
{
    const __m128i sw = _mm_setr_epi8(1, 0, -1, -1, 3, 2, -1, -1, 5, 4, -1, -1, 7, 6, -1, -1);
    return _mm_shuffle_epi8(rsx_v_load(src, n * 2u), sw);
}
static inline __m128i rsx_v_s16(const u8* src, u32 n)
{
    const __m128i sw = _mm_setr_epi8(-1, -1, 1, 0, -1, -1, 3, 2, -1, -1, 5, 4, -1, -1, 7, 6);
    return _mm_srai_epi32(_mm_shuffle_epi8(rsx_v_load(src, n * 2u), sw), 16);
}
static inline __m128 rsx_v_f16(const u8* src, u32 n)
{
    const __m128i h = rsx_v_u16(src, n);
    const __m128i e = _mm_and_si128(h, _mm_set1_epi32(0x7C00));
    const __m128i zero = _mm_cmpeq_epi32(e, _mm_setzero_si128());
    const __m128i inf = _mm_cmpeq_epi32(e, _mm_set1_epi32(0x7C00));
    /* ((exp + 112) << 23) | (man << 13) == ((h & 0x7FFF) << 13) + (112 << 23);
     * exponent 31 wants 0x7F800000 | (man << 13): add 224 << 23 instead. */
    const __m128i add = _mm_add_epi32(_mm_set1_epi32(0x38000000),
                                      _mm_and_si128(inf, _mm_set1_epi32(0x38000000)));
    __m128i r = _mm_add_epi32(_mm_slli_epi32(_mm_and_si128(h, _mm_set1_epi32(0x7FFF)), 13), add);
    r = _mm_andnot_si128(zero, r);
    r = _mm_or_si128(r, _mm_slli_epi32(_mm_and_si128(h, _mm_set1_epi32(0x8000)), 16));
    return rsx_v_w1(_mm_castsi128_ps(r), n);
}
static inline __m128 rsx_v_cmp32(const u8* src)
{
    const u32 word = ((u32)src[0] << 24) | ((u32)src[1] << 16) |
                     ((u32)src[2] << 8) | (u32)src[3];
    const s32 x = (s32)(word << 21) >> 21;
    const s32 y = (s32)(word << 10) >> 21;
    const s32 z = (s32)word >> 22;
    return _mm_div_ps(_mm_cvtepi32_ps(_mm_setr_epi32(x, y, z, 1)),
                      _mm_setr_ps(1023.0f, 1023.0f, 511.0f, 1.0f));
}
#define RSX_FETCH_SIMD 1
#endif

/* The stable-range loop for one attribute and one vertex type: `decode`
 * reads `src` into `v` (pre-set to 0,0,0,1) for components c < N. Elements
 * outside the stable range take the one-vertex path. N is a constant at each
 * use (RSX_FETCH_SIZES), so the component loop unrolls. */
#define RSX_FETCH_LOOP(N, decode)                                               \
    for (u32 i = 0; i < count; i++) {                                           \
        float* v = (float*)(out + (size_t)i * stride + slot * 16u);             \
        const u32 element = plain                                               \
            ? ((refs[i].vertex_id + refs[i].base_index) & 0x000FFFFFu)          \
            : rsx_vertex_element_index(refs[i].vertex_id, refs[i].base_index,   \
                                       fetch->desc.frequency, modulo);          \
        if (element - first > span) {                                          \
            if (!rsx_vertex_fetch_attr_value(plan, attr, &refs[i], v)) return 0;      \
            continue;                                                           \
        }                                                                       \
        const u8* src = base + (u64)(element - first) * fstride;                \
        float w[4] = { 0.0f, 0.0f, 0.0f, 1.0f };                                \
        for (u32 c = 0; c < (N); c++) { decode; } \
        memcpy(v, w, 16);                                                       \
    }
/* One RSX_FETCH_LOOP per component count. */
#define RSX_FETCH_SIZES(decode)                                                 \
    switch (n) {                                                                \
    case 1: RSX_FETCH_LOOP(1, decode) break;                                    \
    case 2: RSX_FETCH_LOOP(2, decode) break;                                    \
    case 3: RSX_FETCH_LOOP(3, decode) break;                                    \
    default: RSX_FETCH_LOOP(4, decode) break;                                   \
    }

#ifdef RSX_FETCH_SIMD
/* RSX_FETCH_LOOP with a four-lane decoder: `expr` gives the vertex's __m128. */
#define RSX_FETCH_LOOP_V(expr)                                                  \
    for (u32 i = 0; i < count; i++) {                                           \
        float* v = (float*)(out + (size_t)i * stride + slot * 16u);             \
        const u32 element = plain                                               \
            ? ((refs[i].vertex_id + refs[i].base_index) & 0x000FFFFFu)          \
            : rsx_vertex_element_index(refs[i].vertex_id, refs[i].base_index,   \
                                       fetch->desc.frequency, modulo);          \
        if (element - first > span) {                                          \
            if (!rsx_vertex_fetch_attr_value(plan, attr, &refs[i], v)) return 0;      \
            continue;                                                           \
        }                                                                       \
        const u8* src = base + (u64)(element - first) * fstride;                \
        _mm_storeu_ps(v, (expr));                                               \
    }
#define RSX_FETCH_SIZES_V(fn)                                                   \
    switch (n) {                                                                \
    case 1: RSX_FETCH_LOOP_V(fn(src, 1)) break;                                 \
    case 2: RSX_FETCH_LOOP_V(fn(src, 2)) break;                                 \
    case 3: RSX_FETCH_LOOP_V(fn(src, 3)) break;                                 \
    default: RSX_FETCH_LOOP_V(fn(src, 4)) break;                                \
    }
#endif

int rsx_vertex_fetch_many(
    const rsx_vertex_fetch_plan* plan, const rsx_vertex_ref* refs, u32 count,
    u8* out)
{
    if (!plan || (!refs && count) || (!out && plan->layout.stride && count))
        return 0;
    const u32 stride = plan->layout.stride;
    for (u32 slot = 0; slot < plan->layout.count; slot++) {
        const u32 attr = plan->layout.attrs[slot];
        const rsx_vertex_fetch_attr* fetch = &plan->attr[attr];
        const u8* base = fetch->stable_base;
        const u32 type = fetch->desc.type;
        /* A disabled array: every vertex reads the attribute's default, as
         * rsx_vertex_fetch_attr_value would one vertex at a time. */
        if (!type || !fetch->desc.size) {
            for (u32 i = 0; i < count; i++)
                memcpy(out + (size_t)i * stride + slot * 16u, fetch->default_value, 16u);
            continue;
        }
        if (!base || type < RSX_VTX_TYPE_SNORM16 || type > RSX_VTX_TYPE_UINT8) {
            for (u32 i = 0; i < count; i++)
                if (!rsx_vertex_fetch_attr_value(plan, attr, &refs[i],
                        (float*)(out + (size_t)i * stride + slot * 16u)))
                    return 0;
            continue;
        }
        const u32 first = fetch->stable_first;
        const u32 span = fetch->stable_last - first;
        const u32 fstride = fetch->stride;
        const u32 modulo = (plan->divider_mask >> attr) & 1u;
        const int plain = rsx_vertex_freq_plain(fetch->desc.frequency, modulo);
        const u32 n = fetch->desc.size < 4 ? fetch->desc.size : 4;
        switch (type) {
        case RSX_VTX_TYPE_FLOAT:
#ifdef RSX_FETCH_SIMD
            RSX_FETCH_SIZES_V(rsx_v_f32)
#else
            RSX_FETCH_SIZES(w[c] = rsx_compact_be_f32(src + c * 4))
#endif
            break;
        case RSX_VTX_TYPE_HALF:
#ifdef RSX_FETCH_SIMD
            RSX_FETCH_SIZES_V(rsx_v_f16)
#else
            RSX_FETCH_SIZES(w[c] = rsx_compact_be_f16(src + c * 2))
#endif
            break;
        case RSX_VTX_TYPE_UNORM8:
            RSX_FETCH_SIZES(w[c] = src[c] / 255.0f)
            break;
        case RSX_VTX_TYPE_UINT8:
            RSX_FETCH_SIZES(w[c] = (float)src[c])
            break;
        case RSX_VTX_TYPE_SNORM16:
#ifdef RSX_FETCH_SIMD
#define RSX_V_SN16(p, k) rsx_v_w1(_mm_div_ps(_mm_cvtepi32_ps(rsx_v_s16(p, k)), _mm_set1_ps(32767.0f)), k)
            RSX_FETCH_SIZES_V(RSX_V_SN16)
#undef RSX_V_SN16
#else
            RSX_FETCH_SIZES(w[c] = (s16)((src[c * 2] << 8) | src[c * 2 + 1]) / 32767.0f)
#endif
            break;
        case RSX_VTX_TYPE_SINT16:
#ifdef RSX_FETCH_SIMD
#define RSX_V_SI16(p, k) rsx_v_w1(_mm_cvtepi32_ps(rsx_v_s16(p, k)), k)
            RSX_FETCH_SIZES_V(RSX_V_SI16)
#undef RSX_V_SI16
#else
            RSX_FETCH_SIZES(w[c] = (float)(s16)((src[c * 2] << 8) | src[c * 2 + 1]))
#endif
            break;
        default:   /* RSX_VTX_TYPE_CMP32: one packed word, whatever the size
                    * (the same arithmetic as rsx_vertex_decode_element) */
#ifdef RSX_FETCH_SIMD
            RSX_FETCH_LOOP_V(rsx_v_cmp32(src))
#else
            RSX_FETCH_LOOP(1, {
                const u32 word = ((u32)src[0] << 24) | ((u32)src[1] << 16) |
                                 ((u32)src[2] << 8) | (u32)src[3];
                s32 x = (s32)(word & 0x7FF);
                s32 y = (s32)((word >> 11) & 0x7FF);
                s32 z = (s32)((word >> 22) & 0x3FF);
                if (x & 0x400) x -= 0x800;
                if (y & 0x400) y -= 0x800;
                if (z & 0x200) z -= 0x400;
                w[0] = (float)x / 1023.0f;
                w[1] = (float)y / 1023.0f;
                w[2] = (float)z / 511.0f;
                w[3] = 1.0f;
            })
#endif
            break;
        }
    }
    return 1;
}
#undef RSX_FETCH_SIZES
#undef RSX_FETCH_LOOP
#ifdef RSX_FETCH_SIMD
#undef RSX_FETCH_SIZES_V
#undef RSX_FETCH_LOOP_V
#endif

static u64 rsx_vertex_remap_hash(u64 value)
{
    value ^= value >> 33;
    value *= 0xff51afd7ed558ccdull;
    value ^= value >> 33;
    value *= 0xc4ceb9fe1a85ec53ull;
    value ^= value >> 33;
    return value;
}

int rsx_vertex_remap_build(
    rsx_vertex_remap* remap, rsx_vertex_ref* refs, u32 count,
    u32* unique_count)
{
    if (!remap || (!refs && count) || !unique_count)
        return 0;
    *unique_count = count;
    if (!count)
        return 1;

    if (count > remap->occurrence_capacity) {
        u32* occurrences = (u32*)realloc(
            remap->occurrence_to_unique, (size_t)count * sizeof(u32));
        if (!occurrences)
            return 0;
        remap->occurrence_to_unique = occurrences;
        remap->occurrence_capacity = count;
    }

    if (count > UINT_MAX / 2u)
        return 0;
    u32 slot_capacity = 16u;
    const u32 required_slots = count * 2u;
    while (slot_capacity < required_slots) {
        if (slot_capacity > UINT_MAX / 2u)
            return 0;
        slot_capacity *= 2u;
    }
    if (slot_capacity > remap->slot_capacity) {
        rsx_vertex_remap_slot* slots = (rsx_vertex_remap_slot*)realloc(
            remap->slots, (size_t)slot_capacity * sizeof(*slots));
        if (!slots)
            return 0;
        remap->slots = slots;
        remap->slot_capacity = slot_capacity;
        memset(remap->slots, 0,
               (size_t)remap->slot_capacity * sizeof(*remap->slots));
        remap->generation = 1u;
    } else {
        remap->generation++;
        if (!remap->generation) {
            memset(remap->slots, 0,
                   (size_t)remap->slot_capacity * sizeof(*remap->slots));
            remap->generation = 1u;
        }
    }

    u32 unique = 0;
    const u32 mask = remap->slot_capacity - 1u;
    for (u32 occurrence = 0; occurrence < count; occurrence++) {
        const rsx_vertex_ref ref = refs[occurrence];
        const u64 key = ((u64)ref.base_index << 32) | ref.vertex_id;
        u32 slot_index = (u32)rsx_vertex_remap_hash(key) & mask;
        for (;;) {
            rsx_vertex_remap_slot* slot = &remap->slots[slot_index];
            if (slot->generation != remap->generation) {
                slot->key = key;
                slot->value = unique;
                slot->generation = remap->generation;
                refs[unique] = ref;
                remap->occurrence_to_unique[occurrence] = unique;
                unique++;
                break;
            }
            if (slot->key == key) {
                remap->occurrence_to_unique[occurrence] = slot->value;
                break;
            }
            slot_index = (slot_index + 1u) & mask;
        }
    }
    *unique_count = unique;
    return 1;
}

u32 rsx_vertex_remap_index(
    const rsx_vertex_remap* remap, u32 occurrence)
{
    return remap && occurrence < remap->occurrence_capacity
        ? remap->occurrence_to_unique[occurrence] : occurrence;
}

void rsx_vertex_remap_destroy(rsx_vertex_remap* remap)
{
    if (!remap)
        return;
    free(remap->occurrence_to_unique);
    free(remap->slots);
    memset(remap, 0, sizeof(*remap));
}

int rsx_vertex_topology_plan(
    u32 primitive, int refs_remapped, int* indexed)
{
    if (!indexed)
        return 0;
    switch (primitive) {
    case RSX_PRIMITIVE_TRIANGLES:
        *indexed = refs_remapped != 0;
        return 1;
    case RSX_PRIMITIVE_TRIANGLE_STRIP:
    case RSX_PRIMITIVE_TRIANGLE_FAN:
    case RSX_PRIMITIVE_QUADS:
        *indexed = 1;
        return 1;
    default:
        *indexed = 0;
        return 0;
    }
}
