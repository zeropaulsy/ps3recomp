/* SPU_DMA_DEFER (spu_channels.c): a plain GET is queued at issue and copied at the SPU's next
 * channel operation; nothing may observe the destination before that.
 *
 *   ./fixture         SPU_DMA_DEFER=1 (set by the runner): queue, flush points, overflow, contexts
 *   ./fixture off     variable unset: every GET is copied at issue
 *   ./fixture ring    + SPU_DGQ_RING=1: a GET into a full queue copies only the oldest entry
 *
 * tools/test_spu_dma_defer.py builds and runs both. */
#include <assert.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
static uint8_t guest[1u << 20];
uint8_t* vm_base = guest;
uint32_t ppu_vm_size;
#include "../spu_channels.c"

unsigned long long ps3_ms_now(void) { return 1000; }

static spu_context a, b;

static void cmd(spu_context* c, uint32_t op, uint32_t ea, uint32_t lsa, uint32_t size)
{
    c->mfc_eah = 0; c->mfc_eal = ea; c->mfc_lsa = lsa; c->mfc_size = size; c->mfc_tag = 3;
    spu_wrch_mfc_cmd(c, op);
}
static int got(spu_context* c, uint32_t lsa, uint32_t ea, uint32_t size)
{
    return !memcmp(c->ls + lsa, guest + ea, size);
}
static int zero(spu_context* c, uint32_t lsa, uint32_t size)
{
    for (uint32_t i = 0; i < size; i++) if (c->ls[lsa + i]) return 0;
    return 1;
}

int main(int argc, char** argv)
{
    const int off = argc > 1 && !strcmp(argv[1], "off");
    spu_context_init(&a, 0); spu_context_init(&b, 1);
    a.image_id = 5; b.image_id = 5;
    for (uint32_t i = 0; i < 0x10000; i++) guest[0x20000 + i] = (uint8_t)(i * 7 + 1);
    g_vm_page_bitmap[0] |= 0x0C;                       /* pages 2 and 3 committed */

    /* The first command takes the slow path (the repeat limit is read there), the next ones
     * the lean path: both queue. */
    cmd(&a, 0x40, 0x20000, 0x1000, 64);
    cmd(&a, 0x40, 0x20100, 0x1100, 256);
    if (off) {
        assert(got(&a, 0x1000, 0x20000, 64) && got(&a, 0x1100, 0x20100, 256));
        assert(t_dgq.n == 0);
        spu_mfc_release(&a); spu_mfc_release(&b);
        puts("PASS: without SPU_DMA_DEFER every GET is copied at issue");
        return 0;
    }
    assert(zero(&a, 0x1000, 64) && zero(&a, 0x1100, 256) && t_dgq.n == 2);

    /* A tag-status update (the usual wait) flushes, in order. */
    spu_wrch_mfc_tag_update(&a, 0);
    assert(got(&a, 0x1000, 0x20000, 64) && got(&a, 0x1100, 0x20100, 256) && t_dgq.n == 0);

    /* A later GET over the same LS replaces it at the flush, not before. */
    cmd(&a, 0x40, 0x20400, 0x1000, 64);
    assert(got(&a, 0x1000, 0x20000, 64));
    assert(spu_rchcnt(&a, MFC_RdTagStat) || 1);         /* a channel count flushes */
    assert(got(&a, 0x1000, 0x20400, 64));

    /* A PUT flushes first: it may read what an earlier GET brought in. */
    cmd(&a, 0x40, 0x20800, 0x2000, 128);
    cmd(&a, 0x20, 0x30000, 0x2000, 128);
    assert(got(&a, 0x2000, 0x20800, 128) && !memcmp(guest + 0x30000, guest + 0x20800, 128));

    if (argc > 1 && !strcmp(argv[1], "ring")) {
        /* Sixteen entries: each further GET runs only the oldest one, the ring wraps, and a
         * flush copies the rest oldest first (two GETs to one LS address: the later wins). */
        for (uint32_t i = 0; i < 20; i++) cmd(&a, 0x40, 0x21000 + i * 32, 0x4000 + i * 16, 16);
        for (uint32_t i = 0; i < 4; i++) assert(got(&a, 0x4000 + i * 16, 0x21000 + i * 32, 16));
        for (uint32_t i = 4; i < 20; i++) assert(zero(&a, 0x4000 + i * 16, 16));
        assert(t_dgq.n == 16 && t_dgq.h == 4);
        cmd(&a, 0x40, 0x21800, 0x4000 + 19 * 16, 16);
        assert(got(&a, 0x4000 + 4 * 16, 0x21000 + 4 * 32, 16) && t_dgq.n == 16);
        spu_wrch_mfc_tag_update(&a, 0);
        for (uint32_t i = 0; i < 19; i++) assert(got(&a, 0x4000 + i * 16, 0x21000 + i * 32, 16));
        assert(got(&a, 0x4000 + 19 * 16, 0x21800, 16) && t_dgq.n == 0 && t_dgq.h == 0);
        spu_mfc_release(&a); spu_mfc_release(&b);
        puts("PASS: SPU_DGQ_RING=1: a full queue copies its oldest GET, a flush the rest in order");
        return 0;
    }

    /* Sixteen entries: the seventeenth GET runs the first sixteen. */
    for (uint32_t i = 0; i < 17; i++) cmd(&a, 0x40, 0x21000 + i * 32, 0x4000 + i * 16, 16);
    for (uint32_t i = 0; i < 16; i++) assert(got(&a, 0x4000 + i * 16, 0x21000 + i * 32, 16));
    assert(zero(&a, 0x4000 + 16 * 16, 16) && t_dgq.n == 1);

    /* Another context's GET runs this one's queue. */
    cmd(&b, 0x40, 0x22000, 0x1000, 32);
    assert(got(&a, 0x4000 + 16 * 16, 0x21000 + 16 * 32, 16));
    assert(zero(&b, 0x1000, 32) && t_dgq.n == 1 && t_dgq.ctx == &b);

    /* stop flushes (stop-and-signal handlers read local store). */
    spu_stop(&b);
    assert(got(&b, 0x1000, 0x22000, 32) && t_dgq.n == 0);

    /* A channel read flushes. */
    cmd(&b, 0x40, 0x22100, 0x1100, 48);
    (void)spu_rdch_u32(&b, SPU_RdEventMask);
    assert(got(&b, 0x1100, 0x22100, 48));

    spu_mfc_release(&a); spu_mfc_release(&b);
    puts("PASS: deferred GETs are copied at tag update, channel count/read, PUT, a full queue, "
         "another context and stop, in issue order");
    return 0;
}

void ppu_resv_break_store(uint64_t ea)                 { (void)ea; }
void ps3_ww_report_inline(uint32_t a_, uint64_t v, int w) { (void)a_; (void)v; (void)w; }
uint32_t g_ww_lo = 0xFFFFFFFFu, g_ww_hi = 0;
int      g_resv_store_active = 0;
uint32_t g_barrier_sync_watch = 0;
void (*g_spurs_kernel_hook)(uint32_t) = 0;
uint32_t g_ydkj_spurs_ctx_ea = 0;
uint32_t g_ydkj_real_spurs_ea = 0;
uint32_t g_ydkj_real_taskset_ea = 0;
uint32_t g_ydkj_real_taskid = 0;
void spurs_ef_set_from_spu(uint32_t ea, uint16_t bits) { (void)ea; (void)bits; }
void ydkj_wake_all_event_flags(void)                   { }
void spurs_pm_build_context(spu_context* c, uint32_t x, uint32_t y, uint32_t d)
{
    (void)c; (void)x; (void)y; (void)d;
}
uint32_t g_spu_image_src_ea, g_spu_image_ls_start, g_spu_image_span;
void spu_thread_publish_ctx(uint32_t tid, void* ctx) { (void)tid; (void)ctx; }
