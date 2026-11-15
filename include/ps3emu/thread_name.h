/*
 * ps3_thread_name: name the calling host thread ("ppu main", "spu <group>:0",
 * "audio mix", ...), so a debugger, a profiler or a per-thread CPU table can
 * tell the guest's PPU and SPU threads and the runtime's own threads apart.
 *
 * Windows: SetThreadDescription (Windows 10 1607 and later) is looked up at
 * run time, so older systems just skip it. Linux: prctl(PR_SET_NAME), which
 * names the calling thread and, unlike pthread_setname_np, needs no
 * _GNU_SOURCE before the first pthread.h. Elsewhere it does nothing.
 */
#ifndef PS3EMU_THREAD_NAME_H
#define PS3EMU_THREAD_NAME_H

#include <stdarg.h>
#include <stdio.h>

#ifdef _WIN32
#include <windows.h>

static inline void ps3_thread_name(const char* fmt, ...)
{
    typedef HRESULT (WINAPI *set_desc_fn)(HANDLE, PCWSTR);
    static set_desc_fn set_desc;
    static int looked_up;
    if (!looked_up) {
        set_desc = (set_desc_fn)(void*)GetProcAddress(GetModuleHandleA("kernel32.dll"),
                                                      "SetThreadDescription");
        looked_up = 1;
    }
    if (!set_desc) return;
    char s[64];
    va_list ap;
    va_start(ap, fmt);
    vsnprintf(s, sizeof s, fmt, ap);
    va_end(ap);
    WCHAR w[64];
    int i = 0;
    for (; s[i] && i < 63; i++) w[i] = (WCHAR)(unsigned char)s[i];
    w[i] = 0;
    set_desc(GetCurrentThread(), w);
}
#elif defined(__linux__)
#include <sys/prctl.h>
#include <sys/syscall.h>
#include <unistd.h>

/* The kernel keeps 15 characters (PR_SET_NAME truncates), shown by ps -L,
 * top -H, perf and /proc/<pid>/task/<tid>/comm. The process's first thread
 * keeps its name: that name is the process name (/proc/<pid>/comm), which
 * pgrep -x, pidof and launcher watchdogs use to find the game. */
static inline void ps3_thread_name(const char* fmt, ...)
{
    if ((pid_t)syscall(SYS_gettid) == getpid()) return;
    char s[64];
    va_list ap;
    va_start(ap, fmt);
    vsnprintf(s, sizeof s, fmt, ap);
    va_end(ap);
    prctl(PR_SET_NAME, (unsigned long)s, 0, 0, 0);
}
#else
static inline void ps3_thread_name(const char* fmt, ...) { (void)fmt; }
#endif

#endif /* PS3EMU_THREAD_NAME_H */
