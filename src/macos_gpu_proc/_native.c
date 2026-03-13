/**
 * macos-gpu-proc: Per-process GPU utilization on macOS Apple Silicon.
 *
 * Reads per-client GPU accounting from the IORegistry's AGXDeviceUserClient
 * entries. Each GPU client (Metal command queue) is a child of the AGX
 * accelerator and carries:
 *   - "IOUserClientCreator" = "pid <N>, <process_name>"
 *   - "AppUsage" = [{"API"="Metal", "accumulatedGPUTime"=<ns>}, ...]
 *
 * This is the same data Activity Monitor reads. No sudo or entitlements
 * needed — the IORegistry is world-readable.
 *
 * Copyright 2026 Adam Mikulis. MIT License.
 */

#define PY_SSIZE_T_CLEAN
#include <Python.h>
#include <IOKit/IOKitLib.h>
#include <CoreFoundation/CoreFoundation.h>
#include <libproc.h>
#include <sys/proc_info.h>
#include <sys/resource.h>

/* ------------------------------------------------------------------ */
/* Helpers                                                             */
/* ------------------------------------------------------------------ */

/**
 * Extract PID from "IOUserClientCreator" value like "pid 418, WindowServer".
 * Returns -1 on parse failure.
 */
static int parse_creator_pid(CFStringRef creator) {
    char buf[256];
    if (!CFStringGetCString(creator, buf, sizeof(buf), kCFStringEncodingUTF8))
        return -1;
    int pid = -1;
    if (sscanf(buf, "pid %d", &pid) == 1)
        return pid;
    return -1;
}

/**
 * Extract process name from "IOUserClientCreator" value.
 * Writes into out_name (up to name_size bytes). Returns 0 on success.
 */
static int parse_creator_name(CFStringRef creator, char *out_name, size_t name_size) {
    char buf[256];
    if (!CFStringGetCString(creator, buf, sizeof(buf), kCFStringEncodingUTF8))
        return -1;
    /* Format: "pid <N>, <name>" */
    char *comma = strchr(buf, ',');
    if (!comma || *(comma + 1) != ' ')
        return -1;
    strncpy(out_name, comma + 2, name_size - 1);
    out_name[name_size - 1] = '\0';
    return 0;
}

/**
 * Sum accumulatedGPUTime from an "AppUsage" CFArray.
 * Each element is a CFDictionary with "accumulatedGPUTime" -> CFNumber.
 */
static long long sum_app_usage_gpu_time(CFArrayRef app_usage) {
    long long total = 0;
    CFIndex count = CFArrayGetCount(app_usage);
    for (CFIndex i = 0; i < count; i++) {
        CFDictionaryRef entry = CFArrayGetValueAtIndex(app_usage, i);
        if (!entry || CFGetTypeID(entry) != CFDictionaryGetTypeID())
            continue;
        CFNumberRef gpu_time = CFDictionaryGetValue(entry,
            CFSTR("accumulatedGPUTime"));
        if (!gpu_time || CFGetTypeID(gpu_time) != CFNumberGetTypeID())
            continue;
        long long ns = 0;
        CFNumberGetValue(gpu_time, kCFNumberSInt64Type, &ns);
        total += ns;
    }
    return total;
}

/* ------------------------------------------------------------------ */
/* Core: iterate AGXDeviceUserClient entries in IORegistry             */
/* ------------------------------------------------------------------ */

/**
 * Result struct for one GPU client.
 */
typedef struct {
    int pid;
    char name[128];
    long long gpu_ns;
} gpu_client_t;

/**
 * Read all AGXDeviceUserClient entries from the IORegistry.
 *
 * User client objects are !registered, so IOServiceGetMatchingServices
 * won't find them. Instead we find the AGXAccelerator parent and iterate
 * its children in the IOService plane.
 *
 * Allocates *out_clients (caller must free). Returns count, or -1 on error.
 */
static int read_gpu_clients(gpu_client_t **out_clients) {
    *out_clients = NULL;

    /* Find the AGX accelerator (parent of all GPU user clients) */
    io_iterator_t accel_iter;
    kern_return_t kr = IOServiceGetMatchingServices(
        kIOMainPortDefault,
        IOServiceMatching("AGXAccelerator"),
        &accel_iter);
    if (kr != KERN_SUCCESS)
        return -1;

    int capacity = 64;
    gpu_client_t *clients = malloc(capacity * sizeof(gpu_client_t));
    if (!clients) {
        IOObjectRelease(accel_iter);
        return -1;
    }
    int count = 0;

    /* Iterate each accelerator (typically just one) */
    io_service_t accel;
    while ((accel = IOIteratorNext(accel_iter)) != 0) {
        io_iterator_t child_iter;
        kr = IORegistryEntryGetChildIterator(accel, kIOServicePlane, &child_iter);
        IOObjectRelease(accel);
        if (kr != KERN_SUCCESS)
            continue;

        io_service_t child;
        while ((child = IOIteratorNext(child_iter)) != 0) {
            /* Read IOUserClientCreator */
            CFStringRef creator = IORegistryEntryCreateCFProperty(
                child, CFSTR("IOUserClientCreator"),
                kCFAllocatorDefault, 0);
            if (!creator || CFGetTypeID(creator) != CFStringGetTypeID()) {
                if (creator) CFRelease(creator);
                IOObjectRelease(child);
                continue;
            }

            int pid = parse_creator_pid(creator);
            if (pid < 0) {
                CFRelease(creator);
                IOObjectRelease(child);
                continue;
            }

            /* Read AppUsage */
            CFArrayRef app_usage = IORegistryEntryCreateCFProperty(
                child, CFSTR("AppUsage"),
                kCFAllocatorDefault, 0);

            long long gpu_ns = 0;
            if (app_usage && CFGetTypeID(app_usage) == CFArrayGetTypeID())
                gpu_ns = sum_app_usage_gpu_time(app_usage);

            /* Grow array if needed */
            if (count >= capacity) {
                capacity *= 2;
                gpu_client_t *tmp = realloc(clients, capacity * sizeof(gpu_client_t));
                if (!tmp) {
                    if (app_usage) CFRelease(app_usage);
                    CFRelease(creator);
                    IOObjectRelease(child);
                    break;
                }
                clients = tmp;
            }

            clients[count].pid = pid;
            clients[count].gpu_ns = gpu_ns;
            parse_creator_name(creator, clients[count].name, sizeof(clients[count].name));
            count++;

            if (app_usage) CFRelease(app_usage);
            CFRelease(creator);
            IOObjectRelease(child);
        }
        IOObjectRelease(child_iter);
    }

    IOObjectRelease(accel_iter);
    *out_clients = clients;
    return count;
}

/* ------------------------------------------------------------------ */
/* Python API                                                          */
/* ------------------------------------------------------------------ */

PyDoc_STRVAR(gpu_time_ns_doc,
"gpu_time_ns(pid=0) -> int\n\n"
"Return cumulative GPU time in nanoseconds for a process.\n\n"
"Reads accumulatedGPUTime from AGXDeviceUserClient entries in the\n"
"IORegistry. Multiple command queues for the same PID are summed.\n\n"
"Args:\n"
"    pid: Process ID. 0 means the calling process.\n\n"
"Returns:\n"
"    Cumulative GPU nanoseconds, or 0 if the process has no GPU clients.\n\n"
"Note:\n"
"    No special privileges required — IORegistry is world-readable.");

static PyObject* py_gpu_time_ns(PyObject* self, PyObject* args) {
    int pid = 0;
    if (!PyArg_ParseTuple(args, "|i", &pid))
        return NULL;

    if (pid == 0)
        pid = getpid();

    gpu_client_t *clients;
    int count = read_gpu_clients(&clients);
    if (count < 0)
        return PyLong_FromLongLong(0);

    long long total = 0;
    for (int i = 0; i < count; i++) {
        if (clients[i].pid == pid)
            total += clients[i].gpu_ns;
    }
    free(clients);
    return PyLong_FromLongLong(total);
}


PyDoc_STRVAR(gpu_time_ns_multi_doc,
"gpu_time_ns_multi(pids: list[int]) -> dict[int, int]\n\n"
"Batch read GPU nanoseconds for multiple PIDs in one IORegistry scan.\n\n"
"Args:\n"
"    pids: List of process IDs. Use 0 for the calling process.\n\n"
"Returns:\n"
"    Dict mapping each PID to its cumulative GPU nanoseconds.\n"
"    PIDs with no GPU clients map to 0.");

static PyObject* py_gpu_time_ns_multi(PyObject* self, PyObject* args) {
    PyObject* pid_list;
    if (!PyArg_ParseTuple(args, "O!", &PyList_Type, &pid_list))
        return NULL;

    pid_t my_pid = getpid();

    gpu_client_t *clients;
    int count = read_gpu_clients(&clients);

    Py_ssize_t n = PyList_GET_SIZE(pid_list);
    PyObject* result = PyDict_New();
    if (!result) {
        free(clients);
        return NULL;
    }

    for (Py_ssize_t i = 0; i < n; i++) {
        PyObject* item = PyList_GET_ITEM(pid_list, i);
        long pid_long = PyLong_AsLong(item);
        if (pid_long == -1 && PyErr_Occurred()) {
            Py_DECREF(result);
            free(clients);
            return NULL;
        }
        int pid = (int)pid_long;
        if (pid == 0) pid = my_pid;

        long long total = 0;
        if (count > 0) {
            for (int j = 0; j < count; j++) {
                if (clients[j].pid == pid)
                    total += clients[j].gpu_ns;
            }
        }

        PyObject* val = PyLong_FromLongLong(total);
        if (PyDict_SetItem(result, item, val) < 0) {
            Py_DECREF(val);
            Py_DECREF(result);
            free(clients);
            return NULL;
        }
        Py_DECREF(val);
    }

    free(clients);
    return result;
}


PyDoc_STRVAR(gpu_clients_doc,
"gpu_clients() -> list[dict]\n\n"
"Return all active GPU clients from the IORegistry.\n\n"
"Each dict has keys:\n"
"    - 'pid': int — process ID\n"
"    - 'name': str — process name (truncated)\n"
"    - 'gpu_ns': int — cumulative GPU nanoseconds\n\n"
"Multiple entries may exist for the same PID (one per command queue).\n"
"No special privileges required.");

static PyObject* py_gpu_clients(PyObject* self, PyObject* args) {
    gpu_client_t *clients;
    int count = read_gpu_clients(&clients);
    if (count < 0)
        return PyList_New(0);

    PyObject* list = PyList_New(count);
    if (!list) {
        free(clients);
        return NULL;
    }

    for (int i = 0; i < count; i++) {
        PyObject* d = PyDict_New();
        if (!d) {
            Py_DECREF(list);
            free(clients);
            return NULL;
        }
        PyObject* pid_obj = PyLong_FromLong(clients[i].pid);
        PyObject* name_obj = PyUnicode_FromString(clients[i].name);
        PyObject* ns_obj = PyLong_FromLongLong(clients[i].gpu_ns);

        PyDict_SetItemString(d, "pid", pid_obj);
        PyDict_SetItemString(d, "name", name_obj);
        PyDict_SetItemString(d, "gpu_ns", ns_obj);

        Py_DECREF(pid_obj);
        Py_DECREF(name_obj);
        Py_DECREF(ns_obj);

        PyList_SET_ITEM(list, i, d);  /* steals ref */
    }

    free(clients);
    return list;
}


/* ------------------------------------------------------------------ */
/* CPU time via proc_pid_rusage                                        */
/* ------------------------------------------------------------------ */

PyDoc_STRVAR(cpu_time_ns_doc,
"cpu_time_ns(pid) -> int\n\n"
"Return cumulative CPU time (user + system) in nanoseconds for a process.\n\n"
"Uses proc_pid_rusage (RUSAGE_INFO_V2). No special privileges needed\n"
"for processes owned by the same user.\n\n"
"Args:\n"
"    pid: Process ID.\n\n"
"Returns:\n"
"    Cumulative CPU nanoseconds, or -1 on error.");

static PyObject* py_cpu_time_ns(PyObject* self, PyObject* args) {
    int pid;
    if (!PyArg_ParseTuple(args, "i", &pid))
        return NULL;

    struct rusage_info_v6 ri;
    int ret = proc_pid_rusage(pid, RUSAGE_INFO_V6, (rusage_info_t *)&ri);
    if (ret != 0)
        return PyLong_FromLongLong(-1);

    long long total = (long long)ri.ri_user_time + (long long)ri.ri_system_time;
    return PyLong_FromLongLong(total);
}


PyDoc_STRVAR(proc_info_doc,
"proc_info(pid) -> dict | None\n\n"
"Return comprehensive process stats from rusage_info_v6 and proc_pidinfo.\n\n"
"Returns dict with keys:\n"
"    CPU:\n"
"    - 'cpu_ns': int — cumulative CPU time (user + system) in nanoseconds\n"
"    - 'cpu_user_ns' / 'cpu_system_ns': int — split CPU time\n"
"    - 'instructions': int — retired instructions\n"
"    - 'cycles': int — CPU cycles\n"
"    - 'runnable_time': int — time process was runnable (ns)\n"
"    - 'billed_system_time' / 'serviced_system_time': int — billed CPU (ns)\n"
"    Memory:\n"
"    - 'memory': int — physical footprint in bytes\n"
"    - 'real_memory': int — resident memory in bytes\n"
"    - 'wired_size': int — wired (non-pageable) memory in bytes\n"
"    - 'peak_memory': int — lifetime peak physical footprint\n"
"    - 'neural_footprint': int — Neural Engine memory in bytes\n"
"    - 'pageins': int — page-in count (memory pressure indicator)\n"
"    Disk:\n"
"    - 'disk_read_bytes' / 'disk_write_bytes': int — cumulative disk I/O\n"
"    - 'logical_writes': int — logical writes including CoW (bytes)\n"
"    Energy:\n"
"    - 'energy_nj': int — cumulative energy in nanojoules (delta for watts)\n"
"    - 'idle_wakeups': int — package idle wakeups\n"
"    - 'interrupt_wakeups': int — interrupt wakeups\n"
"    Other:\n"
"    - 'threads': int — current thread count\n\n"
"Returns None on error. No special privileges needed for same-user processes.");

static PyObject* py_proc_info(PyObject* self, PyObject* args) {
    int pid;
    if (!PyArg_ParseTuple(args, "i", &pid))
        return NULL;

    struct rusage_info_v6 ri;
    int ret = proc_pid_rusage(pid, RUSAGE_INFO_V6, (rusage_info_t *)&ri);
    if (ret != 0)
        Py_RETURN_NONE;

    /* Thread count via proc_pidinfo */
    struct proc_taskinfo pti;
    int pti_size = proc_pidinfo(pid, PROC_PIDTASKINFO, 0, &pti, sizeof(pti));

    PyObject* d = PyDict_New();
    if (!d) return NULL;

    /* Helper macro to reduce boilerplate */
    #define SET_LL(key, val) do { \
        PyObject *v = PyLong_FromLongLong((long long)(val)); \
        PyDict_SetItemString(d, key, v); Py_DECREF(v); \
    } while(0)
    #define SET_ULL(key, val) do { \
        PyObject *v = PyLong_FromUnsignedLongLong((unsigned long long)(val)); \
        PyDict_SetItemString(d, key, v); Py_DECREF(v); \
    } while(0)

    /* CPU time */
    SET_LL("cpu_ns", (long long)ri.ri_user_time + (long long)ri.ri_system_time);
    SET_LL("cpu_user_ns", ri.ri_user_time);
    SET_LL("cpu_system_ns", ri.ri_system_time);

    /* Memory */
    SET_ULL("memory", ri.ri_phys_footprint);
    SET_ULL("real_memory", ri.ri_resident_size);
    SET_ULL("wired_size", ri.ri_wired_size);
    SET_ULL("peak_memory", ri.ri_lifetime_max_phys_footprint);
    SET_ULL("neural_footprint", ri.ri_neural_footprint);
    SET_ULL("pageins", ri.ri_pageins);

    /* Disk I/O */
    SET_ULL("disk_read_bytes", ri.ri_diskio_bytesread);
    SET_ULL("disk_write_bytes", ri.ri_diskio_byteswritten);
    SET_ULL("logical_writes", ri.ri_logical_writes);

    /* Energy (nanojoules) — cumulative, take deltas for power rate */
    SET_ULL("energy_nj", ri.ri_energy_nj);

    /* CPU perf counters */
    SET_ULL("instructions", ri.ri_instructions);
    SET_ULL("cycles", ri.ri_cycles);
    SET_ULL("runnable_time", ri.ri_runnable_time);
    SET_LL("billed_system_time", ri.ri_billed_system_time);
    SET_LL("serviced_system_time", ri.ri_serviced_system_time);

    /* Wakeups (energy efficiency) */
    SET_ULL("idle_wakeups", ri.ri_pkg_idle_wkups);
    SET_ULL("interrupt_wakeups", ri.ri_interrupt_wkups);

    /* Thread count */
    if (pti_size >= (int)sizeof(pti)) {
        SET_LL("threads", pti.pti_threadnum);
    } else {
        SET_LL("threads", 0);
    }

    #undef SET_LL
    #undef SET_ULL

    return d;
}


/* ------------------------------------------------------------------ */
/* System-wide GPU stats from IORegistry PerformanceStatistics         */
/* ------------------------------------------------------------------ */

PyDoc_STRVAR(system_gpu_stats_doc,
"system_gpu_stats() -> dict\n\n"
"Return system-wide GPU performance statistics from the IORegistry.\n\n"
"Reads 'PerformanceStatistics' from the AGXAccelerator entry.\n\n"
"Returns dict with keys:\n"
"    - 'device_utilization': int — Device Utilization %% (0-100)\n"
"    - 'tiler_utilization': int — Tiler Utilization %%\n"
"    - 'renderer_utilization': int — Renderer Utilization %%\n"
"    - 'alloc_system_memory': int — total GPU-allocated system memory (bytes)\n"
"    - 'in_use_system_memory': int — in-use GPU system memory (bytes)\n"
"    - 'in_use_system_memory_driver': int — driver-side in-use memory\n"
"    - 'allocated_pb_size': int — parameter buffer allocation (bytes)\n"
"    - 'recovery_count': int — GPU recovery (crash) count\n"
"    - 'last_recovery_time': int — timestamp of last GPU recovery\n"
"    - 'split_scene_count': int — tiler split scene events\n"
"    - 'tiled_scene_bytes': int — current tiled scene buffer size\n"
"    - 'model': str — GPU model name\n"
"    - 'gpu_core_count': int — number of GPU cores");

static PyObject* py_system_gpu_stats(PyObject* self, PyObject* args) {
    io_iterator_t iter;
    kern_return_t kr = IOServiceGetMatchingServices(
        kIOMainPortDefault,
        IOServiceMatching("AGXAccelerator"),
        &iter);
    if (kr != KERN_SUCCESS)
        return PyDict_New();

    PyObject* result = PyDict_New();
    if (!result) {
        IOObjectRelease(iter);
        return NULL;
    }

    io_service_t accel;
    if ((accel = IOIteratorNext(iter)) != 0) {
        CFDictionaryRef perf = IORegistryEntryCreateCFProperty(
            accel, CFSTR("PerformanceStatistics"),
            kCFAllocatorDefault, 0);
        if (perf && CFGetTypeID(perf) == CFDictionaryGetTypeID()) {
            /* Extract known keys */
            struct { const char *cf_key; const char *py_key; } keys[] = {
                {"Device Utilization %", "device_utilization"},
                {"Tiler Utilization %", "tiler_utilization"},
                {"Renderer Utilization %", "renderer_utilization"},
                {"Alloc system memory", "alloc_system_memory"},
                {"In use system memory", "in_use_system_memory"},
                {"In use system memory (driver)", "in_use_system_memory_driver"},
                {"Allocated PB Size", "allocated_pb_size"},
                {"recoveryCount", "recovery_count"},
                {"lastRecoveryTime", "last_recovery_time"},
                {"SplitSceneCount", "split_scene_count"},
                {"TiledSceneBytes", "tiled_scene_bytes"},
                {NULL, NULL}
            };
            for (int i = 0; keys[i].cf_key; i++) {
                CFStringRef cf_key = CFStringCreateWithCString(
                    kCFAllocatorDefault, keys[i].cf_key, kCFStringEncodingUTF8);
                CFNumberRef num = CFDictionaryGetValue(perf, cf_key);
                CFRelease(cf_key);
                if (num && CFGetTypeID(num) == CFNumberGetTypeID()) {
                    long long val = 0;
                    CFNumberGetValue(num, kCFNumberSInt64Type, &val);
                    PyObject *v = PyLong_FromLongLong(val);
                    PyDict_SetItemString(result, keys[i].py_key, v);
                    Py_DECREF(v);
                }
            }
        }
        if (perf) CFRelease(perf);

        /* Also grab model name and core count */
        CFStringRef model = IORegistryEntryCreateCFProperty(
            accel, CFSTR("model"), kCFAllocatorDefault, 0);
        if (model && CFGetTypeID(model) == CFStringGetTypeID()) {
            char buf[128];
            if (CFStringGetCString(model, buf, sizeof(buf), kCFStringEncodingUTF8)) {
                PyObject *v = PyUnicode_FromString(buf);
                PyDict_SetItemString(result, "model", v);
                Py_DECREF(v);
            }
        }
        if (model) CFRelease(model);

        CFNumberRef cores = IORegistryEntryCreateCFProperty(
            accel, CFSTR("gpu-core-count"), kCFAllocatorDefault, 0);
        if (cores && CFGetTypeID(cores) == CFNumberGetTypeID()) {
            long long val = 0;
            CFNumberGetValue(cores, kCFNumberSInt64Type, &val);
            PyObject *v = PyLong_FromLongLong(val);
            PyDict_SetItemString(result, "gpu_core_count", v);
            Py_DECREF(v);
        }
        if (cores) CFRelease(cores);

        IOObjectRelease(accel);
    }

    IOObjectRelease(iter);
    return result;
}


/* ------------------------------------------------------------------ */
/* Parent PID lookup via proc_pidinfo                                  */
/* ------------------------------------------------------------------ */

PyDoc_STRVAR(ppid_doc,
"ppid(pid) -> int\n\n"
"Return the parent process ID for the given PID.\n\n"
"Uses proc_pidinfo(PROC_PIDTBSDINFO). Returns -1 on error.");

static PyObject* py_ppid(PyObject* self, PyObject* args) {
    int pid;
    if (!PyArg_ParseTuple(args, "i", &pid))
        return NULL;

    struct proc_bsdinfo bsdinfo;
    int ret = proc_pidinfo(pid, PROC_PIDTBSDINFO, 0, &bsdinfo, sizeof(bsdinfo));
    if (ret <= 0)
        return PyLong_FromLong(-1);

    return PyLong_FromLong(bsdinfo.pbi_ppid);
}


/* ------------------------------------------------------------------ */
/* Module definition                                                   */
/* ------------------------------------------------------------------ */

static PyMethodDef methods[] = {
    {"gpu_time_ns",       py_gpu_time_ns,       METH_VARARGS, gpu_time_ns_doc},
    {"gpu_time_ns_multi", py_gpu_time_ns_multi, METH_VARARGS, gpu_time_ns_multi_doc},
    {"gpu_clients",       py_gpu_clients,       METH_NOARGS,  gpu_clients_doc},
    {"cpu_time_ns",       py_cpu_time_ns,       METH_VARARGS, cpu_time_ns_doc},
    {"proc_info",         py_proc_info,         METH_VARARGS, proc_info_doc},
    {"system_gpu_stats",  py_system_gpu_stats,  METH_NOARGS,  system_gpu_stats_doc},
    {"ppid",              py_ppid,              METH_VARARGS, ppid_doc},
    {NULL, NULL, 0, NULL}
};

static struct PyModuleDef module_def = {
    PyModuleDef_HEAD_INIT,
    "_native",
    "Per-process GPU time via IORegistry AGXDeviceUserClient on macOS Apple Silicon.",
    -1,
    methods
};

PyMODINIT_FUNC PyInit__native(void) {
    return PyModule_Create(&module_def);
}
