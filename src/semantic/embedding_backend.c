#include "semantic/embedding_backend.h"
#include "foundation/compat.h"
#include "foundation/compat_fs.h"
#include "foundation/constants.h"
#include "foundation/log.h"
#include "foundation/platform.h"
#include "foundation/subprocess.h"
#include "yyjson/yyjson.h"
#include "pipeline/pipeline.h"

#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#ifdef _WIN32
#include <process.h>
#define getpid _getpid
#else
#include <unistd.h>
#endif

enum {
    GEMMA_MAX_ROWS = 100500,
    GEMMA_RESPONSE_BYTES = 8 * 1024 * 1024,
    GEMMA_REQUEST_BYTES = 256 * 1024,
    GEMMA_DEADLINE_MS = 60000,
    GEMMA_MAX_QUERIES = 32,
    GEMMA_FIRST_ROW = 1,
    GEMMA_POLL_US = 10000,
};

#define GEMMA_MAX_ABS_SCORE 1.00001

static int compare_id(const void *a, const void *b) {
    int64_t x = *(const int64_t *)a;
    int64_t y = *(const int64_t *)b;
    return (x > y) - (x < y);
}

static bool decode_row(cbm_store_t *store, const char *project, yyjson_val *row,
                       cbm_vector_result_t *result) {
    yyjson_val *id = yyjson_obj_get(row, "node_id");
    yyjson_val *score = yyjson_obj_get(row, "score");
    cbm_node_t node = {0};
    double value = yyjson_get_num(score);
    bool ok = yyjson_is_int(id) && yyjson_get_sint(id) > 0 && yyjson_is_num(score) &&
              isfinite(value) && fabs(value) <= GEMMA_MAX_ABS_SCORE &&
              cbm_store_find_node_by_id(store, yyjson_get_sint(id), &node) == CBM_STORE_OK &&
              node.project && strcmp(node.project, project) == 0 && node.label && node.name &&
              node.qualified_name && node.file_path &&
              (strcmp(node.label, "Function") == 0 || strcmp(node.label, "Method") == 0 ||
               strcmp(node.label, "Class") == 0);
    if (ok) {
        *result = (cbm_vector_result_t){
            .node_id = node.id,
            .score = value,
            .name = strdup(node.name),
            .qualified_name = strdup(node.qualified_name),
            .file_path = strdup(node.file_path),
            .label = strdup(node.label),
        };
        ok = result->name && result->qualified_name && result->file_path && result->label;
    }
    cbm_node_free_fields(&node);
    return ok;
}

int cbm_gemma2_decode(cbm_store_t *store, const char *project, const char *generation,
                      const char *json, size_t length, int limit, cbm_vector_result_t **out,
                      int *count) {
    *out = NULL;
    *count = 0;
    if (!store || !project || !generation || !json || !length || length > GEMMA_RESPONSE_BYTES ||
        limit < GEMMA_FIRST_ROW || limit > GEMMA_MAX_ROWS) {
        return CBM_STORE_ERR;
    }
    yyjson_doc *doc = yyjson_read(json, length, 0);
    yyjson_val *root = doc ? yyjson_doc_get_root(doc) : NULL;
    const char *profile = yyjson_get_str(yyjson_obj_get(root, "profile"));
    const char *reply_generation = yyjson_get_str(yyjson_obj_get(root, "generation"));
    yyjson_val *rows = yyjson_obj_get(root, "results");
    char active[CBM_SZ_64];
    bool ok = profile && strcmp(profile, CBM_GEMMA2_PROFILE_ID) == 0 && reply_generation &&
              strcmp(reply_generation, generation) == 0 &&
              cbm_store_generation(store, active, sizeof(active)) == CBM_STORE_OK &&
              strcmp(active, generation) == 0 && yyjson_is_arr(rows) &&
              yyjson_arr_size(rows) <= (size_t)limit;
    int n = ok ? (int)yyjson_arr_size(rows) : 0;
    cbm_vector_result_t *results = n ? calloc((size_t)n, sizeof(*results)) : NULL;
    int64_t *ids = n ? calloc((size_t)n, sizeof(*ids)) : NULL;
    if (n && (!results || !ids)) {
        ok = false;
    }
    for (int i = 0; ok && i < n; i++) {
        ok = decode_row(store, project, yyjson_arr_get(rows, (size_t)i), &results[i]);
        if (!ok) {
            break;
        }
        ids[i] = results[i].node_id;
        if (i > 0) {
            ok = results[i - GEMMA_FIRST_ROW].score > results[i].score ||
                 (results[i - GEMMA_FIRST_ROW].score == results[i].score &&
                  ids[i - GEMMA_FIRST_ROW] < ids[i]);
        }
    }
    if (ok && n > GEMMA_FIRST_ROW) {
        qsort(ids, (size_t)n, sizeof(*ids), compare_id);
        for (int i = GEMMA_FIRST_ROW; ok && i < n; i++) {
            ok = ids[i] != ids[i - GEMMA_FIRST_ROW];
        }
    }
    free(ids);
    if (doc) {
        yyjson_doc_free(doc);
    }
    if (!ok) {
        cbm_store_free_vector_results(results, n);
        return CBM_STORE_ERR;
    }
    *out = results;
    *count = n;
    return CBM_STORE_OK;
}

struct cbm_gemma2_worker {
    cbm_subprocess_t *process;
    char directory[CBM_PATH_MAX];
    char log[CBM_PATH_MAX];
    int64_t request_id;
    bool failed;
};

static void worker_path(const cbm_gemma2_worker_t *worker, const char *name,
                        char path[CBM_PATH_MAX]) {
    (void)snprintf(path, CBM_PATH_MAX, "%s/%s", worker->directory, name);
}

static bool worker_alive(cbm_gemma2_worker_t *worker) {
    if (cbm_subprocess_poll(worker->process, NULL) == CBM_PROC_POLL_RUNNING) {
        return true;
    }
    worker->failed = true;
    return false;
}

void cbm_gemma2_free(cbm_gemma2_worker_t *worker) {
    if (!worker) {
        return;
    }
    bool quiesced = true;
    if (worker->process) {
        cbm_subprocess_request_cancel(worker->process);
        cbm_proc_result_t result = {0};
        while (cbm_subprocess_poll(worker->process, &result) == CBM_PROC_POLL_RUNNING) {
            cbm_usleep(GEMMA_POLL_US);
        }
        quiesced = result.tree_quiesced && !result.supervision_failed;
        if (!quiesced) {
            cbm_log_error("gemma2.teardown_failed", "path", worker->log);
        }
        cbm_subprocess_destroy(worker->process);
    }
    if (quiesced) {
        const char *files[] = {
            "request.json",     "request.pending", "response.json",
            "response.pending", "ready.json",      "ready.pending",
        };
        for (size_t i = 0; i < sizeof(files) / sizeof(files[0]); i++) {
            char path[CBM_PATH_MAX];
            worker_path(worker, files[i], path);
            cbm_unlink(path);
        }
        if (!worker->failed) {
            cbm_unlink(worker->log);
            cbm_rmdir(worker->directory);
        }
    }
    if (worker->failed) {
        cbm_log_warn("gemma2.worker_log", "path", worker->log);
    }
    free(worker);
}

static yyjson_doc *worker_wait(cbm_gemma2_worker_t *worker, const char *name, size_t cap) {
    char path[CBM_PATH_MAX];
    worker_path(worker, name, path);
    uint64_t deadline = cbm_now_ms() + GEMMA_DEADLINE_MS;
    while (worker_alive(worker)) {
        cbm_path_info_t info;
        if (cbm_path_info_utf8(path, &info) == CBM_PATH_INFO_OK) {
            if (!info.is_regular || info.is_symlink || !info.size || info.size > cap) {
                break;
            }
            char *data = malloc((size_t)info.size);
            FILE *file = data ? cbm_fopen(path, "rb") : NULL;
            bool ok =
                file && fread(data, sizeof(*data), (size_t)info.size, file) == (size_t)info.size;
            if (file && fclose(file) != 0) {
                ok = false;
            }
            yyjson_doc *doc = ok ? yyjson_read(data, (size_t)info.size, 0) : NULL;
            free(data);
            cbm_unlink(path);
            if (doc) {
                return doc;
            }
            break;
        }
        if (cbm_now_ms() >= deadline) {
            break;
        }
        cbm_usleep(GEMMA_POLL_US);
    }
    worker->failed = true;
    return NULL;
}

int cbm_gemma2_preload(cbm_gemma2_worker_t **slot) {
    if (!slot) {
        return CBM_STORE_ERR;
    }
    if (*slot && worker_alive(*slot)) {
        return CBM_STORE_OK;
    }
    cbm_gemma2_free(*slot);
    *slot = NULL;
    char python[CBM_PATH_MAX];
    char script[CBM_PATH_MAX];
    if (!cbm_safe_getenv("CBM_GEMMA2_PYTHON", python, sizeof(python), NULL) || !python[0] ||
        !cbm_safe_getenv("CBM_GEMMA2_SCRIPT", script, sizeof(script), NULL) || !script[0]) {
        cbm_log_warn("gemma2.start_refused", "reason", "missing_worker_configuration");
        return CBM_STORE_ERR;
    }
    cbm_gemma2_worker_t *worker = calloc(CBM_ALLOC_ONE, sizeof(*worker));
    if (!worker) {
        return CBM_STORE_ERR;
    }
    int written = snprintf(worker->directory, sizeof(worker->directory), "%s/cbm-gemma2-XXXXXX",
                           cbm_tmpdir());
    if (written < 0 || (size_t)written >= sizeof(worker->directory) - CBM_SZ_32 ||
        !cbm_mkdtemp(worker->directory)) {
        free(worker);
        return CBM_STORE_ERR;
    }
    worker_path(worker, "worker.log", worker->log);
    char parent[CBM_SZ_32];
    (void)snprintf(parent, sizeof(parent), "%d", getpid());
    const char *argv[] = {
        python, script, "serve", "--directory", worker->directory, "--owner-pid", parent, NULL,
    };
    const char *env[] = {"PYTHONUTF8=1", "PYTHONIOENCODING=utf-8", NULL};
    cbm_proc_opts_t opts = {
        .bin = python,
        .argv = argv,
        .env_overrides = env,
        .log_file = worker->log,
    };
    bool ok = cbm_subprocess_spawn(&opts, &worker->process) == 0;
    yyjson_doc *ready = ok ? worker_wait(worker, "ready.json", CBM_SZ_4K) : NULL;
    yyjson_val *root = ready ? yyjson_doc_get_root(ready) : NULL;
    const char *profile = yyjson_get_str(yyjson_obj_get(root, "profile"));
    const char *status = yyjson_get_str(yyjson_obj_get(root, "status"));
    ok = profile && strcmp(profile, CBM_GEMMA2_PROFILE_ID) == 0 && status &&
         strcmp(status, "ready") == 0;
    if (ready) {
        yyjson_doc_free(ready);
    }
    if (!ok) {
        worker->failed = true;
        cbm_gemma2_free(worker);
        return CBM_STORE_ERR;
    }
    cbm_log_info("gemma2.worker_ready", "mode", "persistent", "path", worker->directory);
    *slot = worker;
    return CBM_STORE_OK;
}

static bool worker_publish(cbm_gemma2_worker_t *worker, const char *data, size_t size) {
    char pending[CBM_PATH_MAX];
    char input[CBM_PATH_MAX];
    worker_path(worker, "request.pending", pending);
    worker_path(worker, "request.json", input);
    FILE *file = cbm_fopen(pending, "wb");
    bool ok = file && fwrite(data, sizeof(*data), size, file) == size;
    if (file && fclose(file) != 0) {
        ok = false;
    }
    return ok && cbm_rename_replace(pending, input) == 0;
}

static bool source_current(cbm_store_t *store, const char *project) {
    enum { GEMMA_MAX_FILES = 200000, GEMMA_SOURCE_BYTES = 256 * 1024 * 1024 };
    cbm_index_resource_policy_t policy = {
        .max_files = {.enabled = true, .value = GEMMA_MAX_FILES},
        .max_source_bytes = {.enabled = true, .value = GEMMA_SOURCE_BYTES},
    };
    if (cbm_pipeline_inputs_match_store(store, project, &policy)) {
        return true;
    }
    cbm_log_warn("gemma2.source_stale", "reason",
                 "native_index_pending_or_incomplete_or_input_changed");
    return false;
}

static bool worker_request(cbm_gemma2_worker_t *worker, const char *db_path, const char *project,
                           const char *generation, const char **queries, int query_count,
                           int limit) {
    yyjson_mut_doc *doc = yyjson_mut_doc_new(NULL);
    yyjson_mut_val *root = doc ? yyjson_mut_obj(doc) : NULL;
    bool ok = root != NULL && worker->request_id < INT64_MAX;
    if (ok) {
        yyjson_mut_doc_set_root(doc, root);
        ok = yyjson_mut_obj_add_strcpy(doc, root, "db_path", db_path) &&
             yyjson_mut_obj_add_strcpy(doc, root, "project", project) &&
             yyjson_mut_obj_add_strcpy(doc, root, "generation", generation) &&
             yyjson_mut_obj_add_int(doc, root, "limit", limit) &&
             yyjson_mut_obj_add_int(doc, root, "request_id", ++worker->request_id);
        yyjson_mut_val *array = yyjson_mut_arr(doc);
        ok = ok && array && yyjson_mut_obj_add_val(doc, root, "queries", array);
        for (int i = 0; ok && i < query_count; i++) {
            ok = queries[i] && strlen(queries[i]) <= CBM_SZ_4K &&
                 yyjson_mut_arr_add_strcpy(doc, array, queries[i]);
        }
    }
    size_t request_len = 0;
    char *request = ok ? yyjson_mut_write(doc, 0, &request_len) : NULL;
    ok = request && request_len <= GEMMA_REQUEST_BYTES &&
         worker_publish(worker, request, request_len);
    free(request);
    if (doc) {
        yyjson_mut_doc_free(doc);
    }
    return ok;
}

int cbm_gemma2_search(cbm_gemma2_worker_t **slot, cbm_store_t *store, const char *project,
                      const char **queries, int query_count, int limit, cbm_vector_result_t **out,
                      int *count) {
    *out = NULL;
    *count = 0;
    char generation[CBM_SZ_64];
    const char *db_path = store ? cbm_store_db_path(store) : NULL;
    if (!db_path || !project || !queries || query_count < GEMMA_FIRST_ROW ||
        query_count > GEMMA_MAX_QUERIES || limit < GEMMA_FIRST_ROW || limit > GEMMA_MAX_ROWS ||
        cbm_store_generation(store, generation, sizeof(generation)) != CBM_STORE_OK ||
        strcmp(generation, "legacy") == 0) {
        return CBM_STORE_ERR;
    }
    if (!source_current(store, project)) {
        return CBM_STORE_ERR;
    }
    if (cbm_gemma2_preload(slot) != CBM_STORE_OK) {
        return CBM_STORE_ERR;
    }
    cbm_gemma2_worker_t *worker = *slot;
    bool ok = worker_request(worker, db_path, project, generation, queries, query_count, limit);
    yyjson_doc *reply = ok ? worker_wait(worker, "response.json", GEMMA_RESPONSE_BYTES) : NULL;
    yyjson_val *response = reply ? yyjson_doc_get_root(reply) : NULL;
    yyjson_val *id = yyjson_obj_get(response, "request_id");
    ok = yyjson_is_int(id) && yyjson_get_sint(id) == worker->request_id;
    int rc = CBM_STORE_ERR;
    const char *error = ok ? yyjson_get_str(yyjson_obj_get(response, "error")) : NULL;
    if (error) {
        worker->failed = true; /* retain diagnostics even if the model remains usable */
        cbm_log_warn("gemma2.request_failed", "reason", error, "path", worker->log);
    } else if (ok) {
        size_t length = 0;
        char *json = yyjson_write(reply, 0, &length);
        if (json) {
            rc = cbm_gemma2_decode(store, project, generation, json, length, limit, out, count);
        }
        free(json);
    }
    bool request_error = error != NULL;
    if (reply) {
        yyjson_doc_free(reply);
    }
    if (!ok || (!request_error && rc != CBM_STORE_OK)) {
        worker->failed = true;
        cbm_gemma2_free(worker);
        *slot = NULL;
    }
    if (rc == CBM_STORE_OK && !source_current(store, project)) {
        cbm_store_free_vector_results(*out, *count);
        *out = NULL;
        *count = 0;
        return CBM_STORE_ERR;
    }
    return rc;
}
