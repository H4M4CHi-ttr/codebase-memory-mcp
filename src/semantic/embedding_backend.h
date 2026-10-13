#ifndef CBM_EMBEDDING_BACKEND_H
#define CBM_EMBEDDING_BACKEND_H
#include "store/store.h"

#define CBM_GEMMA2_PROFILE_ID \
    "embeddinggemma-2:914f7f89142e33e77833254d9c9b90c3cef7303b:768:fp32:code-retrieval-v1"

typedef struct cbm_gemma2_worker cbm_gemma2_worker_t;

/* One owned model process per server, started lazily or explicitly at startup. */
int cbm_gemma2_preload(cbm_gemma2_worker_t **slot);
void cbm_gemma2_free(cbm_gemma2_worker_t *worker);

/* Opt-in local worker. It may fail; callers must never fall back to another model. */
int cbm_gemma2_search(cbm_gemma2_worker_t **slot, cbm_store_t *store, const char *project,
                      const char **queries, int query_count, int limit, cbm_vector_result_t **out,
                      int *count);

/* Validate a worker response against the active store before materializing rows. */
int cbm_gemma2_decode(cbm_store_t *store, const char *project, const char *generation,
                      const char *json, size_t length, int limit, cbm_vector_result_t **out,
                      int *count);
#endif
