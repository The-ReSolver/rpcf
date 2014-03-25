#ifndef __dbg_h__
#define __dbg_h__

#include <stdio.h>
#include <errno.h>
#include <string.h>

#ifdef DEBUG
#define debug(Mess, ...) fprintf(stderr, "DEBUG (%s:%d) " Mess "\n", __FILE__, __LINE__, ##__VA_ARGS__)
#else 
#define debug(Mess, ...)
#endif

#define clean_errno() (errno == 0 ? "None" : strerror(errno))

#define log_err(Mess, ... ) fprintf(stderr, "[ERROR] (%s:%d in %s - errno: %s) " Mess "\n", __FILE__, \
                         __LINE__, __FUNCTION__, clean_errno(), ##__VA_ARGS__)
#define log_warn(Mess, ... ) fprintf(stderr, "[WARN] (%s:%d: errno: %s) " Mess "\n", __FILE__, __LINE__, clean_errno(), ##__VA_ARGS__)
#define log_info(Mess, ... ) fprintf(stderr, "[INFO] : " Mess "\n", ##__VA_ARGS__)

#define check(A, Mess, ...) if(!(A)) { log_err(Mess, ##__VA_ARGS__); errno=0; goto error; }
#define check_mem(A) check((A), "Out of memory.")

#define sentinel(Mess, ...)  { log_err(Mess, ##__VA_ARGS__); errno=0; goto error; }
#define check_debug(A, Mess, ...) if(!(A)) { debug(Mess, ##__VA_ARGS__); errno=0; goto error; }

#endif