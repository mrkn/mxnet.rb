#ifndef MXNET_INTERNAL_H
#define MXNET_INTERNAL_H 1

#ifdef __cplusplus
extern "C" {
#if 0
} /* satisfy cc-mode */
#endif
#endif

#include <ruby.h>

#if SIZEOF_LONG == SIZEOF_VOIDP
# define PTR2NUM(x)   (LONG2NUM((long)(x)))
# define NUM2PTR(x)   ((void*)(NUM2ULONG(x)))
#elif SIZEOF_LONG_LONG == SIZEOF_VOIDP
# define PTR2NUM(x)   (LL2NUM((LONG_LONG)(x)))
# define NUM2PTR(x)   ((void*)(NUM2ULL(x)))
#else
# error ---->> ruby requires sizeof(void*) == sizeof(long) or sizeof(LONG_LONG) to be compiled. <<----
#endif

typedef unsigned int mx_uint;
typedef float mx_float;
typedef void *NDArrayHandle;

#define NUM2MXUINT(num) NUM2UINT(num)
#define MXUINT2NUM(val) UINT2NUM(val)

enum TypeFlag {
  kFloat32 = 0,
  kFloat64 = 1,
  kFloat16 = 2,
  kUint8 = 3,
  kInt32 = 4,
  kInt8  = 5,
  kInt64 = 6,
};

struct mxnet_api_table {
  const char * (* MXGetLastError)();
  int (* MXNDArrayCreateEx)(const mx_uint *shape, mx_uint ndim,
                            int dev_type, int dev_id, int delay_alloc,
                            int dtype, NDArrayHandle *out);
  int (* MXNDArrayGetShape)(NDArrayHandle handle, mx_uint *out_dim,
                            const mx_uint **out_pdata);

  int (* MXListAllOpNames)(mx_uint *out_size, const char ***out_array);
  int (* NNGetOpHandle)(char const *name, void **p_handle);
  int (* MXSymbolGetAtomicSymbolInfo)(
      void *op_handle, const char **name, const char **description,
      mx_uint *num_args, const char ***arg_names, const char ***arg_type_infos,
      const char ***arg_descriptions, const char **key_var_num_args,
      const char **return_type);
  int (* MXImperativeInvoke)(
      void *op_handle,
      int num_inputs,
      /* NDArrayHandle */ void **inputs,
      int *num_outputs,
      /* NDArrayHandle */ void ***outputs,
      int num_params,
      const char **param_keys,
      const char **param_vals);
};

struct mxnet_api_table *mxnet_get_api_table(void);
#define MXNET_API(name) (mxnet_get_api_table()->name)

int mxnet_context_get_device_type_id(VALUE ctx);
int mxnet_context_get_device_id(VALUE ctx);

void *mxnet_ndarray_get_handle(VALUE ndary);

void mxnet_init_libmxnet(void);
void mxnet_init_ndarray(void);
void mxnet_init_operations(void);
NORETURN(void mxnet_raise_last_error(void));
#define CHECK_CALL(expr) if ((expr) != 0) mxnet_raise_last_error()

extern VALUE mxnet_mMXNet;
extern VALUE mxnet_mUtils;
extern VALUE mxnet_cContext;
extern VALUE mxnet_cNDArray;
extern VALUE mxnet_mNDArrayOps;

extern VALUE mxnet_sOpInfo;
extern VALUE mxnet_sOpArgInfo;

extern VALUE mxnet_eError;

#ifdef __cplusplus
#if 0
{ /* satisfy cc-mode */
#endif
} /* extern "C" { */
#endif

#endif /* MXNET_INTERNAL_H */
