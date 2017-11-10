#ifndef MXNET_INTERNAL_H
#define MXNET_INTERNAL_H 1

#ifdef __cplusplus
extern "C" {
#if 0
} /* satisfy cc-mode */
#endif
#endif

#include <ruby.h>

#ifdef HAVE_STDINT_H
# include <stdint.h>
#endif

#ifndef HAVE_INT8_T
typedef char int8_t;
typedef unsigned char uint8_t;
#endif

#ifndef HAVE_INT16_T
# if SIZEOF_SHORT == 2
typedef short int16_t;
typedef unsigned short uint16_t;
# elif SIZEOF_INT == 2
typedef int int16_t;
typedef unsigned int uint16_t;
# endif
#endif

#ifndef HAVE_INT32_T
# if SIZEOF_INT == 4
typedef int int32_t;
typedef unsigned int uint32_t;
# elif SIZEOF_LONG == 4
typedef long int32_t;
typedef unsigned long uint32_t;
# elif SIZEOF_SHORT == 4
typedef short int32_t;
typedef unsigned short uint32_t;
# endif
#endif

#ifndef HAVE_INT64_T
# if SIZEOF_LONG == 8
typedef long int64_t;
typedef unsigned long uint64_t;
# elif defined(HAVE_LONG_LONG) && SIZEOF_LONG_LONG == 8
typedef LONG_LONG int64_t;
typedef unsigned LONG_LONG uint64_t;
# endif
#endif

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

enum DTypeID {
  kFloat32 = 0,
  kFloat64 = 1,
  kFloat16 = 2,
  kUint8 = 3,
  kInt32 = 4,
  kInt8  = 5,
  kInt64 = 6,
  NUMBER_OF_DTYPE_IDS
};

struct mxnet_api_table {
  const char * (* MXGetLastError)();
  int (* MXNDArrayCreateEx)(const mx_uint *shape, mx_uint ndim,
                            int dev_type, int dev_id, int delay_alloc,
                            int dtype, NDArrayHandle *out);
  int (* MXNDArrayReshape)(NDArrayHandle handle, int ndim, int *dims,
                           NDArrayHandle *out);
  int (* MXNDArrayGetShape)(NDArrayHandle handle, mx_uint *out_dim,
                            const mx_uint **out_pdata);
  int (* MXNDArrayGetDType)(NDArrayHandle handle, int *out_dtype);
  int (* MXNDArraySyncCopyToCPU)(NDArrayHandle handle, void *data, size_t size);
  int (* MXNDArrayAt)(NDArrayHandle handle, mx_uint idx, NDArrayHandle *out);

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
      NDArrayHandle *inputs,
      int *num_outputs,
      NDArrayHandle **outputs,
      int num_params,
      const char **param_keys,
      const char **param_vals);
};

struct mxnet_api_table *mxnet_get_api_table(void);
#define MXNET_API(name) (mxnet_get_api_table()->name)

int mxnet_context_get_device_type_id(VALUE ctx);
int mxnet_context_get_device_id(VALUE ctx);

VALUE mxnet_dtype_id2name(int dtype_id);
int mxnet_dtype_name2id(VALUE dtype_name);
VALUE mxnet_dtype_name(VALUE id_or_name);

VALUE mxnet_ndarray_new(void *ndarray_handle);
void *mxnet_ndarray_get_handle(VALUE ndary);

void mxnet_init_libmxnet(void);
void mxnet_init_ndarray(void);
void mxnet_init_operations(VALUE klass);
NORETURN(void mxnet_raise_last_error(void));
#define CHECK_CALL(expr) if ((expr) != 0) mxnet_raise_last_error()

extern VALUE mxnet_mMXNet;
extern VALUE mxnet_mUtils;
extern VALUE mxnet_cContext;
extern VALUE mxnet_cNDArray;

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
