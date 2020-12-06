#ifndef MXNET_INTERNAL_H
#define MXNET_INTERNAL_H 1

#ifdef __cplusplus
extern "C" {
#if 0
} /* satisfy cc-mode */
#endif
#endif

#include <ruby.h>

/* Defined only in ruby 2.4.0+. Redefine here for Ruby 2.x backward compatibility */
#ifndef RB_INTEGER_TYPE_P
  # define RB_INTEGER_TYPE_P(c) (FIXNUM_P(c) || RB_TYPE_P(c, T_BIGNUM))
#endif

#include <assert.h>
#ifdef HAVE_STDINT_H
# include <stdint.h>
#endif
/* We assume that <stdbool.h> is available as MXNet uses it. */
#include <stdbool.h>

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
typedef unsigned int nn_uint;
typedef float mx_float;
typedef void *CachedOpHandle;
typedef void *ExecutorHandle;
typedef void *DataIterCreator;
typedef void *DataIterHandle;
typedef void *NDArrayHandle;
typedef void *SymbolHandle;

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

enum StorageTypeID {
  // dense
  kDefaultStorage = 0,
  // row sparse
  kRowSparseStorage = 1,
  // csr
  kCSRStorage = 2,
  NUMBER_OF_STORAGE_TYPE_IDS
};


struct mxnet_api_table {
  const char * (* MXGetLastError)();

  int (* MXRandomSeed)(int seed);

  int (* MXExecutorOutputs)(ExecutorHandle handle,
                            mx_uint *out_size,
                            NDArrayHandle **out);
  int (* MXExecutorForward)(ExecutorHandle handle, int is_train);
  int (* MXExecutorBackwardEx)(ExecutorHandle handle,
                               mx_uint len,
                               void **head_grads,
                               int is_train);
  int (* MXExecutorBindEX)(SymbolHandle symbol_handle,
                           int dev_type,
                           int dev_id,
                           mx_uint num_map_keys,
                           const char** map_keys,
                           const int* map_dev_types,
                           const int* map_dev_ids,
                           mx_uint len,
                           NDArrayHandle *in_args,
                           NDArrayHandle *arg_grad_store,
                           mx_uint *grad_req_type,
                           mx_uint aux_states_len,
                           NDArrayHandle *aux_states,
                           ExecutorHandle shared_exec,
                           ExecutorHandle *out);

  int (* MXNDArrayCreateEx)(const mx_uint *shape, mx_uint ndim,
                            int dev_type, int dev_id, int delay_alloc,
                            int dtype, NDArrayHandle *out);
  int (* MXNDArrayFree)(NDArrayHandle handle);
  int (* MXNDArraySave)(const char *fname, mx_uint num_args,
                        NDArrayHandle *args, const char **keys);
  int (* MXNDArrayLoad)(const char *fname,
                        mx_uint *out_size, NDArrayHandle **out_arr,
                        mx_uint *out_name_size, const char ***out_names);
  int (* MXNDArrayLoadFromBuffer)(const void *ndarray_buffer, size_t size,
                                  mx_uint *out_size, NDArrayHandle **out_arr,
                                  mx_uint *out_name_size, const char ***out_names);
  int (* MXNDArrayReshape)(NDArrayHandle handle, int ndim, int *dims,
                           NDArrayHandle *out);
  int (* MXNDArrayGetContext)(NDArrayHandle handle, int *out_dev_type, int *out_dev_id);
  int (* MXNDArrayGetShape)(NDArrayHandle handle, mx_uint *out_dim,
                            const mx_uint **out_pdata);
  int (* MXNDArrayGetDType)(NDArrayHandle handle, int *out_dtype);
  int (* MXNDArraySyncCopyFromCPU)(NDArrayHandle handle, const void *data, size_t size);
  int (* MXNDArraySyncCopyToCPU)(NDArrayHandle handle, void *data, size_t size);
  int (* MXNDArrayAt)(NDArrayHandle handle, mx_uint idx, NDArrayHandle *out);
  int (* MXNDArraySlice)(NDArrayHandle handle, mx_uint start, mx_uint stop, NDArrayHandle *out);
  int (* MXNDArrayGetGrad)(NDArrayHandle handle, NDArrayHandle *out);
  int (* MXNDArrayWaitToRead)(NDArrayHandle handle);

  int (* MXAutogradSetIsRecording)(int is_recording, int* prev);
  int (* MXAutogradSetIsTraining)(int is_training, int* prev);
  int (* MXAutogradIsRecording)(bool* curr);
  int (* MXAutogradIsTraining)(bool* curr);
  int (* MXAutogradMarkVariables)(mx_uint num_var,
                                  NDArrayHandle *var_handles,
                                  mx_uint *reqs_array,
                                  NDArrayHandle *grad_handles);
  int (* MXAutogradBackwardEx)(mx_uint num_output,
                               NDArrayHandle *output_handles,
                               NDArrayHandle *ograd_handles,
                               mx_uint num_variables,
                               NDArrayHandle *var_handles,
                               int retain_graph,
                               int create_graph,
                               int is_train,
                               NDArrayHandle **grad_handles,
                               int **grad_stypes);

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

  int (* MXListDataIters)(mx_uint *out_size, DataIterCreator **out_array);
  int (* MXDataIterCreateIter)(DataIterCreator handle,
                               mx_uint num_param,
                               const char **keys,
                               const char **vals,
                               DataIterHandle *out);
  int (* MXDataIterGetIterInfo)(DataIterCreator creator,
                                const char **name,
                                const char **description,
                                mx_uint *num_args,
                                const char ***arg_names,
                                const char ***arg_type_infos,
                                const char ***arg_descriptions);
  int (* MXDataIterFree)(DataIterHandle handle);
  int (* MXDataIterNext)(DataIterHandle handle,
                        int *out);
  int (* MXDataIterBeforeFirst)(DataIterHandle handle);
  int (* MXDataIterGetData)(DataIterHandle handle,
                            NDArrayHandle *out);
  int (* MXDataIterGetIndex)(DataIterHandle handle,
                             uint64_t **out_index,
                             uint64_t *out_size);
  int (* MXDataIterGetPadNum)(DataIterHandle handle,
                              int *pad);
  int (* MXDataIterGetLabel)(DataIterHandle handle,
                             NDArrayHandle *out);

  int (* MXSymbolCreateAtomicSymbol)(void *creator,
                                     mx_uint num_param,
                                     const char **keys,
                                     const char **vals,
                                     void *out);
  int (* MXSymbolCreateFromFile)(const char *fname, SymbolHandle *out);
  int (* MXSymbolCreateFromJSON)(const char *json, SymbolHandle *out);
  int (* MXSymbolCreateGroup)(mx_uint num_symbols,
                              SymbolHandle *symbols,
                              SymbolHandle *out);
  int (* NNSymbolCompose)(SymbolHandle sym,
                          const char *name,
                          mx_uint num_args,
                          const char **keys,
                          void **args);
  int (* MXSymbolCopy)(SymbolHandle symbol, SymbolHandle *out);
  int (* MXSymbolCreateVariable)(const char *name, void **out);
  int (* MXSymbolGetName)(SymbolHandle symbol,
                          const char** out,
                          int *success);
  int (* MXSymbolGetAttr)(SymbolHandle symbol,
                          const char *key,
                          const char **out,
                          int *success);
  int (* NNSymbolSetAttrs)(SymbolHandle symbol,
                           nn_uint num_attrs,
                           const char **keys,
                           const char **vals);
  int (* MXSymbolListAttr)(SymbolHandle symbol,
                           mx_uint *out_size,
                           const char ***out);
  int (* MXSymbolListArguments)(SymbolHandle symbol,
                                mx_uint *out_size,
                                const char ***out_str_array);
  int (* MXSymbolListAuxiliaryStates)(SymbolHandle symbol,
                                      mx_uint *out_size,
                                      const char ***out_str_array);
  int (* MXSymbolListOutputs)(SymbolHandle symbol,
                              mx_uint *out_size,
                              const char ***out_str_array);
  int (* MXSymbolInferShape)(SymbolHandle sym,
                             mx_uint num_args,
                             const char** keys,
                             const mx_uint *arg_ind_ptr,
                             const mx_uint *arg_shape_data,
                             mx_uint *in_shape_size,
                             const mx_uint **in_shape_ndim,
                             const mx_uint ***in_shape_data,
                             mx_uint *out_shape_size,
                             const mx_uint **out_shape_ndim,
                             const mx_uint ***out_shape_data,
                             mx_uint *aux_shape_size,
                             const mx_uint **aux_shape_ndim,
                             const mx_uint ***aux_shape_data,
                             int *complete);
  int (* MXSymbolInferShapePartial)(SymbolHandle sym,
                                    mx_uint num_args,
                                    const char** keys,
                                    const mx_uint *arg_ind_ptr,
                                    const mx_uint *arg_shape_data,
                                    mx_uint *in_shape_size,
                                    const mx_uint **in_shape_ndim,
                                    const mx_uint ***in_shape_data,
                                    mx_uint *out_shape_size,
                                    const mx_uint **out_shape_ndim,
                                    const mx_uint ***out_shape_data,
                                    mx_uint *aux_shape_size,
                                    const mx_uint **aux_shape_ndim,
                                    const mx_uint ***aux_shape_data,
                                    int *complete);
  int (* MXSymbolInferType)(SymbolHandle sym,
                            mx_uint num_args,
                            const char** keys,
                            const int *arg_type_data,
                            mx_uint *in_type_size,
                            const int **in_type_data,
                            mx_uint *out_type_size,
                            const int **out_type_data,
                            mx_uint *aux_type_size,
                            const int **aux_type_data,
                            int *complete);
  int (* MXSymbolSaveToFile)(SymbolHandle symbol, const char *fname);
  int (* MXSymbolSaveToJSON)(SymbolHandle symbol, const char **out_json);

  int (* MXCreateCachedOpEx)(SymbolHandle symbol,
                             int num_flags,
                             const char **keys,
                             const char **vals,
                             CachedOpHandle *cached_op);
  int (* MXFreeCachedOp)(CachedOpHandle cached_op);
  int (* MXInvokeCachedOpEx)(CachedOpHandle cached_op,
                             int num_inputs,
                             NDArrayHandle *inputs,
                             int *num_outputs,
                             NDArrayHandle **outputs,
                             int **out_stypes);
  int (* MXNDArrayGetStorageType)(NDArrayHandle handle, int *out_storage_type);
};

struct mxnet_api_table *mxnet_get_api_table(void);
#define MXNET_API(name) (mxnet_get_api_table()->name)

int mxnet_context_get_device_type_id(VALUE ctx);
int mxnet_context_get_device_id(VALUE ctx);

void *mxnet_get_handle(VALUE obj);
void mxnet_set_handle(VALUE obj, VALUE handle_v);

VALUE mxnet_dtype_id2name(int dtype_id);
VALUE mxnet_storage_type_id2name(int stype_id);
int mxnet_dtype_name2id(VALUE dtype_name);
VALUE mxnet_dtype_name(VALUE id_or_name);

VALUE mxnet_grad_req_map(void);

VALUE mxnet_executor_new(ExecutorHandle executor_handle, VALUE symbol, VALUE ctx, VALUE grad_req, VALUE group2ctx);
void mxnet_executor_set_arg_arrays(VALUE obj, VALUE args);
void mxnet_executor_set_grad_arrays(VALUE obj, VALUE args_grad);
void mxnet_executor_set_aux_arrays(VALUE obj, VALUE aux_states);

void mxnet_check_type(VALUE obj, VALUE klass);

VALUE mxnet_ndarray_new(NDArrayHandle ndarray_handle);
NDArrayHandle mxnet_ndarray_get_handle(VALUE obj);
VALUE mxnet_ndarray_get_shape(VALUE obj);

VALUE mxnet_symbol_new(SymbolHandle mxsymbol_handle);
VALUE mxnet_symbol_list_outputs(VALUE obj);

CachedOpHandle mxnet_cached_op_get_handle(VALUE obj);

void mxnet_init_libmxnet(void);
void mxnet_init_autograd(void);
void mxnet_init_cached_op(void);
void mxnet_init_executor(void);
void mxnet_init_io(void);
void mxnet_init_ndarray(void);
void mxnet_init_symbol(void);
void mxnet_init_operations(VALUE klass);
void mxnet_init_random(void);
void mxnet_init_utils(void);

NORETURN(void mxnet_raise_last_error(void));
#define CHECK_CALL(expr) if ((expr) != 0) mxnet_raise_last_error()

extern VALUE mxnet_mMXNet;
extern VALUE mxnet_mUtils;
extern VALUE mxnet_cCachedOp;
extern VALUE mxnet_cContext;
extern VALUE mxnet_cExecutor;
extern VALUE mxnet_cMXDataIter;
extern VALUE mxnet_cNDArray;
extern VALUE mxnet_cSymbol;

extern VALUE mxnet_sOpInfo;
extern VALUE mxnet_sOpArgInfo;

extern VALUE mxnet_eError;

static inline int
mxnet_is_ndarray(VALUE obj)
{
  return RTEST(rb_obj_is_kind_of(obj, mxnet_cNDArray));
}

static inline void
mxnet_check_ndarray(VALUE obj)
{
  mxnet_check_type(obj, mxnet_cNDArray);
}

static inline int
mxnet_is_symbol(VALUE obj)
{
  return RTEST(rb_obj_is_kind_of(obj, mxnet_cSymbol));
}

static inline void
mxnet_check_symbol(VALUE obj)
{
  mxnet_check_type(obj, mxnet_cSymbol);
}

static inline int
mxnet_is_cached_op(VALUE obj)
{
  return RTEST(rb_obj_is_kind_of(obj, mxnet_cCachedOp));
}

static inline void
mxnet_check_cached_op(VALUE obj)
{
  mxnet_check_type(obj, mxnet_cCachedOp);
}

#ifdef __cplusplus
#if 0
{ /* satisfy cc-mode */
#endif
} /* extern "C" { */
#endif

#endif /* MXNET_INTERNAL_H */
