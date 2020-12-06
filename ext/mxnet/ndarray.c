#include "mxnet_internal.h"

VALUE mxnet_cNDArray;

static size_t dtype_sizes[NUMBER_OF_DTYPE_IDS];
static ID dtype_name_ids[NUMBER_OF_DTYPE_IDS];

VALUE
mxnet_dtype_id2name(int dtype_id)
{
  if (0 <= dtype_id && dtype_id < NUMBER_OF_DTYPE_IDS) {
    return ID2SYM(dtype_name_ids[dtype_id]);
  }

  return Qnil;
}

static VALUE
dtype_m_id2name(VALUE mod, VALUE dtype_id_v)
{
  int dtype_id = NUM2INT(dtype_id_v);
  return mxnet_dtype_id2name(dtype_id);
}

int
mxnet_dtype_name2id(VALUE dtype_name)
{
  ID dtype_name_id;
  int i;

  if (!RB_TYPE_P(dtype_name, T_SYMBOL)) {
    dtype_name = rb_to_symbol(StringValue(dtype_name));
  }
  dtype_name_id = SYM2ID(dtype_name);

  for (i = 0; i < NUMBER_OF_DTYPE_IDS; ++i) {
    if (dtype_name_ids[i] == dtype_name_id) {
      return i;
    }
  }

  return -1;
}

static VALUE
dtype_m_name2id(VALUE mod, VALUE dtype_name)
{
  int dtype_id;
  dtype_id = mxnet_dtype_name2id(dtype_name);
  return dtype_id == -1 ? Qnil : INT2NUM(dtype_id);
}

VALUE
mxnet_dtype_name(VALUE id_or_name)
{
  int dtype_id;

  if (RB_INTEGER_TYPE_P(id_or_name)) {
    dtype_id = NUM2INT(id_or_name);
  }
  else {
    dtype_id = mxnet_dtype_name2id(id_or_name);
  }
  if (0 <= dtype_id && dtype_id < NUMBER_OF_DTYPE_IDS)
    return mxnet_dtype_id2name(dtype_id);

  return Qnil;
}

static VALUE
dtype_m_name(VALUE mod, VALUE id_or_name)
{
  return mxnet_dtype_name(id_or_name);
}

int
mxnet_dtype_is_available(VALUE dtype)
{
  int dtype_id;
  if (RB_INTEGER_TYPE_P(dtype)) {
    dtype_id = NUM2INT(dtype);
  }
  else {
    if (!RB_TYPE_P(dtype, T_SYMBOL)) {
      VALUE str = rb_check_convert_type(dtype, T_STRING, "String", "to_str");
      if (NIL_P(str)) {
        rb_raise(rb_eTypeError, "Invalid type for dtype (%"PRIsVALUE")", dtype);
      }
      dtype = str;
    }
    dtype_id = mxnet_dtype_name2id(dtype);
  }
  return 0 <= dtype_id && dtype_id < NUMBER_OF_DTYPE_IDS;
}

static VALUE
dtype_m_available_p(VALUE mod, VALUE dtype)
{
  return mxnet_dtype_is_available(dtype) ? Qtrue : Qfalse;
}


static size_t storage_type_sizes[NUMBER_OF_STORAGE_TYPE_IDS];
static ID storage_type_name_ids[NUMBER_OF_STORAGE_TYPE_IDS];

VALUE
mxnet_storage_type_id2name(int stype_id)
{
  if (0 <= stype_id && stype_id < NUMBER_OF_STORAGE_TYPE_IDS) {
    return ID2SYM(storage_type_name_ids[stype_id]);
  }

  return Qnil;
}

static VALUE
storage_type_m_id2name(VALUE mod, VALUE stype_id_v)
{
  int stype_id = NUM2INT(stype_id_v);
  return mxnet_storage_type_id2name(stype_id);
}

int
mxnet_storage_type_name2id(VALUE stype_name)
{
  ID stype_name_id;
  int i;

  if (!RB_TYPE_P(stype_name, T_SYMBOL)) {
    stype_name = rb_to_symbol(StringValue(stype_name));
  }
  stype_name_id = SYM2ID(stype_name);

  for (i = 0; i < NUMBER_OF_STORAGE_TYPE_IDS; ++i) {
    if (storage_type_name_ids[i] == stype_name_id) {
      return i;
    }
  }

  return -1;
}

static VALUE
storage_type_m_name2id(VALUE mod, VALUE stype_name)
{
  int stype_id;
  stype_id = mxnet_storage_type_name2id(stype_name);
  return stype_id == -1 ? Qnil : INT2NUM(stype_id);
}

VALUE
mxnet_storage_type_name(VALUE id_or_name)
{
  int stype_id;

  if (RB_INTEGER_TYPE_P(id_or_name)) {
    stype_id = NUM2INT(id_or_name);
  }
  else {
    stype_id = mxnet_storage_type_name2id(id_or_name);
  }
  if (0 <= stype_id && stype_id < NUMBER_OF_STORAGE_TYPE_IDS)
    return mxnet_storage_type_id2name(stype_id);

  return Qnil;
}

static VALUE
storage_type_m_name(VALUE mod, VALUE id_or_name)
{
  return mxnet_storage_type_name(id_or_name);
}

int
mxnet_storage_type_is_available(VALUE stype)
{
  int stype_id;
  if (RB_INTEGER_TYPE_P(stype)) {
    stype_id = NUM2INT(stype);
  }
  else {
    if (!RB_TYPE_P(stype, T_SYMBOL)) {
      VALUE str = rb_check_convert_type(stype, T_STRING, "String", "to_str");
      if (NIL_P(str)) {
        rb_raise(rb_eTypeError, "Invalid type for storage type (%"PRIsVALUE")", stype);
      }
      stype = str;
    }
    stype_id = mxnet_storage_type_name2id(stype);
  }
  return 0 <= stype_id && stype_id < NUMBER_OF_STORAGE_TYPE_IDS;
}

static VALUE
storage_type_m_available_p(VALUE mod, VALUE stype)
{
  return mxnet_storage_type_is_available(stype) ? Qtrue : Qfalse;
}





static void
ndarray_free(void *ptr)
{
  if (ptr != NULL) {
    CHECK_CALL(MXNET_API(MXNDArrayFree)((NDArrayHandle)ptr));
  }
}

static size_t
ndarray_memsize(void const *ptr)
{
  return 0;
}

static const rb_data_type_t ndarray_data_type = {
  "MXNet::NDArray",
  {
    NULL,
    ndarray_free,
    ndarray_memsize,
  },
  0, 0, RUBY_TYPED_FREE_IMMEDIATELY
};

NDArrayHandle
mxnet_ndarray_get_handle(VALUE obj)
{
  NDArrayHandle handle;
  TypedData_Get_Struct(obj, void, &ndarray_data_type, handle);
  return handle;
}

static VALUE
ndarray_allocate(VALUE klass)
{
  return TypedData_Wrap_Struct(klass, &ndarray_data_type, NULL);
}

VALUE
mxnet_ndarray_new(NDArrayHandle ndarray_handle)
{
  VALUE obj = rb_class_new_instance(0, NULL, mxnet_cNDArray);
  DATA_PTR(obj) = ndarray_handle;
  return obj;
}

static NDArrayHandle
ndarray_allocate_handle(VALUE shape_v, VALUE ctx_v, VALUE delay_alloc, VALUE dtype_v)
{
  mx_uint *shape;
  mx_uint ndim, i;
  void *handle;
  int dev_type, dev_id, dtype, rv;

  shape_v = rb_convert_type(shape_v, T_ARRAY, "Array", "to_ary");
  ndim = (mx_uint)RARRAY_LEN(shape_v);
  shape = ALLOC_N(mx_uint, ndim);
  for (i = 0; i < ndim; ++i) {
    shape[i] = NUM2MXUINT(RARRAY_AREF(shape_v, i));
  }

  dev_type = mxnet_context_get_device_type_id(ctx_v);
  dev_id = mxnet_context_get_device_id(ctx_v);
  if (NIL_P(dtype_v))
    dtype = kFloat32;
  else if (RB_INTEGER_TYPE_P(dtype_v)) {
    dtype = NUM2INT(dtype_v);
    if (NIL_P(mxnet_dtype_id2name(dtype))) {
      rb_raise(rb_eArgError, "invalid id of dtype: %d", dtype);
    }
  }
  else
    dtype = mxnet_dtype_name2id(dtype_v);

  rv = MXNET_API(MXNDArrayCreateEx)(shape, ndim, dev_type, dev_id, RTEST(delay_alloc), dtype, &handle);
  if (rv == -1) {
    mxnet_raise_last_error();
  }

  return handle;
}

static VALUE
ndarray_s_empty(int argc, VALUE *argv, VALUE klass)
{
  VALUE shape_v, opts, ctx_v, dtype_v;
  NDArrayHandle handle;

  rb_scan_args(argc, argv, "1:", &shape_v, &opts);
  shape_v = rb_convert_type(shape_v, T_ARRAY, "Array", "to_ary");
  ctx_v = Qundef;
  dtype_v = Qundef;
  if (!NIL_P(opts)) {
    static ID keywords[2];
    VALUE kwargs[2];
    if (!keywords[0]) {
      keywords[0] = rb_intern("ctx");
      keywords[1] = rb_intern("dtype");
    }
    rb_get_kwargs(opts, keywords, 0, 2, kwargs);
    ctx_v = kwargs[0];
    dtype_v = kwargs[1];
  }
  if (ctx_v == Qundef) {
    ctx_v = rb_funcallv(mxnet_cContext, rb_intern("default"), 0, NULL);
  }
  if (dtype_v == Qundef) {
    dtype_v = Qnil;
  }

  handle = ndarray_allocate_handle(shape_v, ctx_v, Qfalse, dtype_v);
  return mxnet_ndarray_new(handle);
}

static int
ndarray_s_save_extract_hash_i(VALUE key, VALUE val, VALUE arg)
{
  VALUE *pmemo = (VALUE *)arg;
  NDArrayHandle *handles = (NDArrayHandle *)pmemo[0];
  char const **keys = (char const **)pmemo[1];

  if (RB_TYPE_P(key, T_SYMBOL)) {
    key = rb_sym_to_s(key);
  }
  key = rb_check_string_type(key);

  if (NIL_P(key) || !mxnet_is_ndarray(val)) {
    rb_raise(rb_eArgError, "save only accept dict String => NDArray "
             "or an Array of NDArrays.");
  }

  *keys = RSTRING_PTR(key);
  rb_ary_push(pmemo[2], key);

  *handles = mxnet_ndarray_get_handle(val);

  pmemo[0] = (VALUE)(handles + 1);
  pmemo[1] = (VALUE)(keys + 1);

  return ST_CONTINUE;
}

/* Saves a list of arrays or a dict of str => array to file.
 *
 * Examples of filenames:
 *
 * - `/path/to/file`
 * - `s3://my-bucket/path/to/file` (if compiled with AWS S3 supports)
 * - `hdfs://path/to/file` (if compiled with HDFS supports)
 */
static VALUE
ndarray_s_save(VALUE klass, VALUE fname, VALUE data)
{
  char const *fname_cstr, **keys = NULL;
  VALUE handles_str, keys_str;
  NDArrayHandle *handles;
  mx_uint len;

  fname_cstr = StringValueCStr(fname); /* TODO: support pathname */

  if (mxnet_is_ndarray(data)) {
    len = 1;
    handles_str = rb_str_tmp_new(sizeof(NDArrayHandle)*len);
    handles = (NDArrayHandle *)RSTRING_PTR(handles_str);
    handles[0] = mxnet_ndarray_get_handle(data);
  }
  else if (RB_TYPE_P(data, T_HASH)) {
    VALUE memo[3];

#if LONG_MAX > UINT_MAX
    if (RHASH_SIZE(data) > UINT_MAX) {
      rb_raise(rb_eArgError, "the size of the give hash is too long.");
    }
#endif

    len = (mx_uint)RHASH_SIZE(data);
    handles_str = rb_str_tmp_new(sizeof(NDArrayHandle)*len);
    handles = (NDArrayHandle *)RSTRING_PTR(handles_str);
    keys_str = rb_str_tmp_new(sizeof(char const *)*len);
    keys = (char const **)RSTRING_PTR(keys_str);

    memo[0] = (VALUE)handles;
    memo[1] = (VALUE)keys;
    memo[2] = rb_ary_tmp_new(len);
    rb_hash_foreach(data, ndarray_s_save_extract_hash_i, (VALUE)memo);
  }
  else if (RB_TYPE_P(data, T_ARRAY)) {
    mx_uint i;

#if LONG_MAX > UINT_MAX
    if (RARRAY_LEN(data) > UINT_MAX) {
      rb_raise(rb_eArgError, "the size of the give array is too long.");
    }
#endif

    len = (mx_uint)RARRAY_LEN(data);
    handles_str = rb_str_tmp_new(sizeof(NDArrayHandle)*len);
    handles = (NDArrayHandle *)RSTRING_PTR(handles_str);
    for (i = 0; i < len; ++i) {
      VALUE ndary = RARRAY_AREF(data, i);
      if (!mxnet_is_ndarray(ndary)) {
        rb_raise(rb_eArgError, "save only accept Hash of String => NDArray "
                 "or Array of NDArrays.");
      }
      handles[i] = mxnet_ndarray_get_handle(ndary);
    }
  }
  else {
    rb_raise(rb_eArgError,
             "data needs to either be a NDArray, Hash of String => NDArray, "
             "or an Array of NDArrays.");
  }

  CHECK_CALL(MXNET_API(MXNDArraySave)(fname_cstr, len, handles, keys));

  return Qnil;
}

/* Loads an array from file.
 * See more details in `save`.
 */
static VALUE
ndarray_s_load(VALUE obj, VALUE fname)
{
  char const *fname_cstr;
  mx_uint out_size, out_name_size;
  NDArrayHandle *handles;
  char const **names;

  fname_cstr = StringValueCStr(fname);
  CHECK_CALL(MXNET_API(MXNDArrayLoad)(
    fname_cstr, &out_size, &handles, &out_name_size, &names));

  if (out_name_size == 0) {
    mx_uint i;
    VALUE ary = rb_ary_new_capa(out_size);
    for (i = 0; i < out_size; ++i) {
      VALUE ndary = mxnet_ndarray_new(handles[i]);
      rb_ary_push(ary, ndary);
    }
    return ary;
  }
  else {
    VALUE hsh;
    mx_uint i;
    if (out_size != out_name_size) {
      rb_raise(rb_eRuntimeError,
               "the loaded file is broken (out_size != out_name_size).");
    }
    hsh = rb_hash_new();
    for (i = 0; i < out_size; ++i) {
      VALUE name, ndary;
      name = rb_str_new2(names[i]);
      ndary = mxnet_ndarray_new(handles[i]);
      rb_hash_aset(hsh, name, ndary);
    }
    return hsh;
  }
}

/* Returns a **view**  of this array with a new shape without altering any data.
 *
 * @param [Array<Integer>] shape  The new shape should not change the array size.
 * @return [NDArray] An array with desired shape that shares data with this array.
 */
static VALUE
ndarray_reshape(VALUE obj, VALUE shape_v)
{
  NDArrayHandle handle, out_handle;
  VALUE dims_str;
  int ndim, *dims, i;

  handle = mxnet_ndarray_get_handle(obj);
  shape_v = rb_convert_type(shape_v, T_ARRAY, "Array", "to_ary");

  /* TODO: check INT_MAX */
  ndim = (int)RARRAY_LEN(shape_v);
  dims_str = rb_str_tmp_new(sizeof(int) * ndim);
  dims = (int *)RSTRING_PTR(dims_str);

  for (i = 0; i < ndim; ++i) {
    dims[i] = NUM2INT(RARRAY_AREF(shape_v, i));
  }

  CHECK_CALL(MXNET_API(MXNDArrayReshape)(handle, ndim, dims, &out_handle));

  return mxnet_ndarray_new(out_handle);
}

static VALUE
ndarray_get_mxnet_handle(VALUE obj)
{
  NDArrayHandle handle = mxnet_ndarray_get_handle(obj);
  return PTR2NUM(handle);
}

static VALUE
ndarray_get_context_params(VALUE obj)
{
  NDArrayHandle handle;
  int dev_typeid, dev_id;

  handle = mxnet_ndarray_get_handle(obj);

  CHECK_CALL(MXNET_API(MXNDArrayGetContext)(handle, &dev_typeid, &dev_id));

  return rb_assoc_new(INT2NUM(dev_typeid), INT2NUM(dev_id));
}

static int
ndarray_get_dtype_id(VALUE obj)
{
  NDArrayHandle handle;
  int dtype_id;

  handle = mxnet_ndarray_get_handle(obj);
  CHECK_CALL(MXNET_API(MXNDArrayGetDType)(handle, &dtype_id));
  return dtype_id;
}

static VALUE
ndarray_get_dtype(VALUE obj)
{
  int dtype_id = ndarray_get_dtype_id(obj);
  return mxnet_dtype_id2name(dtype_id);
}

static int
ndarray_get_storage_type_id(VALUE obj)
{
  NDArrayHandle handle;
  int stype_id;

  handle = mxnet_ndarray_get_handle(obj);
  CHECK_CALL(MXNET_API(MXNDArrayGetStorageType)(handle, &stype_id));
  return stype_id;
}

static VALUE
ndarray_get_storage_type(VALUE obj)
{
  int stype_id = ndarray_get_storage_type_id(obj);
  return mxnet_storage_type_id2name(stype_id);
}

VALUE
mxnet_ndarray_get_shape(VALUE obj)
{
  void *handle;
  mx_uint ndim, i;
  mx_uint const* shape;
  VALUE ary;

  handle = mxnet_ndarray_get_handle(obj);
  CHECK_CALL(MXNET_API(MXNDArrayGetShape)(handle, &ndim, &shape));

  ary = rb_ary_new_capa(ndim);
  for (i = 0; i < ndim; ++i) {
    rb_ary_push(ary, MXUINT2NUM(shape[i]));
  }

  return ary;
}

/* Returns a view of the array sliced at `idx` in the first dim.
 * This method is called through `x[idx]`.
 *
 * @param [Integer] idx  index of slicing the `NDArray` in the first dim.
 * @return [NDArray] `NDArray` sharing the memory with the current one sliced at `idx` in the first dim.
 */
static VALUE
ndarray_at(VALUE obj, VALUE idx_v)
{
  void *handle, *out_handle;
  mx_uint idx;

  handle = mxnet_ndarray_get_handle(obj);
  idx = NUM2MXUINT(idx_v);
  CHECK_CALL(MXNET_API(MXNDArrayAt)(handle, idx, &out_handle));

  return mxnet_ndarray_new(out_handle);
}

static VALUE
ndarray_slice(VALUE obj, VALUE start_v, VALUE stop_v)
{
  NDArrayHandle handle, out_handle;
  long start, stop, length;
  VALUE shape;

  shape = mxnet_ndarray_get_shape(obj);

  if (NIL_P(start_v)) {
    start = 0;
  }
  else {
    start = NUM2LONG(start_v);
    if (start < 0) {
      length = NUM2LONG(RARRAY_AREF(shape, 0));
      start += length;
      if (start < 0) {
        rb_raise(rb_eArgError, "Slicing start %ld exceeds limit of %ld", start - length, length);
      }
    }
  }

  if (NIL_P(stop_v)) {
    stop = NUM2LONG(RARRAY_AREF(shape, 0));
  }
  else {
    stop = NUM2LONG(stop_v);
    if (stop < 0) {
      length = NUM2LONG(RARRAY_AREF(shape, 0));
      stop += length;
      if (stop < 0) {
        rb_raise(rb_eArgError, "Slicing stop %ld exceeds limit of %ld", stop - length, length);
      }
    }
  }

  handle = mxnet_ndarray_get_handle(obj);
  CHECK_CALL(MXNET_API(MXNDArraySlice)(handle, (mx_uint)start, (mx_uint)stop, &out_handle));

  return mxnet_ndarray_new(out_handle);
}

static VALUE
ndarray_attach_grad(VALUE obj, VALUE grad_req_v, VALUE grad)
{
  mx_uint grad_req;
  NDArrayHandle self_handle, grad_handle;

  self_handle = mxnet_ndarray_get_handle(obj);
  grad_req = NUM2UINT(grad_req_v);
  grad_handle = mxnet_ndarray_get_handle(grad);

  CHECK_CALL(MXNET_API(MXAutogradMarkVariables)(1, &self_handle, &grad_req, &grad_handle));

  return Qnil;
}

static VALUE
ndarray_grad(VALUE obj)
{
  NDArrayHandle handle, grad_handle;
  VALUE grad;

  handle = mxnet_ndarray_get_handle(obj);
  CHECK_CALL(MXNET_API(MXNDArrayGetGrad)(handle, &grad_handle));
  if (grad_handle == NULL) {
    return Qnil;
  }
  grad = mxnet_ndarray_new(grad_handle);
  return grad;
}

/* Compute the gradients of this NDARray w.r.t variables.
 *
 * @param out_grad [MXNet::NDArray] Gradient with respect to head.
 * @param retain_graph [true, false]
 *   Whether to retain the computation graph for another backward
 *   pass on the same graph.  By default the computation history
 *   is cleared.
 * @param train_mode [true, false]
 *   Whether to compute gradient for training or inference.
 */
static VALUE
ndarray_backward(int argc, VALUE *argv, VALUE obj)
{
  VALUE opts;
  NDArrayHandle self_handle, ograd_handle = NULL;
  int retain_graph = 0, is_train = 1;

  rb_scan_args(argc, argv, ":", &opts);
  if (!NIL_P(opts)) {
    static ID keywords[3];
    VALUE kwargs[3];
    if (!keywords[0]) {
      keywords[0] = rb_intern("out_grad");
      keywords[1] = rb_intern("retain_graph");
      keywords[2] = rb_intern("train_mode");
    }
    rb_get_kwargs(opts, keywords, 0, 3, kwargs);
    /* out_grad */
    if (kwargs[0] != Qundef) {
      if (!rb_obj_is_kind_of(kwargs[0], mxnet_cNDArray)) {
        rb_raise(rb_eArgError, "out_grad must be a NDArray");
      }
      ograd_handle = mxnet_ndarray_get_handle(kwargs[0]);
    }
    /* retain_graph */
    if (kwargs[1] != Qundef) {
      retain_graph = RTEST(kwargs[1]);
    }
    /* train_mode */
    if (kwargs[2] != Qundef) {
      is_train = RTEST(kwargs[2]);
    }
  }

  self_handle = mxnet_ndarray_get_handle(obj);
  CHECK_CALL(MXNET_API(MXAutogradBackwardEx)(
        1, &self_handle, &ograd_handle,
        0, NULL,
        retain_graph, 0, is_train, NULL, NULL));

  return Qnil;
}

/* This function is based on npy_half_to_double and npy_halfbits_to_doublebits in numpy */
static double
float16_to_double(uint16_t h)
{
  union { double ret; uint64_t retbits; } conv;
  uint16_t h_exp, h_sig;
  uint64_t d_sgn, d_exp, d_sig;

  h_exp = (h&0x7c00u);
  d_sgn = ((uint64_t)h&0x8000u) << 48;
  switch (h_exp) {
    case 0x0000u: /* 0 or subnormal */
      h_sig = (h&0x03ffu);
      /* Signed zero */
      if (h_sig == 0) {
        conv.retbits = d_sgn;
        return conv.ret;
      }
      /* Subnormal */
      h_sig <<= 1;
      while ((h_sig&0x0400u) == 0) {
        h_sig <<= 1;
        h_exp++;
      }
      d_exp = ((uint64_t)(1023 - 15 - h_exp)) << 52;
      d_sig = ((uint64_t)(h_sig&0x03ffu)) << 42;
      conv.retbits = d_sgn + d_exp + d_sig;
      return conv.ret;
    case 0x7c00u: /* inf or NaN */
      /* All-ones exponent and a copy of the significand */
      conv.retbits = d_sgn + 0x7ff0000000000000ULL +
        (((uint64_t)(h&0x03ffu)) << 42);
      return conv.ret;
    default: /* normalized */
      /* Just need to adjust the exponent and shift */
      conv.retbits = d_sgn + (((uint64_t)(h&0x7fffu) + 0xfc000u) << 42);
      return conv.ret;
  }
}

static VALUE
ndarray_to_a(VALUE obj)
{
  void *handle;
  int dtype_id;
  size_t elsize, length;
  mx_uint ndim, i;
  mx_uint const* shape;
  VALUE data_str, ary;

  handle = mxnet_ndarray_get_handle(obj);

  CHECK_CALL(MXNET_API(MXNDArrayGetShape)(handle, &ndim, &shape));
  if (ndim > 1) {
    rb_raise(rb_eTypeError, "The current array is not a 1D array");
  }

  CHECK_CALL(MXNET_API(MXNDArrayGetDType)(handle, &dtype_id));
  if (dtype_id < 0 || NUMBER_OF_DTYPE_IDS <= dtype_id) {
    rb_raise(rb_eRuntimeError, "NDArray has an unexpected dtype %d", dtype_id);
  }

  length = shape[0];
  elsize = dtype_sizes[dtype_id];
  data_str = rb_str_tmp_new(elsize * length);
  CHECK_CALL(MXNET_API(MXNDArraySyncCopyToCPU)(handle, (void *)RSTRING_PTR(data_str), length));

  ary = rb_ary_new_capa(length);
  switch (dtype_id) {
    case kFloat32:
      {
        float *data = (float *)RSTRING_PTR(data_str);
        for (i = 0; i < length; ++i) {
          rb_ary_push(ary, rb_float_new(data[i]));
        }
      }
      break;
    case kFloat64:
      {
        double *data = (double *)RSTRING_PTR(data_str);
        for (i = 0; i < length; ++i) {
          rb_ary_push(ary, rb_float_new(data[i]));
        }
      }
      break;
    case kFloat16:
      {
        uint16_t *data = (uint16_t *)RSTRING_PTR(data_str);
        for (i = 0; i < length; ++i) {
          rb_ary_push(ary, rb_float_new(float16_to_double(data[i])));
        }
      }
      break;
    case kUint8:
      {
        uint8_t *data = (uint8_t *)RSTRING_PTR(data_str);
        for (i = 0; i < length; ++i) {
          rb_ary_push(ary, UINT2NUM(data[i]));
        }
      }
      break;
    case kInt32:
      {
        int32_t *data = (int32_t *)RSTRING_PTR(data_str);
        for (i = 0; i < length; ++i) {
          rb_ary_push(ary, INT2NUM(data[i]));
        }
      }
      break;
    case kInt8:
      {
        int8_t *data = (int8_t *)RSTRING_PTR(data_str);
        for (i = 0; i < length; ++i) {
          rb_ary_push(ary, INT2NUM(data[i]));
        }
      }
      break;
    case kInt64:
      {
#if SIZEOF_LONG == 8
        long *data = (long *)RSTRING_PTR(data_str);
        for (i = 0; i < length; ++i) {
          rb_ary_push(ary, LONG2NUM(data[i]));
        }
#elif SIZEOF_LONG_LONG == 8
        LONG_LONG *data = (LONG_LONG *)RSTRING_PTR(data_str);
        for (i = 0; i < length; ++i) {
          rb_ary_push(ary, LL2NUM(data[i]));
        }
#endif
      }
      break;
  }

  return ary;
}

static VALUE
ndarray_wait_to_read(VALUE obj)
{
  NDArrayHandle *handle;

  handle = mxnet_ndarray_get_handle(obj);
  CHECK_CALL(MXNET_API(MXNDArrayWaitToRead)(handle));

  return Qnil;
}

void
mxnet_init_ndarray(void)
{
  VALUE cNDArray, mDType, mStorageType;

  cNDArray = rb_const_get_at(mxnet_mMXNet, rb_intern("NDArray"));

  rb_define_alloc_func(cNDArray, ndarray_allocate);
  rb_undef_method(CLASS_OF(cNDArray), "new");

  rb_define_singleton_method(cNDArray, "empty", ndarray_s_empty, -1);
  rb_define_singleton_method(cNDArray, "save", ndarray_s_save, 2);
  rb_define_singleton_method(cNDArray, "load", ndarray_s_load, 1);
  /* TODO: rb_define_singleton_method(cNDArray, "load_from_buffer", ndarray_s_load_from_buffer, 1); */

  rb_define_method(cNDArray, "dtype", ndarray_get_dtype, 0);
  rb_define_method(cNDArray, "stype", ndarray_get_storage_type, 0);
  rb_define_method(cNDArray, "shape", mxnet_ndarray_get_shape, 0);
  rb_define_method(cNDArray, "reshape", ndarray_reshape, 1);
  rb_define_method(cNDArray, "grad", ndarray_grad, 0);
  rb_define_method(cNDArray, "backward", ndarray_backward, -1);
  rb_define_method(cNDArray, "to_a", ndarray_to_a, 0);
  rb_define_method(cNDArray, "wait_to_read", ndarray_wait_to_read, 0);

  rb_define_private_method(cNDArray, "__mxnet_handle__", ndarray_get_mxnet_handle, 0);
  rb_define_private_method(cNDArray, "_get_context_params", ndarray_get_context_params, 0);
  rb_define_private_method(cNDArray, "_at", ndarray_at, 1);
  rb_define_private_method(cNDArray, "_slice", ndarray_slice, 2);
  rb_define_private_method(cNDArray, "_attach_grad", ndarray_attach_grad, 2);

  mxnet_cNDArray = cNDArray;

  mDType = rb_define_module_under(mxnet_mMXNet, "DType");

  rb_define_module_function(mDType, "id2name", dtype_m_id2name, 1);
  rb_define_module_function(mDType, "name2id", dtype_m_name2id, 1);
  rb_define_module_function(mDType, "name", dtype_m_name, 1);
  rb_define_module_function(mDType, "available?", dtype_m_available_p, 1);

#define INIT_DTYPE(id, ctype, name) do { \
    dtype_name_ids[id] = rb_intern(name); \
    dtype_sizes[id]    = sizeof(ctype); \
  } while (0)

  INIT_DTYPE(kFloat32, float,    "float32");
  INIT_DTYPE(kFloat64, double,   "float64");
  INIT_DTYPE(kFloat16, uint16_t, "float16");
  INIT_DTYPE(kUint8,   uint8_t,  "uint8");
  INIT_DTYPE(kInt32,   int32_t,  "int32");
  INIT_DTYPE(kInt8,    int8_t,   "int8");
  INIT_DTYPE(kInt64,   int64_t,  "int64");

#undef INIT_DTYPE


  mStorageType = rb_define_module_under(mxnet_mMXNet, "StorageType");

  rb_define_module_function(mStorageType, "id2name", storage_type_m_id2name, 1);
  rb_define_module_function(mStorageType, "name2id", storage_type_m_name2id, 1);
  rb_define_module_function(mStorageType, "name", storage_type_m_name, 1);
  rb_define_module_function(mStorageType, "available?", storage_type_m_available_p, 1);



#define INIT_STYPE(id, name) do { \
    storage_type_name_ids[id] = rb_intern(name); \
  } while (0)

  INIT_STYPE(kDefaultStorage, "default");
  INIT_STYPE(kRowSparseStorage, "row_sparse");
  INIT_STYPE(kCSRStorage, "csr");

#undef INIT_STYPE

}
