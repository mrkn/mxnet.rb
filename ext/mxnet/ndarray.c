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

VALUE
mxnet_ndarray_new(void *ndarray_handle)
{
  VALUE handle_v = PTR2NUM(ndarray_handle);
  return rb_class_new_instance(1, &handle_v, mxnet_cNDArray);
}

static VALUE
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

  return PTR2NUM(handle);
}

static VALUE
ndarray_s_empty(int argc, VALUE *argv, VALUE klass)
{
  VALUE shape_v, opts, ctx_v, dtype_v, handle_v, obj;

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

  handle_v = ndarray_allocate_handle(shape_v, ctx_v, Qfalse, dtype_v);

  obj = rb_obj_alloc(klass);
  rb_obj_call_init(obj, 1, &handle_v);

  return obj;
}

static VALUE
ndarray_initialize(VALUE obj, VALUE handle_v)
{
  rb_call_super(1, &handle_v);
  return obj;
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

  handle = mxnet_get_handle(obj);
  shape_v = rb_check_convert_type(shape_v, T_ARRAY, "Array", "to_ary");

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
ndarray_get_context_params(VALUE obj)
{
  NDArrayHandle handle;
  int dev_typeid, dev_id;

  handle = mxnet_get_handle(obj);

  CHECK_CALL(MXNET_API(MXNDArrayGetContext)(handle, &dev_typeid, &dev_id));

  return rb_assoc_new(INT2NUM(dev_typeid), INT2NUM(dev_id));
}

static int
ndarray_get_dtype_id(VALUE obj)
{
  void *handle;
  int dtype_id;

  handle = mxnet_get_handle(obj);
  CHECK_CALL(MXNET_API(MXNDArrayGetDType)(handle, &dtype_id));
  return dtype_id;
}

static VALUE
ndarray_get_dtype(VALUE obj)
{
  return INT2NUM(ndarray_get_dtype_id(obj));
}

static VALUE
ndarray_get_dtype_name(VALUE obj)
{
  int dtype_id = ndarray_get_dtype_id(obj);
  return mxnet_dtype_id2name(dtype_id);
}

VALUE
mxnet_ndarray_get_shape(VALUE obj)
{
  void *handle;
  mx_uint ndim, i;
  mx_uint const* shape;
  VALUE ary;

  handle = mxnet_get_handle(obj);
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
  
  handle = mxnet_get_handle(obj);
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

  handle = mxnet_get_handle(obj);
  CHECK_CALL(MXNET_API(MXNDArraySlice)(handle, (mx_uint)start, (mx_uint)stop, &out_handle));

  return mxnet_ndarray_new(out_handle);
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

  handle = mxnet_get_handle(obj);

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

void
mxnet_init_ndarray(void)
{
  VALUE cNDArray, mDType;

  cNDArray = rb_const_get_at(mxnet_mMXNet, rb_intern("NDArray"));

  rb_define_singleton_method(cNDArray, "empty", ndarray_s_empty, -1);

  rb_define_method(cNDArray, "initialize", ndarray_initialize, 1);
  rb_define_method(cNDArray, "dtype", ndarray_get_dtype, 0);
  rb_define_method(cNDArray, "dtype_name", ndarray_get_dtype_name, 0);
  rb_define_method(cNDArray, "shape", mxnet_ndarray_get_shape, 0);
  rb_define_method(cNDArray, "reshape", ndarray_reshape, 1);
  rb_define_method(cNDArray, "to_a", ndarray_to_a, 0);

  rb_define_private_method(cNDArray, "_get_context_params", ndarray_get_context_params, 0);
  rb_define_private_method(cNDArray, "_at", ndarray_at, 1);
  rb_define_private_method(cNDArray, "_slice", ndarray_slice, 2);

  mxnet_cNDArray = cNDArray;

  mDType = rb_define_module_under(mxnet_mMXNet, "DType");

  rb_define_module_function(mDType, "id2name", dtype_m_id2name, 1);
  rb_define_module_function(mDType, "name2id", dtype_m_name2id, 1);
  rb_define_module_function(mDType, "name", dtype_m_name, 1);

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
}
