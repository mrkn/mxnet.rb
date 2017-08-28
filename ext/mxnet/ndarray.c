#include "mxnet_internal.h"

VALUE mxnet_cNDArray;

void *
mxnet_ndarray_get_handle(VALUE obj)
{
  VALUE handle_v = rb_ivar_get(obj, rb_intern("mxnet_handle"));
  if (NIL_P(handle_v)) return NULL;
  return NUM2PTR(handle_v);
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
  dtype = kFloat32; /* TODO */

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
  rb_ivar_set(obj, rb_intern("mxnet_handle"), handle_v);
  return obj;
}

static VALUE
ndarray_get_shape(VALUE obj)
{
  void *handle;
  mx_uint ndim, i;
  mx_uint const* shape;
  int rv;
  VALUE ary;

  handle = mxnet_ndarray_get_handle(obj);
  rv = MXNET_API(MXNDArrayGetShape)(handle, &ndim, &shape);
  if (rv == -1) {
    mxnet_raise_last_error();
  }

  ary = rb_ary_new_capa(ndim);
  for (i = 0; i < ndim; ++i) {
    rb_ary_push(ary, MXUINT2NUM(shape[i]));
  }

  return ary;
}

void
mxnet_init_ndarray(void)
{
  VALUE cNDArray;
  cNDArray = rb_const_get_at(mxnet_mMXNet, rb_intern("NDArray"));

  rb_define_singleton_method(cNDArray, "empty", ndarray_s_empty, -1);

  rb_define_method(cNDArray, "initialize", ndarray_initialize, 1);
  rb_define_method(cNDArray, "shape", ndarray_get_shape, 0);

  mxnet_cNDArray = cNDArray;
}
