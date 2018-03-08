#include "mxnet_internal.h"

VALUE mxnet_mMXNet;
VALUE mxnet_mUtils;
VALUE mxnet_cContext;
VALUE mxnet_eError;

/* ==== Context ==== */

int
mxnet_context_get_device_type_id(VALUE ctx)
{
  VALUE v;
  v = rb_funcallv(ctx, rb_intern("device_type_id"), 0, NULL);
  return NUM2INT(v);
}

int
mxnet_context_get_device_id(VALUE ctx)
{
  VALUE v;
  v = rb_funcallv(ctx, rb_intern("device_id"), 0, NULL);
  return NUM2INT(v);
}

/* ==== Error ==== */

void
mxnet_raise_last_error(void)
{
  char const *last_error;

  last_error = MXNET_API(MXGetLastError)();
  rb_raise(mxnet_eError, "%s", last_error);
}

NORETURN(static void mxnet_unexpected_type(VALUE obj, char const *expected_type_name));

static void
mxnet_unexpected_type(VALUE obj, char const *expected_type_name)
{
  rb_raise(rb_eTypeError, "wrong argument type %s (expected %s)",
      rb_obj_classname(obj), expected_type_name);
}

void
mxnet_check_type(VALUE obj, VALUE klass)
{
  if (rb_obj_is_kind_of(obj, klass)) return;
  mxnet_unexpected_type(obj, rb_class2name(klass));
}

/* ==== Handle ==== */

static VALUE
handle_wrapper_initialize(VALUE obj, VALUE handle)
{
  mxnet_set_handle(obj, handle);
  return obj;
}

static inline VALUE
get_mxnet_handle(VALUE obj)
{
  return rb_ivar_get(obj, rb_intern("mxnet_handle"));
}

static VALUE
handle_wrapper_get_mxnet_handle(VALUE obj)
{
  return get_mxnet_handle(obj);
}

void *
mxnet_get_handle(VALUE obj)
{
  VALUE handle_v = get_mxnet_handle(obj);
  if (NIL_P(handle_v)) return NULL;
  return NUM2PTR(handle_v);
}

void
mxnet_set_handle(VALUE obj, VALUE handle_v)
{
  rb_ivar_set(obj, rb_intern("mxnet_handle"), handle_v);
}

VALUE
mxnet_grad_req_map(void)
{
  return rb_ivar_get(mxnet_mMXNet, rb_intern("GRAD_REQ_MAP"));
}

static void
init_grad_req_map(void)
{
  VALUE map;

  map = rb_hash_new();
  rb_hash_aset(map, ID2SYM(rb_intern("null")),  INT2FIX(0));
  rb_hash_aset(map, ID2SYM(rb_intern("write")), INT2FIX(1));
  rb_hash_aset(map, ID2SYM(rb_intern("add")),   INT2FIX(3));

  rb_ivar_set(mxnet_mMXNet, rb_intern("GRAD_REQ_MAP"), map);
}

void
Init_mxnet(void)
{
  VALUE mHandleWrapper;

  mxnet_mMXNet = rb_define_module("MXNet");
  mxnet_mUtils = rb_const_get_at(mxnet_mMXNet, rb_intern("Utils"));
  mxnet_cContext = rb_const_get_at(mxnet_mMXNet, rb_intern("Context"));
  mxnet_eError = rb_define_class_under(mxnet_mMXNet, "Error", rb_eStandardError);

  mHandleWrapper = rb_const_get_at(mxnet_mMXNet, rb_intern("HandleWrapper"));
  rb_define_method(mHandleWrapper, "initialize", handle_wrapper_initialize, 1);
  rb_define_private_method(mHandleWrapper, "__mxnet_handle__", handle_wrapper_get_mxnet_handle, 0);

  init_grad_req_map();
  mxnet_init_libmxnet();

  mxnet_init_autograd();

  mxnet_init_executor();

  mxnet_init_io();

  mxnet_init_ndarray();
  mxnet_init_operations(mxnet_cNDArray);

  mxnet_init_symbol();
  mxnet_init_operations(mxnet_cSymbol);

  mxnet_init_random();
  mxnet_init_utils();
}
