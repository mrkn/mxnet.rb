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

void
Init_mxnet(void)
{
  mxnet_mMXNet = rb_define_module("MXNet");
  mxnet_mUtils = rb_const_get_at(mxnet_mMXNet, rb_intern("Utils"));
  mxnet_cContext = rb_const_get_at(mxnet_mMXNet, rb_intern("Context"));
  mxnet_eError = rb_define_class_under(mxnet_mMXNet, "Error", rb_eStandardError);
  mxnet_init_libmxnet();
  mxnet_init_ndarray();
  mxnet_init_operations();
}
