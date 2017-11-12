#include "mxnet_internal.h"

VALUE mxnet_cExecutor;

VALUE
mxnet_executor_new(ExecutorHandle handle, VALUE symbol, VALUE ctx, VALUE grad_req, VALUE group2ctx)
{
  VALUE argv[5];
  argv[0] = PTR2NUM(handle);
  argv[1] = symbol;
  argv[2] = ctx;
  argv[3] = grad_req;
  argv[4] = group2ctx;
  return rb_class_new_instance(5, argv, mxnet_cExecutor);
}

void
mxnet_executor_set_arg_arrays(VALUE obj, VALUE args)
{
  rb_ivar_set(obj, rb_intern("@arg_arrays"), args);
}

void
mxnet_executor_set_grad_arrays(VALUE obj, VALUE args_grad)
{
  rb_ivar_set(obj, rb_intern("@grad_arrays"), args_grad);
}

void
mxnet_executor_set_aux_arrays(VALUE obj, VALUE aux_states)
{
  rb_ivar_set(obj, rb_intern("@aux_arrays"), aux_states);
}

static VALUE
executor_get_arg_dict(VALUE obj)
{
  return rb_ivar_get(obj, rb_intern("@arg_dict"));
}

static VALUE
executor_get_symbol(VALUE obj)
{
  return rb_ivar_get(obj, rb_intern("@symbol"));
}

static VALUE
executor_outputs(VALUE obj)
{
  mx_uint i, size;
  ExecutorHandle handle;
  NDArrayHandle *ndary_handles;
  VALUE res;

  handle = mxnet_get_handle(obj);
  CHECK_CALL(MXNET_API(MXExecutorOutputs)(handle, &size, &ndary_handles));

  res = rb_ary_new_capa(size);
  for (i = 0; i < size; ++i) {
    rb_ary_push(res, mxnet_ndarray_new(ndary_handles[i]));
  }

  return res;
}

struct process_kwargs_params {
  VALUE arg_dict;
};

static int
executer_forward_process_kwargs_i(VALUE key, VALUE val, VALUE arg)
{
  struct process_kwargs_params *params = (struct process_kwargs_params *)arg;
  VALUE ndary, ndary_shape, val_shape;

  /* TODO: support NumBuffer objects */

  if (!rb_obj_is_kind_of(val, mxnet_cNDArray)) {
    rb_raise(rb_eArgError, "only accept keyword argument of NDArrays");
  }

  ndary = rb_hash_lookup2(params->arg_dict, key, Qundef);
  if (ndary == Qundef) {
    rb_raise(rb_eTypeError, "Unknown argument %"PRIsVALUE, key);
  }

  ndary_shape = mxnet_ndarray_get_shape(ndary);
  val_shape = mxnet_ndarray_get_shape(val);
  if (!RTEST(rb_funcall(ndary_shape, rb_intern("=="), 1, val_shape))) {
    rb_raise(rb_eArgError,
        "Shape not match! Argument %"PRIsVALUE
        ", need: %"PRIsVALUE", received: %"PRIsVALUE,
        key, ndary_shape, val_shape);
  }

  rb_funcall(val, rb_intern("copyto"), 1, ndary);

  return ST_CONTINUE;
}

static VALUE
executor_forward(int argc, VALUE *argv, VALUE obj)
{
  VALUE kwargs, is_train, arg_dict;
  ExecutorHandle handle;

  rb_scan_args(argc, argv, "0:", &kwargs);
  is_train = Qundef;
  if (!NIL_P(kwargs)) {
    static ID kwarg_key;

    if (!kwarg_key) {
      kwarg_key = rb_intern("is_train");
    }
    rb_get_kwargs(kwargs, &kwarg_key, 0, -1, &is_train);
  }

  is_train = is_train == Qundef ? 0 : RTEST(is_train);

  if (!NIL_P(kwargs) && RHASH_SIZE(kwargs) > 0) {
    struct process_kwargs_params params;
    arg_dict = executor_get_arg_dict(obj);
    params.arg_dict = arg_dict;
    rb_hash_foreach(kwargs, executer_forward_process_kwargs_i, (VALUE)&params);
  }

  handle = mxnet_get_handle(obj);
  CHECK_CALL(MXNET_API(MXExecutorForward)(handle, (int)is_train));

  return executor_outputs(obj);
}

static VALUE
executor_backward(int argc, VALUE *argv, VALUE obj)
{
  ExecutorHandle handle;
  VALUE kwargs, out_grads, is_train;
  long i, num_ndarray_handles;
  VALUE ndarray_handles_str;
  NDArrayHandle *ndarray_handles;

  rb_scan_args(argc, argv, "0:", &kwargs);
  out_grads = Qundef;
  is_train = Qundef;
  if (!NIL_P(kwargs)) {
    static ID kwarg_keys[2];
    VALUE kwarg_vals[2];

    if (!kwarg_keys[0]) {
      kwarg_keys[0] = rb_intern("out_grads");
      kwarg_keys[1] = rb_intern("is_train");
    }
    rb_get_kwargs(kwargs, kwarg_keys, 0, 2, kwarg_vals);

    out_grads = kwarg_vals[0];
    is_train = kwarg_vals[1];
  }

  is_train = is_train == Qundef ? 1 : RTEST(is_train);

  if (NIL_P(out_grads) || out_grads == Qundef) {
    out_grads = rb_ary_new_capa(0);
  }
  else if (rb_obj_is_kind_of(out_grads, mxnet_cNDArray)) {
    VALUE ary = rb_ary_new_capa(1);
    rb_ary_push(ary, out_grads);
    out_grads = ary;
  }
  else if (RB_TYPE_P(out_grads, T_HASH)) {
    VALUE sym, outputs, ary;

    sym = executor_get_symbol(obj);
    outputs = mxnet_symbol_list_outputs(sym);

    ary = rb_ary_new_capa(RARRAY_LEN(outputs));
    for (i = 0; i < RARRAY_LEN(outputs); ++i) {
      VALUE k = RARRAY_AREF(outputs, i);
      rb_ary_push(ary, rb_hash_aref(out_grads, k));
    }
    out_grads = ary;
  }

  for (i = 0; i < RARRAY_LEN(out_grads); ++i) {
    if (!rb_obj_is_kind_of(RARRAY_AREF(out_grads, i), mxnet_cNDArray)) {
      rb_raise(rb_eTypeError, "inputs must be NDArray");
    }
  }

  num_ndarray_handles = RARRAY_LEN(out_grads);
  ndarray_handles_str = rb_str_tmp_new(sizeof(void *)*num_ndarray_handles);
  ndarray_handles = (void **)RSTRING_PTR(ndarray_handles_str);
  for (i = 0; i < RARRAY_LEN(out_grads); ++i) {
    VALUE ndary = RARRAY_AREF(out_grads, i);
    ndarray_handles[i] = mxnet_get_handle(ndary);
  }

  handle = mxnet_get_handle(obj);
  CHECK_CALL(MXNET_API(MXExecutorBackwardEx)(
        handle, (mx_uint)num_ndarray_handles, ndarray_handles, (int)is_train));

  return Qnil;
}

void
mxnet_init_executor(void)
{
  VALUE cExecutor;

  cExecutor = rb_const_get_at(mxnet_mMXNet, rb_intern("Executor"));

  rb_define_method(cExecutor, "forward", executor_forward, -1);
  rb_define_method(cExecutor, "backward", executor_backward, -1);

  rb_define_private_method(cExecutor, "get_outputs", executor_outputs, 0);

  mxnet_cExecutor = cExecutor;
}
