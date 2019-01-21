#include "mxnet_internal.h"

VALUE mxnet_cCachedOp;

static void
cached_op_free(void *ptr)
{
  if (ptr != NULL) {
    CHECK_CALL(MXNET_API(MXFreeCachedOp)((CachedOpHandle)ptr));
  }
}

static size_t
cached_op_memsize(void const *ptr)
{
  return 0;
}

static const rb_data_type_t cached_op_data_type = {
  "MXNet::CachedOp",
  {
    NULL,
    cached_op_free,
    cached_op_memsize,
  },
  0, 0, RUBY_TYPED_FREE_IMMEDIATELY
};

CachedOpHandle
mxnet_cached_op_get_handle(VALUE obj)
{
  CachedOpHandle handle;
  TypedData_Get_Struct(obj, void, &cached_op_data_type, handle);
  return handle;
}

static VALUE
cached_op_allocate(VALUE klass)
{
  return TypedData_Wrap_Struct(klass, &cached_op_data_type, NULL);
}

struct collect_params_args {
  VALUE gc_guard;
  char const **keys;
  char const **vals;
  size_t cursor;
};

static int
cached_op_collect_params_i(VALUE key, VALUE val, VALUE arg)
{
  struct collect_params_args *args = (struct collect_params_args *)arg;
  VALUE gc_guard = args->gc_guard;
  char const **keys = args->keys;
  char const **vals = args->vals;
  const size_t cursor = args->cursor;

  if (SYMBOL_P(key)) {
    key = rb_sym_to_s(key);
  }
  rb_ary_push(gc_guard, key);
  keys[cursor] = StringValueCStr(key);

  if (!RB_TYPE_P(val, T_STRING)) {
    val = rb_funcall(val, rb_intern("to_s"), 0);
  }
  rb_ary_push(gc_guard, val);
  vals[cursor] = StringValueCStr(val);

  ++args->cursor;

  return ST_CONTINUE;
}

static VALUE
cached_op_initialize(int argc, VALUE *argv, VALUE obj)
{
  VALUE sym, opts;
  SymbolHandle symbol_handle;
  CachedOpHandle cached_op_handle;
  int num_params = 0;

  VALUE gc_guard = Qnil, keys_str = Qnil, vals_str = Qnil;
  char const **keys = NULL, **vals = NULL;

  rb_scan_args(argc, argv, "1:", &sym, &opts);
  mxnet_check_symbol(sym);

  if (!NIL_P(opts)) {
    struct collect_params_args args;

    if (RHASH_SIZE(opts) > INT_MAX) {
      rb_raise(rb_eArgError, "too many flags");
    }
    num_params = (int)RHASH_SIZE(opts);

    gc_guard = rb_ary_new_capa((long)num_params*2);
    keys_str = rb_str_tmp_new(sizeof(char const *) * num_params);
    keys = (char const**)RSTRING_PTR(keys_str);
    vals_str = rb_str_tmp_new(sizeof(char const *) * num_params);
    vals = (char const**)RSTRING_PTR(vals_str);

    args.gc_guard = gc_guard;
    args.keys = keys;
    args.vals = vals;
    args.cursor = 0;

    rb_hash_foreach(opts, cached_op_collect_params_i, (VALUE)&args);
  }

  symbol_handle = (SymbolHandle)mxnet_get_handle(sym);

  CHECK_CALL(MXNET_API(MXCreateCachedOpEx)(
        symbol_handle,
        num_params,
        keys,
        vals,
        &cached_op_handle));

  DATA_PTR(obj) = cached_op_handle;

  return obj;
}

static VALUE
cached_op_call(int argc, VALUE *argv, VALUE obj)
{
  CachedOpHandle handle;
  VALUE args, kwargs, out, orig_out, output_vars_str, input_vars_str, result;
  NDArrayHandle *output_vars, *input_vars;
  int i, num_output, num_input, *out_stypes;

  rb_scan_args(argc, argv, "0*:", &args, &kwargs);

  out = Qundef;
  if (!NIL_P(kwargs)) {
    static ID kwarg_keys[1];
    VALUE kwarg_vals[1];

    if (!kwarg_keys[0]) {
      kwarg_keys[0] = rb_intern("out");
    }

    rb_get_kwargs(kwargs, kwarg_keys, 0, 1, kwarg_vals);
    out = kwarg_vals[0];
  }

  if (out != Qundef && !NIL_P(out)) {
    orig_out = out;
    if (mxnet_is_ndarray(out)) {
      out = rb_ary_new_capa(1);
      rb_ary_push(out, orig_out);
    }
    if (RARRAY_LEN(out) > INT_MAX) {
      rb_raise(rb_eArgError, "too many output NDArrays");
    }
    num_output = (int)RARRAY_LEN(out);
    output_vars_str = rb_str_tmp_new(sizeof(NDArrayHandle) * num_output);
    output_vars = (NDArrayHandle *)RSTRING_PTR(output_vars_str);
    for (i = 0; i < num_output; ++i) {
      output_vars[i] = mxnet_ndarray_get_handle(RARRAY_AREF(out, i));
    }
  }
  else {
    orig_out = Qnil;
    output_vars = NULL;
    num_output = 0;
  }

  out_stypes = NULL;

  if (RARRAY_LEN(args) > INT_MAX) {
    rb_raise(rb_eArgError, "too many input NDArrays");
  }
  num_input = (int)RARRAY_LEN(args);
  input_vars_str = rb_str_tmp_new(sizeof(NDArrayHandle) * num_input);
  input_vars = (NDArrayHandle *)RSTRING_PTR(input_vars_str);
  for (i = 0; i < num_input; ++i) {
    input_vars[i] = mxnet_ndarray_get_handle(RARRAY_AREF(args, i));
  }

  handle = mxnet_cached_op_get_handle(obj);
  CHECK_CALL(MXNET_API(MXInvokeCachedOpEx)(
        handle,
        num_input,
        input_vars,
        &num_output,
        &output_vars,
        &out_stypes));

  if (!NIL_P(orig_out)) {
    return orig_out;
  }

  if (num_output == 1) {
    return mxnet_ndarray_new(output_vars[0]);
  }

  result = rb_ary_new_capa(num_output);
  for (i = 0; i < num_output; ++i) {
    rb_ary_push(result, mxnet_ndarray_new(output_vars[i]));
  }

  return result;
}

void
mxnet_init_cached_op(void)
{
  mxnet_cCachedOp = rb_define_class_under(mxnet_mMXNet, "CachedOp", rb_cObject);

  rb_define_alloc_func(mxnet_cCachedOp, cached_op_allocate);

  rb_define_private_method(mxnet_cCachedOp, "initialize", cached_op_initialize, -1);
  rb_define_method(mxnet_cCachedOp, "call", cached_op_call, -1);
}
