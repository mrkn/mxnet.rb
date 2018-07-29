#include "mxnet_internal.h"

VALUE cCachedOp;

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

static VALUE
cached_op_allocate(VALUE klass)
{
  return TypedData_Wrap_Struct(klass, &cached_op_data_type, NULL);
}

struct create_cached_op_args {
  SymbolHandle symbol_handle;
  CachedOpHandle cached_op_handle;
  mx_uint num_params;
  char const **keys;
  char const **vals;
  int cursor;
};

static VALUE
call_create_cached_op(VALUE value) {
  struct create_cached_op_args *cargs = (struct create_cached_op_args *)value;

  CHECK_CALL(MXNET_API(MXCreateCachedOpEx)(cargs->symbol_handle,
                                           cargs->num_params,
                                           cargs->keys,
                                           cargs->vals,
                                           &cargs->cached_op_handle));

  return Qnil;
}

static int
check_opts(VALUE key, VALUE val, VALUE opts)
{
  if (SYMBOL_P(key)) {
    key = rb_sym_to_s(key);
  }
  StringValue(key);

  if (TYPE(val) != T_STRING) {
    val = rb_sprintf("%" PRIsVALUE, val);
  }
  StringValue(val);

  rb_hash_aset(opts, key, val);

  return ST_CONTINUE;
}

static int
collect_opts(VALUE key, VALUE val, VALUE value)
{
  struct create_cached_op_args *args = (struct create_cached_op_args *)value;

  args->keys[args->cursor] = StringValueCStr(key);
  args->vals[args->cursor] = StringValueCStr(val);
  ++args->cursor;

  return ST_CONTINUE;
}

static VALUE
cached_op_initialize(int argc, VALUE *argv, VALUE obj)
{
  int state = 0;
  VALUE sym, opts, temp, result;
  struct create_cached_op_args cargs = {0};
  unsigned long n;

  rb_scan_args(argc, argv, "1:", &sym, &opts);

  if (!RTEST(rb_obj_is_kind_of(sym, mxnet_cSymbol))) {
    rb_raise(rb_eTypeError, "wrong argument type %s (expected %"PRIsVALUE")",
             rb_obj_classname(sym), mxnet_cSymbol);
  }
  if (NIL_P(opts)) {
    opts = rb_hash_new();
  }

  n = RHASH_SIZE(opts);

  temp = rb_hash_new();
  rb_hash_foreach(opts, check_opts, temp);
  opts = temp;

  cargs.num_params = (mx_uint)n;
  cargs.keys = ALLOC_N(const char *, n);
  cargs.vals = ALLOC_N(const char *, n);
  cargs.cursor = 0;

  rb_hash_foreach(opts, collect_opts, (VALUE)&cargs);

  cargs.symbol_handle = (SymbolHandle)mxnet_get_handle(sym);

  result = rb_protect((VALUE (*)(VALUE))call_create_cached_op, (VALUE)&cargs, &state);

  xfree(cargs.keys);
  xfree(cargs.vals);

  if (state) {
    rb_jump_tag(state);
  }

  DATA_PTR(obj) = cargs.cached_op_handle;

  return obj;
}

struct invoke_cached_op_args {
  CachedOpHandle cached_op_handle;
  int num_inputs, num_outputs, *out_stypes;
  NDArrayHandle *inputs, *outputs;
};

static VALUE
call_invoke_cached_op(VALUE value) {
  struct invoke_cached_op_args *cargs = (struct invoke_cached_op_args *)value;

  CHECK_CALL(MXNET_API(MXInvokeCachedOpEx)(cargs->cached_op_handle,
                                           cargs->num_inputs,
                                           cargs->inputs,
                                           &cargs->num_outputs,
                                           &cargs->outputs,
                                           &cargs->out_stypes));

  return Qnil;
}

static VALUE
cached_op_call(VALUE obj, VALUE args)
{
  int state = 0;
  VALUE result;
  struct invoke_cached_op_args cargs = {0};
  unsigned long i, n;

  n = RARRAY_LEN(args);

  cargs.num_inputs = (mx_uint)n;
  cargs.inputs = ALLOC_N(NDArrayHandle, n);

  for (i = 0; i < n; ++i) {
    cargs.inputs[i] = mxnet_ndarray_get_handle(RARRAY_AREF(args, i));
  }

  cargs.cached_op_handle = (CachedOpHandle)DATA_PTR(obj);

  result = rb_protect((VALUE (*)(VALUE))call_invoke_cached_op, (VALUE)&cargs, &state);

  xfree(cargs.inputs);

  if (state) {
    rb_jump_tag(state);
  }

  if (cargs.num_outputs == 0) {
    return Qnil;
  }
  else if (cargs.num_outputs == 1) {
    return mxnet_ndarray_new(cargs.outputs[0]);
  }

  n = cargs.num_outputs;

  result = rb_ary_new2(n);

  for (i = 0; i < n; ++i) {
    rb_ary_push(result, mxnet_ndarray_new(cargs.outputs[i]));
  }

  return result;
}

void
mxnet_init_cached_op(void)
{
  cCachedOp = rb_define_class_under(mxnet_mMXNet, "CachedOp", rb_cObject);

  rb_define_alloc_func(cCachedOp, cached_op_allocate);

  rb_define_private_method(cCachedOp, "initialize", cached_op_initialize, -1);
  rb_define_method(cCachedOp, "call", cached_op_call, -2);
}
