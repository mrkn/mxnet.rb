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

  if (!RTEST(rb_obj_is_kind_of(sym, mxnet_cSymbol))) {
    rb_raise(rb_eTypeError, "wrong argument type %s (expected %"PRIsVALUE")",
             rb_obj_classname(sym), mxnet_cSymbol);
  }

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
