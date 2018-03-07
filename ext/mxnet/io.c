#include "mxnet_internal.h"

VALUE mxnet_cMXDataIter;

enum {
  DEBUG_SKIP_LOAD = 0x01,
  DEBUG_AT_BEGIN = 0x02,
};

typedef struct {
  DataIterHandle handle;
  VALUE flags;
  VALUE first_batch;
  VALUE provide_data;
  VALUE provide_label;
} mx_data_iter;

static void
data_iter_free(void *ptr)
{
  if (ptr != NULL) {
    CHECK_CALL(MXNET_API(MXDataIterFree)((DataIterHandle)ptr));
  }
}

static size_t
data_iter_memsize(void const *ptr)
{
  return 0;
}

static const rb_data_type_t data_iter_data_type = {
  "MXDataIter",
  {
    NULL,
    data_iter_free,
    data_iter_memsize,
  },
  0, 0, RUBY_TYPED_FREE_IMMEDIATELY
};

static DataIterHandle
get_data_iter_handle(VALUE obj)
{
  DataIterHandle handle;
  TypedData_Get_Struct(obj, void, &data_iter_data_type, handle);
  return handle;
}

static VALUE
data_iter_allocate(VALUE klass)
{
  return TypedData_Wrap_Struct(klass, &data_iter_data_type, NULL);
}

static int
data_iter_initialize_extract_kwargs_i(VALUE key, VALUE val, VALUE arg)
{
  VALUE *pmemo = (VALUE *)arg;
  char const **param_keys = (char const **)pmemo[0];
  char const **param_vals = (char const **)pmemo[1];

  if (RB_TYPE_P(key, T_SYMBOL)) {
    key = rb_sym_to_s(key);
  }
  *param_keys = StringValueCStr(key);
  rb_ary_push(pmemo[2], key);

  val = rb_String(val);
  *param_vals = StringValueCStr(val);
  rb_ary_push(pmemo[2], val);

  pmemo[0] = (VALUE)(param_keys + 1);
  pmemo[1] = (VALUE)(param_vals + 1);

  return ST_CONTINUE;
}

static VALUE
data_iter_initialize(int argc, VALUE *argv, VALUE obj)
{
  DataIterCreator creator_handle;
  DataIterHandle iter_handle;
  VALUE kwargs;
  VALUE buf = Qnil;
  mx_uint num_param = 0;
  char const **param_keys = NULL, **param_vals = NULL;

  creator_handle = (DataIterCreator)mxnet_get_handle(CLASS_OF(obj));
  if (creator_handle == NULL) {
    rb_raise(rb_eTypeError, "%s does not have a handle of DataIterCreator", rb_class2name(CLASS_OF(obj)));
  }

  rb_scan_args(argc, argv, ":", &kwargs);

  if (!NIL_P(kwargs)) {
    VALUE memo[3];

    if (RHASH_SIZE(kwargs) > UINT_MAX) {
      rb_raise(rb_eArgError, "too many keyword arguments");
    }

    num_param = (mx_uint)RHASH_SIZE(kwargs);
    buf = rb_str_tmp_new(sizeof(char const **) * num_param * 2);
    param_keys = (char const **)RSTRING_PTR(buf);
    param_vals = param_keys + num_param;

    memo[0] = (VALUE)param_keys;
    memo[1] = (VALUE)param_vals;
    memo[2] = rb_ary_tmp_new(num_param * 2);
    rb_hash_foreach(kwargs, data_iter_initialize_extract_kwargs_i, (VALUE)memo);
  }

  CHECK_CALL(MXNET_API(MXDataIterCreateIter)(creator_handle, num_param, param_keys, param_vals, &iter_handle));
  DATA_PTR(obj) = iter_handle;

  rb_call_super(argc, argv);

  return obj;
}

static VALUE
data_iter_reset_impl(VALUE obj)
{
  DataIterHandle handle;

  handle = get_data_iter_handle(obj);
  CHECK_CALL(MXNET_API(MXDataIterBeforeFirst)(handle));

  return Qnil;
}

static VALUE
data_iter_iter_next_impl(VALUE obj)
{
  DataIterHandle handle;
  int next_res = 0;

  handle = get_data_iter_handle(obj);
  CHECK_CALL(MXNET_API(MXDataIterNext)(handle, &next_res));

  return INT2NUM(next_res);
}

static VALUE
data_iter_current_data_impl(VALUE obj)
{
  DataIterHandle handle;
  NDArrayHandle ndary_handle;
  VALUE ndary;

  handle = get_data_iter_handle(obj);
  CHECK_CALL(MXNET_API(MXDataIterGetData)(handle, &ndary_handle));

  ndary = mxnet_ndarray_new(ndary_handle);
  return ndary;
}

static VALUE
data_iter_current_label_impl(VALUE obj)
{
  DataIterHandle handle;
  NDArrayHandle ndary_handle;
  VALUE ndary;

  handle = get_data_iter_handle(obj);
  CHECK_CALL(MXNET_API(MXDataIterGetLabel)(handle, &ndary_handle));

  ndary = mxnet_ndarray_new(ndary_handle);
  return ndary;
}

static VALUE
data_iter_current_pad_impl(VALUE obj)
{
  DataIterHandle handle;
  int pad;

  handle = get_data_iter_handle(obj);
  CHECK_CALL(MXNET_API(MXDataIterGetPadNum)(handle, &pad));

  return INT2NUM(pad);
}

static VALUE
data_iter_current_index_impl(VALUE obj)
{
  DataIterHandle handle;
  uint64_t i, size, *data;
  VALUE ary;

  handle = get_data_iter_handle(obj);
  CHECK_CALL(MXNET_API(MXDataIterGetIndex)(handle, &data, &size));

  if (size > LONG_MAX) {
    rb_raise(rb_eRuntimeError, "too many items from MXDataIterGetIndex");
  }

  ary = rb_ary_new_capa((long)size);
  for (i = 0; i < size; ++i) {
    rb_ary_push(ary, ULL2NUM(data[i]));
  }

  return ary;
}

static void
define_data_iter(VALUE mod, DataIterCreator handle)
{
  char const *name, *desc;
  mx_uint num_args;
  char const **arg_names, **arg_types, **arg_descs;
  VALUE klass;

  CHECK_CALL(MXNET_API(MXDataIterGetIterInfo)(handle, &name, &desc, &num_args, &arg_names, &arg_types, &arg_descs));

  klass = rb_define_class_under(mod, name, mxnet_cMXDataIter);
  rb_define_private_method(klass, "initialize", data_iter_initialize, -1);
  mxnet_set_handle(klass, PTR2NUM(handle));
}

void
mxnet_init_io(void)
{
  VALUE mIO;

  mIO = rb_const_get_at(mxnet_mMXNet, rb_intern("IO"));
  mxnet_cMXDataIter = rb_const_get_at(mIO, rb_intern("MXDataIter"));
  rb_define_alloc_func(mxnet_cMXDataIter, data_iter_allocate);
  rb_define_private_method(mxnet_cMXDataIter, "_reset", data_iter_reset_impl, 0);
  rb_define_private_method(mxnet_cMXDataIter, "_iter_next", data_iter_iter_next_impl, 0);
  rb_define_private_method(mxnet_cMXDataIter, "_current_data", data_iter_current_data_impl, 0);
  rb_define_private_method(mxnet_cMXDataIter, "_current_label", data_iter_current_label_impl, 0);
  rb_define_private_method(mxnet_cMXDataIter, "_current_pad", data_iter_current_pad_impl, 0);
  rb_define_private_method(mxnet_cMXDataIter, "_current_index", data_iter_current_index_impl, 0);

  {
    mx_uint i, size;
    DataIterCreator *creators;

    CHECK_CALL(MXNET_API(MXListDataIters)(&size, &creators));

    for (i = 0; i < size; ++i) {
      define_data_iter(mIO, creators[i]);
    }
  }
}
