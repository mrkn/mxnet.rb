#include "mxnet_internal.h"

VALUE mxnet_mLibMXNet;
VALUE mxnet_eAPINotFound;
struct mxnet_api_table api_table;

struct mxnet_api_table *
mxnet_get_api_table(void)
{
  return &api_table;
}

struct lookup_api_args {
  VALUE handle;
  char const *name;
};

static VALUE
lookup_libmxnet_api_0(struct lookup_api_args *args)
{
  return rb_funcall(args->handle, rb_intern("sym"), 1, rb_str_new2(args->name));
}

static void *
lookup_libmxnet_api(VALUE handle, char const *name)
{
  struct lookup_api_args arg;
  VALUE addr;
  int state;

  arg.handle = handle;
  arg.name = name;
  addr = rb_protect((VALUE (*)(VALUE))lookup_libmxnet_api_0, (VALUE)&arg, &state);
  return (state || NIL_P(addr)) ? NULL : NUM2PTR(addr);
}

static void
init_api_table(VALUE handle)
{
#define LOOKUP_API_ENTRY(api_name) lookup_libmxnet_api(handle, #api_name)
#define CHECK_API_ENTRY(api_name) (LOOKUP_API_ENTRY(api_name) != NULL)
#define INIT_API_TABLE_ENTRY2(member_name, api_name) do { \
    void *fptr = LOOKUP_API_ENTRY(api_name); \
    if (!fptr) { \
      rb_raise(mxnet_eAPINotFound, "Unable to find the required symbol in libpython: %s", #api_name); \
    } \
    ((api_table).member_name) = fptr; \
  } while (0)
#define INIT_API_TABLE_ENTRY(api_name) INIT_API_TABLE_ENTRY2(api_name, api_name)

  INIT_API_TABLE_ENTRY(MXGetLastError);
  INIT_API_TABLE_ENTRY(MXNDArrayCreateEx);
  INIT_API_TABLE_ENTRY(MXNDArrayGetShape);

  INIT_API_TABLE_ENTRY(MXListAllOpNames);
  INIT_API_TABLE_ENTRY(NNGetOpHandle);
  INIT_API_TABLE_ENTRY(MXSymbolGetAtomicSymbolInfo);
  INIT_API_TABLE_ENTRY(MXImperativeInvoke);
}

static VALUE
imperative_invoke(VALUE mod, VALUE handle, VALUE ndargs, VALUE keys, VALUE vals, VALUE out)
{
  VALUE inputs_str, outputs_str, keys_str, vals_str;
  int i;
  int num_inputs, num_params, num_outputs = 0;
  void **inputs, **outputs = NULL;
  char const **params_keys, **params_vals;

  ndargs = rb_check_convert_type(ndargs, T_ARRAY, "Array", "to_ary");
  keys = rb_check_convert_type(keys, T_ARRAY, "Array", "to_ary");
  vals = rb_check_convert_type(vals, T_ARRAY, "Array", "to_ary");
  if (!NIL_P(out) && !RTEST(rb_obj_is_kind_of(out, mxnet_cNDArray))) {
    out = rb_check_convert_type(out, T_ARRAY, "Array", "to_ary");
  }

  num_inputs = (int)RARRAY_LEN(ndargs);
  inputs_str = rb_str_tmp_new(sizeof(void *)*num_inputs);
  inputs = (void **)RSTRING_PTR(inputs_str);
  for (i = 0; i < num_inputs; ++i) {
    inputs[i] = mxnet_ndarray_get_handle(RARRAY_AREF(ndargs, i));
  }

  num_params = (int)RARRAY_LEN(keys);
  keys_str = rb_str_tmp_new(sizeof(char const *)*num_params);
  params_keys = (char const **)RSTRING_PTR(keys_str);
  vals_str = rb_str_tmp_new(sizeof(char const *)*num_params);
  params_vals = (char const **)RSTRING_PTR(vals_str);
  for (i = 0; i < num_params; ++i) {
    VALUE key, val;

    key = RARRAY_AREF(keys, i);
    if (RB_TYPE_P(key, T_SYMBOL)) {
      key = rb_sym_to_s(key);
    }
    params_keys[i] = StringValueCStr(key);

    val = rb_String(RARRAY_AREF(vals, i));
    params_vals[i] = StringValueCStr(val);
  }

  if (!NIL_P(out)) {
    if (RTEST(rb_obj_is_kind_of(out, mxnet_cNDArray))) {
      num_outputs = 1;
      outputs_str = rb_str_tmp_new(sizeof(void *));
      outputs = (void **)RSTRING_PTR(outputs_str);
      outputs[0] = mxnet_ndarray_get_handle(out);
    }
    else {
      out = rb_check_convert_type(out, T_ARRAY, "Array", "to_ary");
      if (RARRAY_LEN(out) > INT_MAX) {
        rb_raise(rb_eArgError, "too many outputs (%ld)", RARRAY_LEN(out));
      }
      num_outputs = (int)RARRAY_LEN(out);
      outputs_str = rb_str_tmp_new(sizeof(void *) * num_outputs);
      outputs = (void **)RSTRING_PTR(outputs_str);
      for (i = 0; i < num_outputs; ++i) {
        VALUE v = RARRAY_AREF(out, i);
        outputs[i] = mxnet_ndarray_get_handle(v);
      }
    }
  }

  CHECK_CALL(MXNET_API(MXImperativeInvoke)(
        NUM2PTR(handle),
        num_inputs, inputs,
        &num_outputs, &outputs,
        num_params, params_keys, params_vals));

  if (!NIL_P(out)) {
    return out;
  }
  if (num_outputs == 1) {
    out = PTR2NUM(outputs[0]);
    return rb_class_new_instance(1, &out, mxnet_cNDArray);
  }

  out = rb_ary_new_capa(num_outputs);
  for (i = 0; i < num_outputs; ++i) {
    VALUE out_handle = PTR2NUM(outputs[i]);
    rb_ary_push(out, rb_class_new_instance(1, &out_handle, mxnet_cNDArray));
  }
  return out;
}

void
mxnet_init_libmxnet(void)
{
  VALUE handle;
  mxnet_mLibMXNet = rb_const_get_at(mxnet_mMXNet, rb_intern("LibMXNet"));
  mxnet_eAPINotFound = rb_define_class_under(mxnet_mMXNet, "APINotFound", mxnet_eError);
  handle = rb_funcallv(mxnet_mLibMXNet, rb_intern("handle"), 0, 0);
  rb_define_module_function(mxnet_mLibMXNet, "imperative_invoke", imperative_invoke, 5);
  init_api_table(handle);
}
