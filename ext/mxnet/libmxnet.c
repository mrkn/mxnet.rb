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
  INIT_API_TABLE_ENTRY(MXRandomSeed);

  INIT_API_TABLE_ENTRY(MXExecutorOutputs);
  INIT_API_TABLE_ENTRY(MXExecutorForward);
  INIT_API_TABLE_ENTRY(MXExecutorBackwardEx);
  INIT_API_TABLE_ENTRY(MXExecutorBindEX);

  INIT_API_TABLE_ENTRY(MXNDArrayCreateEx);
  INIT_API_TABLE_ENTRY(MXNDArrayFree);
  INIT_API_TABLE_ENTRY(MXNDArraySave);
  INIT_API_TABLE_ENTRY(MXNDArrayLoad);
  /* TODO: INIT_API_TABLE_ENTRY(MXNDArrayLoadFromBuffer); */
  INIT_API_TABLE_ENTRY(MXNDArrayReshape);
  INIT_API_TABLE_ENTRY(MXNDArrayGetContext);
  INIT_API_TABLE_ENTRY(MXNDArrayGetShape);
  INIT_API_TABLE_ENTRY(MXNDArrayGetDType);
  INIT_API_TABLE_ENTRY(MXNDArrayGetStorageType);
  INIT_API_TABLE_ENTRY(MXNDArraySyncCopyFromCPU);
  INIT_API_TABLE_ENTRY(MXNDArraySyncCopyToCPU);
  INIT_API_TABLE_ENTRY(MXNDArrayAt);
  INIT_API_TABLE_ENTRY(MXNDArraySlice);
  INIT_API_TABLE_ENTRY(MXNDArrayGetGrad);
  INIT_API_TABLE_ENTRY(MXNDArrayWaitToRead);

  INIT_API_TABLE_ENTRY(MXAutogradSetIsRecording);
  INIT_API_TABLE_ENTRY(MXAutogradSetIsTraining);
  INIT_API_TABLE_ENTRY(MXAutogradIsRecording);
  INIT_API_TABLE_ENTRY(MXAutogradIsTraining);
  INIT_API_TABLE_ENTRY(MXAutogradMarkVariables);
  INIT_API_TABLE_ENTRY(MXAutogradBackwardEx);

  INIT_API_TABLE_ENTRY(MXListAllOpNames);
  INIT_API_TABLE_ENTRY(NNGetOpHandle);
  INIT_API_TABLE_ENTRY(MXSymbolGetAtomicSymbolInfo);
  INIT_API_TABLE_ENTRY(MXImperativeInvoke);

  INIT_API_TABLE_ENTRY(MXListDataIters);
  INIT_API_TABLE_ENTRY(MXDataIterCreateIter);
  INIT_API_TABLE_ENTRY(MXDataIterGetIterInfo);
  INIT_API_TABLE_ENTRY(MXDataIterFree);
  INIT_API_TABLE_ENTRY(MXDataIterNext);
  INIT_API_TABLE_ENTRY(MXDataIterBeforeFirst);
  INIT_API_TABLE_ENTRY(MXDataIterGetData);
  INIT_API_TABLE_ENTRY(MXDataIterGetIndex);
  INIT_API_TABLE_ENTRY(MXDataIterGetPadNum);
  INIT_API_TABLE_ENTRY(MXDataIterGetLabel);

  INIT_API_TABLE_ENTRY(MXSymbolCreateFromFile);
  INIT_API_TABLE_ENTRY(MXSymbolCreateFromJSON);
  INIT_API_TABLE_ENTRY(MXSymbolCreateAtomicSymbol);
  INIT_API_TABLE_ENTRY(MXSymbolCreateGroup);
  INIT_API_TABLE_ENTRY(NNSymbolCompose);
  INIT_API_TABLE_ENTRY(MXSymbolCopy);
  INIT_API_TABLE_ENTRY(MXSymbolCreateVariable);
  INIT_API_TABLE_ENTRY(MXSymbolGetName);
  INIT_API_TABLE_ENTRY(MXSymbolGetAttr);
  INIT_API_TABLE_ENTRY(NNSymbolSetAttrs);
  INIT_API_TABLE_ENTRY(MXSymbolListAttr);
  INIT_API_TABLE_ENTRY(MXSymbolListArguments);
  INIT_API_TABLE_ENTRY(MXSymbolListAuxiliaryStates);
  INIT_API_TABLE_ENTRY(MXSymbolListOutputs);
  INIT_API_TABLE_ENTRY(MXSymbolInferShape);
  INIT_API_TABLE_ENTRY(MXSymbolInferShapePartial);
  INIT_API_TABLE_ENTRY(MXSymbolInferType);
  INIT_API_TABLE_ENTRY(MXSymbolSaveToFile);
  INIT_API_TABLE_ENTRY(MXSymbolSaveToJSON);

  INIT_API_TABLE_ENTRY(MXCreateCachedOpEx);
  INIT_API_TABLE_ENTRY(MXFreeCachedOp);
  INIT_API_TABLE_ENTRY(MXInvokeCachedOpEx);
}

static VALUE
imperative_invoke(VALUE mod, VALUE handle, VALUE ndargs, VALUE keys, VALUE vals, VALUE out)
{
  VALUE inputs_str, outputs_str, keys_str, vals_str;
  int i;
  int num_inputs, num_params, num_outputs = 0;
  void **inputs, **outputs = NULL;
  char const **params_keys, **params_vals;

  ndargs = rb_convert_type(ndargs, T_ARRAY, "Array", "to_ary");
  keys = rb_convert_type(keys, T_ARRAY, "Array", "to_ary");
  vals = rb_convert_type(vals, T_ARRAY, "Array", "to_ary");
  

  if (!NIL_P(out) && !RTEST(rb_obj_is_kind_of(out, mxnet_cNDArray))) {
    out = rb_convert_type(out, T_ARRAY, "Array", "to_ary");
  }

  num_inputs = (int)RARRAY_LEN(ndargs);
  inputs_str = rb_str_tmp_new(sizeof(void *)*num_inputs);
  inputs = (void **)RSTRING_PTR(inputs_str);
  for (i = 0; i < num_inputs; ++i) {
    inputs[i] = mxnet_ndarray_get_handle(RARRAY_AREF(ndargs, i));
  }

  num_params = (int)RARRAY_LEN(keys);
  keys_str = rb_str_tmp_new(sizeof(char const *)*num_params);
  rb_gc_mark(keys_str);
  params_keys = (char const **)RSTRING_PTR(keys_str);
  vals_str = rb_str_tmp_new(sizeof(char const *)*num_params);
  rb_gc_mark(vals_str);
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
      out = rb_convert_type(out, T_ARRAY, "Array", "to_ary");
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
    return mxnet_ndarray_new(outputs[0]);
  }

  out = rb_ary_new_capa(num_outputs);
  for (i = 0; i < num_outputs; ++i) {
    rb_ary_push(out, mxnet_ndarray_new(outputs[i]));
  }
  return out;
}

struct collect_sym_args_params {
  int cursor;
  int num_sym_args;
  char const **sym_keys;
  void **sym_args;
};

static int
collect_sym_args_i(VALUE key, VALUE val, VALUE arg)
{
  struct collect_sym_args_params *params = (struct collect_sym_args_params *)arg;
  if (RB_TYPE_P(key, T_SYMBOL)) {
    key = rb_sym_to_s(key);
  }
  params->sym_keys[params->cursor] = StringValueCStr(key);
  params->sym_args[params->cursor] = mxnet_get_handle(val);
  ++params->cursor;
  return ST_CONTINUE;
}

static VALUE
symbol_creator(VALUE mod, VALUE handle, VALUE args, VALUE kwargs, VALUE keys, VALUE vals, VALUE name)
{
  VALUE keys_str, vals_str, sym_keys_str, sym_args_str;
  int i;
  int num_params, num_sym_args;
  char const **params_keys, **params_vals, **sym_keys;
  void **sym_handle, **sym_args;

  keys = rb_convert_type(keys, T_ARRAY, "Array", "to_ary");
  vals = rb_convert_type(vals, T_ARRAY, "Array", "to_ary");

  num_params = (int)RARRAY_LEN(keys);
  keys_str = rb_str_tmp_new(sizeof(char const **)*num_params);
  rb_gc_mark(keys_str);
  params_keys = (char const **)RSTRING_PTR(keys_str);
  vals_str = rb_str_tmp_new(sizeof(char const **)*num_params);
  rb_gc_mark(vals_str);
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

  CHECK_CALL(MXNET_API(MXSymbolCreateAtomicSymbol)(
        NUM2PTR(handle),
        num_params,
        params_keys,
        params_vals,
        &sym_handle));

  if (!NIL_P(args) && !NIL_P(kwargs)) {
    rb_raise(rb_eTypeError,
        "Operators with variable length input can only accept input "
        "Symbols either as positional or keyword arguments, not both");
  }

  if (!NIL_P(args)) {
    num_sym_args = (int)RARRAY_LEN(args);
    sym_keys = NULL;
    sym_args_str = rb_str_tmp_new(sizeof(void *)*num_sym_args);
    sym_args = (void **)RSTRING_PTR(sym_args_str);
    for (i = 0; i < num_sym_args; ++i) {
      sym_args[i] = NUM2PTR(RARRAY_AREF(args, i));
    }
  }
  else if (!NIL_P(kwargs)) {
    struct collect_sym_args_params iter_params;

    num_sym_args = (int)RHASH_SIZE(kwargs);
    sym_keys_str = rb_str_tmp_new(sizeof(char const *)*num_sym_args);
    sym_keys = (char const **)RSTRING_PTR(sym_keys_str);
    sym_args_str = rb_str_tmp_new(sizeof(void *)*num_sym_args);
    sym_args = (void **)RSTRING_PTR(sym_args_str);

    iter_params.cursor = 0;
    iter_params.num_sym_args = num_sym_args;
    iter_params.sym_keys = sym_keys;
    iter_params.sym_args = sym_args;
    rb_hash_foreach(kwargs, collect_sym_args_i, (VALUE)&iter_params);
  }
  else {
    num_sym_args = 0;
    sym_keys = NULL;
    sym_args = NULL;
  }

  if (RB_TYPE_P(name, T_SYMBOL)) {
    name = rb_sym_to_s(name);
  }

  CHECK_CALL(MXNET_API(NNSymbolCompose)(
        sym_handle,
        StringValueCStr(name),
        num_sym_args, sym_keys, sym_args));

  return mxnet_symbol_new(sym_handle);
}

static VALUE
create_variable(VALUE mod, VALUE name)
{
  void *handle;
  char const *name_cstr;

  if (RB_TYPE_P(name, T_SYMBOL)) {
    name = rb_sym_to_s(name);
  }

  name_cstr = StringValueCStr(name);

  CHECK_CALL(MXNET_API(MXSymbolCreateVariable)(name_cstr, &handle));
  return PTR2NUM(handle);
}

void
mxnet_init_libmxnet(void)
{
  VALUE handle;
  mxnet_mLibMXNet = rb_const_get_at(mxnet_mMXNet, rb_intern("LibMXNet"));
  mxnet_eAPINotFound = rb_define_class_under(mxnet_mMXNet, "APINotFound", mxnet_eError);
  handle = rb_funcallv(mxnet_mLibMXNet, rb_intern("handle"), 0, 0);
  rb_define_module_function(mxnet_mLibMXNet, "imperative_invoke", imperative_invoke, 5);
  rb_define_module_function(mxnet_mLibMXNet, "symbol_creator", symbol_creator, 6);
  rb_define_module_function(mxnet_mLibMXNet, "create_variable", create_variable, 1);
  init_api_table(handle);
}
