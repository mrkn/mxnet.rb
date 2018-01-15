#include "mxnet_internal.h"

VALUE mxnet_cSymbol;

static VALUE
symbol_new_with_klass(VALUE klass, SymbolHandle symbol_handle)
{
  VALUE handle_v = PTR2NUM(symbol_handle);
  return rb_class_new_instance(1, &handle_v, klass);
}

VALUE
mxnet_symbol_new(SymbolHandle symbol_handle)
{
  return symbol_new_with_klass(mxnet_cSymbol, symbol_handle);
}

static VALUE
symbol_s_load(VALUE klass, VALUE filename)
{
  SymbolHandle handle;
  char const *fn_cstr;
  ID const id_to_path = rb_intern_const("to_path");

  if (rb_respond_to(filename, id_to_path)) {
    filename = rb_funcall(filename, id_to_path, 1, filename);
  }
  fn_cstr = StringValueCStr(filename);

  CHECK_CALL(MXNET_API(MXSymbolCreateFromFile)(fn_cstr, &handle));
  return symbol_new_with_klass(klass, handle);
}

/* Loads symbol from json string. */
static VALUE
symbol_s_load_json(VALUE klass, VALUE json_str)
{
  SymbolHandle handle;
  char const *json_cstr;

  json_cstr = StringValueCStr(json_str);

  CHECK_CALL(MXNET_API(MXSymbolCreateFromJSON)(json_cstr, &handle));
  return symbol_new_with_klass(klass, handle);
}

static VALUE
symbol_initialize(VALUE obj, VALUE handle_v)
{
  rb_call_super(1, &handle_v);
  return obj;
}

/* Gets name of the MXNet::Symbol, this function only works for non-grouped symbol.
 *
 * @return [Symbol]  The name of this symbol, returns `nil` for groupd symbol.
 */
static VALUE
symbol_get_name(VALUE obj)
{
  void *handle;
  char const *name;
  int success;

  handle = mxnet_get_handle(obj);
  CHECK_CALL(MXNET_API(MXSymbolGetName)(handle, &name, &success));

  return success ? ID2SYM(rb_intern(name)) : Qnil;
}

/* List all the arguments in the symbol.
 *
 * Example:
 *
 *     > a = MXNet.var(:a)
 *     > b = MXNet.var(:b)
 *     > c = a + b
 *     > c.list_arguments
 *     [:a, :b]
 *
 * @return [Array<Symbol>]
 *   List containing the names of the arguments required to compute the symbol.
 */
static VALUE
symbol_list_arguments(VALUE obj)
{
  void *handle;
  mx_uint i, size;
  char const **args;
  VALUE res;

  handle = mxnet_get_handle(obj);
  CHECK_CALL(MXNET_API(MXSymbolListArguments)(handle, &size, &args));

  res = rb_ary_new_capa(size);
  for (i = 0; i < size; ++i) {
    rb_ary_push(res, ID2SYM(rb_intern(args[i])));
  }

  return res;
}

/* List all the auxiliary states in the symbol.
 *
 * Example:
 *
 *     > a = MXNet.var(:a)
 *     > b = MXNet.var(:b)
 *     > c = a + b
 *     > c.list_auxiliary_states
 *     []
 *
 * Example of auxiliary states in `BatchNorm`.
 *
 *     > data = MXNet.var(:data)
 *     > weight = MXNet.var(:fc1_weight)
 *     > fc1 = MXNet::FullyConnected(data: data, weight: weight, name: :fc1, num_hidden: 128)
 *     > fc2 = MXNet::BatchNorm(fc1, name: :batchnorm0)
 *     > fc2.list_auxiliary_states
 *     [:batchnorm0_moving_mean, :batchnorm0_moving_var]
 *
 * Notes:
 *
 * Auxiliary states are special states of symbols that do not correspond to an argument,
 * and are not updated by gradient discent. Common examples of auxiliary states
 * include the `moving_mean` and `moving_variance` in `BatchNorm`.
 * Most operators do not have auxiliary states.
 *
 * @return [Array<Symbol>]  List of the auxiliary states in input symbol.
 */
static VALUE
symbol_list_auxiliary_states(VALUE obj)
{
  void *handle;
  mx_uint i, size;
  char const **states;
  VALUE res;

  handle = mxnet_get_handle(obj);
  CHECK_CALL(MXNET_API(MXSymbolListAuxiliaryStates)(handle, &size, &states));

  res = rb_ary_new_capa(size);
  for (i = 0; i < size; ++i) {
    rb_ary_push(res, ID2SYM(rb_intern(states[i])));
  }

  return res;
}

/* Lists all the outputs in the symbol.
 *
 * Example:
 *
 *     > a = MXNet.var(:a)
 *     > b = MXNet.var(:b)
 *     > c = a + b
 *     > c.list_outputs
 *     [:_plus12_output]
 *
 * @return [Array<Symbol>]  List of all the outputs.
 *   For most symbols, this list contains only the name of this symbol.
 *   For symbol groups, this is a list with the name of all symbols in the group.
 */
VALUE
mxnet_symbol_list_outputs(VALUE obj)
{
  SymbolHandle handle;
  mx_uint i, size;
  char const **outputs;
  VALUE res;

  handle = mxnet_get_handle(obj);
  CHECK_CALL(MXNET_API(MXSymbolListOutputs)(handle, &size, &outputs));

  res = rb_ary_new_capa(size);
  for (i = 0; i < size; ++i) {
    rb_ary_push(res, ID2SYM(rb_intern(outputs[i])));
  }

  return res;
}

struct infer_type_process_kwargs_params {
  char const **keys;
  int *tdata;
  long i;
};

static int
infer_type_process_kwargs_i(VALUE key, VALUE val, VALUE arg)
{
  struct infer_type_process_kwargs_params *params = (struct infer_type_process_kwargs_params *)arg;
  int dtype;

  if (RB_INTEGER_TYPE_P(val)) {
    dtype = NUM2INT(val);
  }
  else {
    dtype = mxnet_dtype_name2id(val);
  }

  if (dtype >= 0) {
    if (RB_TYPE_P(key, T_SYMBOL)) {
      key = rb_sym_to_s(key);
    }
    params->keys[params->i] = StringValueCStr(key);
    params->tdata[params->i] = dtype;
    ++params->i;
  }

  return ST_CONTINUE;
}

static inline VALUE
make_type_array(mx_uint size, int const *types)
{
  VALUE res;
  mx_uint i;
 
  res = rb_ary_new_capa(size);
  for (i = 0; i < size; ++i) {
    rb_ary_push(res, mxnet_dtype_id2name(types[i]));
  }

  return res;
}

/* TODO: document */
static VALUE
symbol_infer_type(int argc, VALUE *argv, VALUE obj)
{
  SymbolHandle handle;
  VALUE args, kwargs, tdata_str, keys_str;
  long i, args_len;
  int *tdata;
  char const **keys;
  mx_uint arg_type_size, out_type_size, aux_type_size;
  int const *arg_type_data, *out_type_data, *aux_type_data;
  int complete;

  rb_scan_args(argc, argv, "0*:", &args, &kwargs);

  if (RARRAY_LEN(args) != 0 && RHASH_SIZE(kwargs) != 0) {
    rb_raise(rb_eArgError, "Can only specify known argument shapes either by positional or kwargs way.");
  }

  if (RARRAY_LEN(args) != 0) {
    args_len = RARRAY_LEN(args);
    tdata_str = rb_str_tmp_new(sizeof(int) * args_len);
    tdata = (int *)RSTRING_PTR(tdata_str);
    keys = NULL;
    for (i = 0; i < args_len; ++i) {
      VALUE dtype = RARRAY_AREF(args, i);
      if (!NIL_P(dtype)) {
        if (RB_INTEGER_TYPE_P(dtype)) {
          tdata[i] = NUM2INT(dtype);
        }
        else {
          tdata[i] = mxnet_dtype_name2id(dtype);
        }
      }
      else {
        tdata[i] = -1;
      }
    }
  }
  else {
    struct infer_type_process_kwargs_params params;

    args_len = RHASH_SIZE(kwargs);
    keys_str = rb_str_tmp_new(sizeof(char const *) * args_len);
    keys = (char const **)RSTRING_PTR(keys_str);
    tdata_str = rb_str_tmp_new(sizeof(int) * args_len);
    tdata = (int *)RSTRING_PTR(tdata_str);

    params.keys = keys;
    params.tdata = tdata;
    params.i = 0;

    rb_hash_foreach(kwargs, infer_type_process_kwargs_i, (VALUE)&params);
  }

  handle = mxnet_get_handle(obj);
  CHECK_CALL(MXNET_API(MXSymbolInferType)(
        handle,
        (mx_uint)args_len,
        keys,
        tdata,
        &arg_type_size,
        &arg_type_data,
        &out_type_size,
        &out_type_data,
        &aux_type_size,
        &aux_type_data,
        &complete));

  if (complete) {
    VALUE arg_types, out_types, aux_types;
    arg_types = make_type_array(arg_type_size, arg_type_data);
    out_types = make_type_array(out_type_size, out_type_data);
    aux_types = make_type_array(aux_type_size, aux_type_data);
    return rb_ary_new_from_args(3, arg_types, out_types, aux_types);
  }
  else {
    return rb_ary_new_from_args(3, Qnil, Qnil, Qnil);
  }
}

/* Save symbol to a file. */
static VALUE
symbol_save(VALUE obj, VALUE filename)
{
  SymbolHandle handle;
  char const *fn_cstr;
  ID const id_to_path = rb_intern_const("to_path");

  if (rb_respond_to(filename, id_to_path)) {
    filename = rb_funcall(filename, id_to_path, 1, filename);
  }
  fn_cstr = StringValueCStr(filename);

  handle = mxnet_get_handle(obj);
  CHECK_CALL(MXNET_API(MXSymbolSaveToFile)(handle, fn_cstr));

  return obj;
}

/* Saves symbol to a JSON string. */
static VALUE
symbol_to_json(VALUE obj)
{
  SymbolHandle handle;
  char const *out_json;

  handle = mxnet_get_handle(obj);
  CHECK_CALL(MXNET_API(MXSymbolSaveToJSON)(handle, &out_json));
  return rb_str_new2(out_json);
}

/* Helper function to get NDArray arrays handles from various inputs.
 *
 * @param arg_key [char const*]  The name of argument, used for error message.
 * @param args [Array<NDArray>, Hash<Symbol, NDArray>]
 *   Input arguments to the symbols.
 *   If type is Array of NDArray, the position is in the same order of arg_names.
 *   If type is Hash of Symbol to NDArray, then it maps the name of arguments to the corresponding NDArray.
 * @param arg_names [Array<Symbol>]  List of argument names.
 * @param allow_missing [true, false]  Whether missing argument is allowed.
 *   When allowed, the missing handle will be set to NULL.
 * @param out_handles [void ***]  The positional array of NDArray handles generated from the input.
 * @return [Array<NDArray>]
 */
static VALUE
get_ndarray_inputs(char const *arg_key, VALUE args, VALUE arg_names, int allow_missing, void ***out_handles)
{
  long i;
  void **arg_handles;
  VALUE arg_arrays;

  assert(RB_TYPE_P(arg_names, T_ARRAY));

  if (RB_TYPE_P(args, T_ARRAY)) {
    long const n = RARRAY_LEN(args);
    if (n != RARRAY_LEN(arg_names)) {
      rb_raise(rb_eArgError, "Length of %s does not match the number of arguments", arg_key);
    }
    arg_handles = ALLOC_N(void *, n);
    for (i = 0; i < n; ++i) {
      VALUE ndary = RARRAY_AREF(args, i);
      if (!rb_obj_is_kind_of(ndary, mxnet_cNDArray)) {
        rb_raise(rb_eTypeError, "Only accept Array of NDArrays or Hash of Symbol to NDArray");
      }
      arg_handles[i] = mxnet_get_handle(ndary);
    }
    arg_arrays = args;
  }
  else if (RB_TYPE_P(args, T_HASH)) {
    long const n = RARRAY_LEN(arg_names);
    arg_handles = ALLOC_N(void *, n);
    arg_arrays = rb_ary_new();
    for (i = 0; i < RARRAY_LEN(arg_names); ++i) {
      VALUE name, ndary;
      name = RARRAY_AREF(arg_names, i);
      ndary = rb_hash_lookup2(args, name, Qundef);
      if (ndary != Qundef) {
        if (!rb_obj_is_kind_of(ndary, mxnet_cNDArray)) {
          rb_raise(rb_eTypeError, "Only accept array of NDArrays or hash of Symbol to NDArray");
        }
        arg_handles[i] = mxnet_get_handle(ndary);
        rb_ary_push(arg_arrays, ndary);
      }
      else if (allow_missing) {
        arg_handles[i] = NULL;
        rb_ary_push(arg_arrays, Qnil);
      }
      else {
        rb_raise(rb_eArgError, "key `%"PRIsVALUE"` is missing in `%s`", name, arg_key);
      }
    }
  }
  else {
    rb_raise(rb_eTypeError, "Only accept Array of NDArrays or Hash of Symbol to NDArray");
  }

  *out_handles = arg_handles;
  return arg_arrays;
}

struct group2ctx_process_params {
  mx_uint cursor;
  char const **keys;
  int *dev_types;
  int *dev_ids;
};

static int
symbol_bind_group2ctx_process_i(VALUE key, VALUE val, VALUE arg)
{
  struct group2ctx_process_params *params = (struct group2ctx_process_params *)arg;

  if (RB_TYPE_P(key, T_SYMBOL)) {
    key = rb_sym_to_s(key);
  }
  params->keys[params->cursor] = StringValueCStr(key);
  params->dev_types[params->cursor] = mxnet_context_get_device_type_id(val);
  params->dev_ids[params->cursor] = mxnet_context_get_device_id(val);

  ++params->cursor;

  return ST_CONTINUE;
}

static VALUE
symbol_bind(int argc, VALUE *argv, VALUE obj)
{
  long i;
  VALUE ctx, args, kwargs, listed_arguments;
  VALUE args_grad, grad_req, aux_states, group2ctx, shared_exec;
  VALUE args_grad_handle_str, grad_req_map, reqs_array_str;
  void **args_handle, **args_grad_handle, **aux_args_handle;
  void *handle, *shared_exec_handle, *exec_handle;
  mx_uint *reqs_array, num_ctx_map_keys;
  VALUE ctx_map_keys_str, ctx_map_dev_types_str, ctx_map_dev_ids_str;
  char const **ctx_map_keys;
  int *ctx_map_dev_types, *ctx_map_dev_ids;
  VALUE executor;

  rb_scan_args(argc, argv, "2:", &ctx, &args, &kwargs);

  args_grad = Qundef;
  grad_req = Qundef;
  aux_states = Qundef;
  group2ctx = Qundef;
  shared_exec = Qundef;
  if (!NIL_P(kwargs)) {
    static ID kwarg_keys[5];
    VALUE kwarg_vals[5];

    if (!kwarg_keys[0]) {
      kwarg_keys[0] = rb_intern("args_grad");
      kwarg_keys[1] = rb_intern("grad_req");
      kwarg_keys[2] = rb_intern("aux_states");
      kwarg_keys[3] = rb_intern("group2ctx");
      kwarg_keys[4] = rb_intern("shared_exec");
    }

    rb_get_kwargs(kwargs, kwarg_keys, 0, 5, kwarg_vals);
    args_grad = kwarg_vals[0];
    grad_req = kwarg_vals[1];
    aux_states = kwarg_vals[2];
    group2ctx = kwarg_vals[3];
    shared_exec = kwarg_vals[4];
  }

  if (!rb_obj_is_kind_of(ctx, mxnet_cContext)) {
    rb_raise(rb_eTypeError, "Context type error");
  }

  listed_arguments = symbol_list_arguments(obj);
  args = get_ndarray_inputs("args", args, listed_arguments, 0, &args_handle);

  /* setup args gradient */
  if (args_grad == Qundef) {
    args_grad = Qnil;
  }
  if (NIL_P(args_grad)) {
    args_grad_handle_str = rb_str_tmp_new(sizeof(void *) * RARRAY_LEN(args));
    args_grad_handle = (void **)RSTRING_PTR(args_grad_handle_str);
    for (i = 0; i < RARRAY_LEN(args); ++i) {
      args_grad_handle[i] = NULL;
    }
  }
  else {
    args_grad = get_ndarray_inputs("args_grad", args_grad, listed_arguments, 1, &args_grad_handle);
    args_grad_handle_str = Qundef;
  }

  if (aux_states == Qundef) {
    aux_states = Qnil;
  }
  if (NIL_P(aux_states)) {
    aux_states = rb_ary_new();
  }
  aux_states = get_ndarray_inputs(
      "aux_states", aux_states, symbol_list_auxiliary_states(obj), 0,
      &aux_args_handle);

  /* setup requirements */
  if (grad_req == Qundef) {
    grad_req = ID2SYM(rb_intern("write"));
  }
  if (RB_TYPE_P(grad_req, T_STRING)) {
    grad_req = rb_str_intern(grad_req);
  }
  grad_req_map = mxnet_grad_req_map();
  if (RB_TYPE_P(grad_req, T_SYMBOL)) {
    mx_uint req;

    if (rb_hash_lookup2(grad_req_map, grad_req, Qundef) == Qundef) {
      rb_raise(rb_eArgError, "grad_req must be in %"PRIsVALUE, grad_req_map);
    }

    reqs_array_str = rb_str_tmp_new(sizeof(mx_uint) * RARRAY_LEN(listed_arguments));
    reqs_array = (mx_uint *)RSTRING_PTR(reqs_array_str);
    req = NUM2UINT(rb_hash_lookup(grad_req_map, grad_req));
    for (i = 0; i < RARRAY_LEN(listed_arguments); ++i) {
      reqs_array[i] = req;
    }
  }
  else if (RB_TYPE_P(grad_req, T_ARRAY)) {
    reqs_array_str = rb_str_tmp_new(sizeof(mx_uint) * RARRAY_LEN(grad_req));
    reqs_array = (mx_uint *)RSTRING_PTR(reqs_array_str);
      for (i = 0; i < RARRAY_LEN(grad_req); ++i) {
        VALUE item = RARRAY_AREF(grad_req, i);
        mx_uint req = NUM2UINT(rb_hash_lookup(grad_req_map, item));
        reqs_array[i] = req;
      }
  }
  else if (RB_TYPE_P(grad_req, T_HASH)) {
    reqs_array_str = rb_str_tmp_new(sizeof(mx_uint) * RARRAY_LEN(listed_arguments));
    reqs_array = (mx_uint *)RSTRING_PTR(reqs_array_str);
    for (i = 0; i < RARRAY_LEN(listed_arguments); ++i) {
      VALUE name = RARRAY_AREF(listed_arguments, i);
      mx_uint req = NUM2UINT(rb_hash_lookup2(grad_req, name, INT2FIX(0)));
      reqs_array[i] = req;
    }
  }
  else {
    rb_raise(rb_eArgError,
        "Invalid type of grad_req (%"PRIsVALUE" for Symbol, Array, or Hash)",
        CLASS_OF(grad_req));
  }

  if (group2ctx == Qundef) {
    group2ctx = Qnil;
  }
  if (!NIL_P(group2ctx)) {
    struct group2ctx_process_params params;

    Check_Type(group2ctx, T_HASH);
    num_ctx_map_keys = (mx_uint)RHASH_SIZE(group2ctx);
    ctx_map_keys_str = rb_str_tmp_new(sizeof(char *) * num_ctx_map_keys);
    ctx_map_keys = (char const **) RSTRING_PTR(ctx_map_keys_str);
    ctx_map_dev_types_str = rb_str_tmp_new(sizeof(int) * num_ctx_map_keys);
    ctx_map_dev_types = (int *) RSTRING_PTR(ctx_map_dev_types_str);
    ctx_map_dev_ids_str = rb_str_tmp_new(sizeof(int) * num_ctx_map_keys);
    ctx_map_dev_ids = (int *) RSTRING_PTR(ctx_map_dev_ids_str);

    params.cursor = 0;
    params.keys = ctx_map_keys;
    params.dev_types = ctx_map_dev_types;
    params.dev_ids = ctx_map_dev_ids;
    rb_hash_foreach(group2ctx, symbol_bind_group2ctx_process_i, (VALUE)&params);
  }
  else {
    num_ctx_map_keys = 0;
    ctx_map_keys = NULL;
    ctx_map_dev_types = NULL;
    ctx_map_dev_ids = NULL;
  }

  if (shared_exec == Qundef) {
    shared_exec = Qnil;
  }
  if (!NIL_P(shared_exec)) {
    shared_exec_handle = mxnet_get_handle(shared_exec);
  }
  else {
    shared_exec_handle = NULL;
  }

  handle = mxnet_get_handle(obj);
  CHECK_CALL(MXNET_API(MXExecutorBindEX)(
        handle,
        mxnet_context_get_device_type_id(ctx),
        mxnet_context_get_device_id(ctx),
        num_ctx_map_keys,
        ctx_map_keys,
        ctx_map_dev_types,
        ctx_map_dev_ids,
        (mx_uint) RARRAY_LEN(args),
        args_handle,
        args_grad_handle,
        reqs_array,
        (mx_uint)RARRAY_LEN(aux_states),
        aux_args_handle,
        shared_exec_handle,
        &exec_handle));

  executor = mxnet_executor_new(exec_handle, obj, ctx, grad_req, group2ctx);
  mxnet_executor_set_arg_arrays(executor, args);
  mxnet_executor_set_grad_arrays(executor, args_grad);
  mxnet_executor_set_aux_arrays(executor, aux_states);

  /* TODO: should use rb_ensure */

  xfree(args_handle);
  if (args_grad_handle_str == Qundef) {
    xfree(args_grad_handle);
  }
  xfree(aux_args_handle);

  return executor;
}

static VALUE
symbol_dup(VALUE obj)
{
  SymbolHandle handle, copy_handle;

  handle = mxnet_get_handle(obj);
  CHECK_CALL(MXNET_API(MXSymbolCopy)(handle, &copy_handle));
  
  return mxnet_symbol_new(copy_handle);
}

static inline void
sdata_extend(VALUE sdata_str, long *sdata_capa, long *sdata_len, VALUE shape, long remaining_count)
{
  mx_uint *sdata = (mx_uint *)RSTRING_PTR(sdata_str);
  long i;

  if (*sdata_capa < *sdata_len + RARRAY_LEN(shape)) {
    *sdata_capa += remaining_count * 3;
    rb_str_resize(sdata_str, sizeof(mx_uint) * *sdata_capa);
    sdata = (mx_uint *)RSTRING_PTR(sdata_str);
  }
  for (i = 0; i < RARRAY_LEN(shape); ++i) {
    sdata[(*sdata_len)++] = NUM2UINT(RARRAY_AREF(shape, i));
  }
}

struct infer_shape_process_kwargs_params {
  char const **keys;
  mx_uint *indptr;
  VALUE sdata_str;
  long *sdata_capa;
  long *sdata_len;
  long i;
  long total_count;
};

static int
infer_shape_process_kwargs_i(VALUE key, VALUE val, VALUE arg)
{
  struct infer_shape_process_kwargs_params *params = (struct infer_shape_process_kwargs_params *)arg;

  if (!RB_TYPE_P(val, T_ARRAY)) {
    rb_raise(rb_eTypeError, "Arguments need to be shapes (arrays), but `%"PRIsVALUE"` is %"PRIsVALUE, key, val);
  }

  if (RB_TYPE_P(key, T_SYMBOL)) {
    key = rb_sym_to_s(key);
  }
  params->keys[params->i] = StringValueCStr(key);
  sdata_extend(params->sdata_str, params->sdata_capa, params->sdata_len, val, params->total_count - params->i);
  params->indptr[params->i + 1] = (mx_uint)*params->sdata_len;
  ++params->i;

  return ST_CONTINUE;
}

static inline VALUE
make_shape_array(mx_uint size, mx_uint const *ndim, mx_uint const **data)
{
  VALUE shapes;
  mx_uint i, j;

  shapes = rb_ary_new_capa(size);
  for (i = 0; i < size; ++i) {
    VALUE shape = rb_ary_new_capa(ndim[i]);
    for (j = 0; j < ndim[i]; ++j) {
      rb_ary_push(shape, UINT2NUM(data[i][j]));
    }
    rb_ary_push(shapes, shape);
  }

  return shapes;
}

/* The actual implementation for calling shape inference API. */
static VALUE
symbol_infer_shape_impl(int argc, VALUE *argv, VALUE obj)
{
  SymbolHandle handle;
  VALUE partial, args, kwargs;
  VALUE indptr_str, sdata_str, keys_str;
  mx_uint *indptr, *sdata;
  char const **keys;
  long i, args_len, sdata_capa, sdata_len;
  mx_uint arg_shape_size, out_shape_size, aux_shape_size;
  mx_uint const *arg_shape_ndim, *out_shape_ndim, *aux_shape_ndim;
  mx_uint const **arg_shape_data, **out_shape_data, **aux_shape_data;
  int complete;

  int (* infer_func)(
      SymbolHandle, mx_uint, char const **, mx_uint const *, mx_uint const *,
      mx_uint *, mx_uint const **, mx_uint const ***,
      mx_uint *, mx_uint const **, mx_uint const ***,
      mx_uint *, mx_uint const **, mx_uint const ***,
      int *);

  rb_scan_args(argc, argv, "1*:", &partial, &args, &kwargs);

  if (RARRAY_LEN(args) != 0 && RHASH_SIZE(kwargs) != 0) {
    rb_raise(rb_eArgError, "Can only specify known argument shapes either by positional or kwargs way.");
  }

  if (RARRAY_LEN(args) != 0) {
    args_len = RARRAY_LEN(args);

    keys = NULL;

    indptr_str = rb_str_tmp_new(sizeof(mx_uint) * (args_len + 1));
    indptr = (mx_uint *)RSTRING_PTR(indptr_str);
    indptr[0] = 0;

    sdata_capa = RARRAY_LEN(args) * 3;
    sdata_str = rb_str_tmp_new(sizeof(mx_uint) * sdata_capa);
    sdata = (mx_uint *)RSTRING_PTR(sdata_str);
    sdata_len = 0;

    for (i = 0; i < RARRAY_LEN(args); ++i) {
      VALUE s = RARRAY_AREF(args, i);
      if (!NIL_P(s)) {
        if (!RB_TYPE_P(s, T_ARRAY)) {
          rb_raise(rb_eTypeError, "Arguments need to be shapes (arrays), but argument %ld is %"PRIsVALUE, i, s);
        }
        sdata_extend(sdata_str, &sdata_capa, &sdata_len, s, RARRAY_LEN(args) - i);
      }
      indptr[i + 1] = (mx_uint)sdata_len;
    }
  }
  else {
    struct infer_shape_process_kwargs_params params;

    args_len = RHASH_SIZE(kwargs);

    keys_str = rb_str_tmp_new(sizeof(char const *) * RHASH_SIZE(kwargs));
    keys = (char const **)RSTRING_PTR(keys_str);

    indptr_str = rb_str_tmp_new(sizeof(mx_uint) * (args_len + 1));
    indptr = (mx_uint *)RSTRING_PTR(indptr_str);
    indptr[0] = 0;

    sdata_capa = RHASH_SIZE(kwargs) * 3;
    sdata_str = rb_str_tmp_new(sizeof(mx_uint) * sdata_capa);
    sdata = (mx_uint *)RSTRING_PTR(sdata_str);
    sdata_len = 0;

    params.keys = keys;
    params.indptr = indptr;
    params.sdata_str = sdata_str;
    params.sdata_capa = &sdata_capa;
    params.sdata_len = &sdata_len;
    params.i = 0;
    params.total_count = RHASH_SIZE(kwargs);

    rb_hash_foreach(kwargs, infer_shape_process_kwargs_i, (VALUE)&params);

    sdata = (mx_uint *)RSTRING_PTR(sdata_str);
  }

  handle = mxnet_get_handle(obj);

  if (RTEST(partial)) {
    infer_func = MXNET_API(MXSymbolInferShapePartial);
  }
  else {
    infer_func = MXNET_API(MXSymbolInferShape);
  }
  CHECK_CALL(infer_func(
        handle,
        (mx_uint)args_len,
        keys,
        indptr,
        sdata,
        &arg_shape_size,
        &arg_shape_ndim,
        &arg_shape_data,
        &out_shape_size,
        &out_shape_ndim,
        &out_shape_data,
        &aux_shape_size,
        &aux_shape_ndim,
        &aux_shape_data,
        &complete));

  if (complete != 0) {
    VALUE arg_shapes, out_shapes, aux_shapes;
    arg_shapes = make_shape_array(arg_shape_size, arg_shape_ndim, arg_shape_data);
    out_shapes = make_shape_array(out_shape_size, out_shape_ndim, out_shape_data);
    aux_shapes = make_shape_array(aux_shape_size, aux_shape_ndim, aux_shape_data);
    return rb_ary_new_from_args(3, arg_shapes, out_shapes, aux_shapes);
  }
  else {
    return rb_ary_new_from_args(3, Qnil, Qnil, Qnil);
  }
}

void
mxnet_init_symbol(void)
{
  VALUE cSymbol;

  cSymbol = rb_const_get_at(mxnet_mMXNet, rb_intern("Symbol"));

  rb_define_singleton_method(cSymbol, "load", symbol_s_load, 1);
  rb_define_singleton_method(cSymbol, "load_json", symbol_s_load_json, 1);

  rb_define_method(cSymbol, "initialize", symbol_initialize, 1);
  rb_define_method(cSymbol, "name", symbol_get_name, 0);
  rb_define_method(cSymbol, "list_arguments", symbol_list_arguments, 0);
  rb_define_method(cSymbol, "list_auxiliary_states", symbol_list_auxiliary_states, 0);
  rb_define_method(cSymbol, "list_outputs", mxnet_symbol_list_outputs, 0);
  rb_define_method(cSymbol, "infer_type", symbol_infer_type, -1);
  rb_define_method(cSymbol, "save", symbol_save, 1);
  rb_define_method(cSymbol, "to_json", symbol_to_json, 0);
  rb_define_method(cSymbol, "bind", symbol_bind, -1);
  rb_define_method(cSymbol, "dup", symbol_dup, 0);

  rb_define_private_method(cSymbol, "infer_shape_impl", symbol_infer_shape_impl, -1);

  mxnet_cSymbol = cSymbol;
}
