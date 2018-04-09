#include "mxnet_internal.h"

VALUE mxnet_cKVStore;

static void
kvstore_free(void *ptr)
{
  if (ptr != NULL) {
    CHECK_CALL(MXNET_API(MXKVStoreFree)((KVStoreHandle)ptr));
  }
}

static size_t
kvstore_memsize(void const *ptr)
{
  return 0;
}

static const rb_data_type_t kvstore_data_type = {
  "MXNet::KVStore",
  {
    NULL,
    kvstore_free,
    kvstore_memsize,
  },
  0, 0, RUBY_TYPED_FREE_IMMEDIATELY
};

int
mxnet_obj_is_kvstore(VALUE obj)
{
  return rb_typeddata_is_kind_of(obj, &kvstore_data_type);
}

KVStoreHandle
mxnet_kvstore_get_handle(VALUE obj)
{
  KVStoreHandle handle;
  TypedData_Get_Struct(obj, void, &kvstore_data_type, handle);
  return handle;
}

static VALUE
kvstore_allocate(VALUE klass)
{
  return TypedData_Wrap_Struct(klass, &kvstore_data_type, NULL);
}

static VALUE
kvstore_initialize(int argc, VALUE *argv, VALUE obj)
{
  VALUE name;
  char const *name_cstr;
  KVStoreHandle handle;

  if (rb_scan_args(argc, argv, "01", &name) == 0) {
    name = rb_str_new2("local");
  }
  name_cstr = StringValueCStr(name);
  CHECK_CALL(MXNET_API(MXKVStoreCreate)(name_cstr, &handle));
  DATA_PTR(obj) = handle;
  return obj;
}

static VALUE
kvstore_eq(VALUE obj, VALUE other)
{
  if (mxnet_obj_is_kvstore(other) && DATA_PTR(obj) == DATA_PTR(other))
    return Qtrue;
  return Qfalse;
}

static VALUE
kvstore_get_type(VALUE obj)
{
  char const *kv_type;
  KVStoreHandle handle;

  handle = mxnet_kvstore_get_handle(obj);
  CHECK_CALL(MXNET_API(MXKVStoreGetType)(handle, &kv_type));

  return rb_str_new2(kv_type);
}

static int
convert_multiple_key_value(VALUE keys, VALUE vals, mx_uint *num_keys, VALUE *keys_bufstr, VALUE *vals_bufstr, VALUE *gc_guard)
{
  long n, i, k;
  int str_key_p;

  assert(gc_guard != NULL);
  assert(RB_TYPE_P(*gc_guard, T_ARRAY));

  keys = rb_check_array_type(keys);
  vals = rb_check_array_type(vals);

  n = RARRAY_LEN(keys);
  if (n != RARRAY_LEN(vals)) {
    rb_raise(rb_eArgError, "key value lengths mismatched");
  }
#if SIZEOF_LONG > SIZEOF_INT
  if (n > UINT_MAX) {
    rb_raise(rb_eArgError, "key value lengths too large");
  }
#endif
  *num_keys = (mx_uint)n;

  *keys_bufstr = rb_str_tmp_new(sizeof(void *) * n);
  *vals_bufstr = rb_str_tmp_new(sizeof(void *) * n);

  str_key_p = RB_TYPE_P(RARRAY_AREF(keys, 0), T_STRING);

  k = 0;
  for (i = 0; i < n; ++i) {
    VALUE key, val;
    key = RARRAY_AREF(keys, i);
    val = RARRAY_AREF(vals, i);

    if (RB_TYPE_P(key, T_SYMBOL)) {
      key = rb_sym_to_s(key);
      rb_ary_push(*gc_guard, key);
    }
    if (str_key_p != RB_TYPE_P(key, T_STRING)) {
      rb_raise(rb_eArgError, "inconsistent types of keys detected.");
    }

    if (RTEST(rb_obj_is_kind_of(val, mxnet_cNDArray))) {
      if (str_key_p) {
        ((char const **)RSTRING_PTR(*keys_bufstr))[k] = StringValueCStr(key);
      }
      else {
        ((int *)RSTRING_PTR(*keys_bufstr))[k] = NUM2INT(key);
      }
      ((NDArrayHandle *)RSTRING_PTR(*vals_bufstr))[k] = mxnet_ndarray_get_handle(val);
      ++k;
    }
    else {
      long j, n2;
      val = rb_check_array_type(val);
      n2 = RARRAY_LEN(val);
#if SIZEOF_LONG > SIZEOF_INT
      if (n2 > UINT_MAX) {
        rb_raise(rb_eArgError, "values array is too long");
      }
#endif

      *num_keys += n2 - 1;
      rb_str_resize(*keys_bufstr, sizeof(void *) * *num_keys);
      rb_str_resize(*vals_bufstr, sizeof(void *) * *num_keys);
      if (str_key_p) {
        char const *key_cstr = StringValueCStr(key);
        for (j = 0; j < n2; ++j) {
          ((char const **)RSTRING_PTR(*keys_bufstr))[k] = key_cstr;
          ((NDArrayHandle *)RSTRING_PTR(*vals_bufstr))[k] = mxnet_ndarray_get_handle(RARRAY_AREF(val, j));
          ++k;
        }
      }
      else {
        int key_cint = NUM2INT(key);
        for (j = 0; j < n2; ++j) {
          ((int *)RSTRING_PTR(*keys_bufstr))[k] = key_cint;
          ((NDArrayHandle *)RSTRING_PTR(*vals_bufstr))[k] = mxnet_ndarray_get_handle(RARRAY_AREF(val, j));
          ++k;
        }
      }
    }
  }

  return str_key_p;
}

static int
convert_key_value(VALUE key, VALUE val, mx_uint *num_keys, VALUE *keys_bufstr, VALUE *vals_bufstr, VALUE *gc_guard)
{
  int str_key_p;

  assert(gc_guard != NULL);
  *gc_guard = rb_ary_new();

  if (RB_TYPE_P(key, T_ARRAY)) {
    return convert_multiple_key_value(key, val, num_keys, keys_bufstr, vals_bufstr, gc_guard);
  }

  if (RB_TYPE_P(key, T_SYMBOL)) {
    key = rb_sym_to_s(key);
    rb_ary_push(*gc_guard, key);
  }
  str_key_p = RB_TYPE_P(key, T_STRING);
  if (!RB_INTEGER_TYPE_P(key) && !str_key_p) {
    rb_raise(rb_eArgError, "unexpected type for keys: %s", rb_class2name(CLASS_OF(key)));
  }

  if (RTEST(rb_obj_is_kind_of(val, mxnet_cNDArray))) { /* TODO: mxnet_obj_is_ndarray */
    *num_keys = 1;
    *keys_bufstr = rb_str_tmp_new(sizeof(void *));
    *vals_bufstr = rb_str_tmp_new(sizeof(void *));
    if (str_key_p) {
      ((char const **)RSTRING_PTR(*keys_bufstr))[0] = StringValueCStr(key);
    }
    else {
      ((int *)RSTRING_PTR(*keys_bufstr))[0] = NUM2INT(key);
    }
    ((NDArrayHandle *)RSTRING_PTR(*vals_bufstr))[0] = mxnet_ndarray_get_handle(val);
  }
  else {
    long i, n;

    val = rb_check_array_type(val);
    n = RARRAY_LEN(val);
#if SIZEOF_LONG > SIZEOF_INT
    if (n > UINT_MAX) {
      rb_raise(rb_eArgError, "values array is too long");
    }
#endif

    *num_keys = (mx_uint)n;
    for (i = 0; i < n; ++i) {
      mxnet_check_ndarray(RARRAY_AREF(val, i));
    }

    *keys_bufstr = rb_str_tmp_new(sizeof(void *) * n);
    *vals_bufstr = rb_str_tmp_new(sizeof(void *) * n);

    if (str_key_p) {
      char const *key_cstr = StringValueCStr(key);
      for (i = 0; i < n; ++i) {
        ((char const **)RSTRING_PTR(*keys_bufstr))[i] = key_cstr;
        ((NDArrayHandle *)RSTRING_PTR(*vals_bufstr))[i] = mxnet_ndarray_get_handle(RARRAY_AREF(val, i));
      }
    }
    else {
      int key_cint = NUM2INT(key);
      for (i = 0; i < n; ++i) {
        ((int *)RSTRING_PTR(*keys_bufstr))[i] = key_cint;
        ((NDArrayHandle *)RSTRING_PTR(*vals_bufstr))[i] = mxnet_ndarray_get_handle(RARRAY_AREF(val, i));
      }
    }
  }

  return str_key_p;
}

static VALUE
kvstore_init(VALUE obj, VALUE key, VALUE value)
{
  KVStoreHandle handle;
  mx_uint num_keys;
  int str_key_p;
  VALUE keys_bufstr, vals_bufstr, gc_guard;
  NDArrayHandle *vals_cptr;

  str_key_p = convert_key_value(key, value, &num_keys, &keys_bufstr, &vals_bufstr, &gc_guard);
  vals_cptr = (NDArrayHandle *)RSTRING_PTR(vals_bufstr);

  handle = mxnet_kvstore_get_handle(obj);
  if (str_key_p) {
    char const **keys_cptr = (char const **)RSTRING_PTR(keys_bufstr);
    CHECK_CALL(MXNET_API(MXKVStoreInitEx)(handle, num_keys, keys_cptr, vals_cptr));
  }
  else {
    int *keys_cptr = (int *)RSTRING_PTR(keys_bufstr);
    CHECK_CALL(MXNET_API(MXKVStoreInit)(handle, num_keys, keys_cptr, vals_cptr));
  }

  rb_str_resize(keys_bufstr, 0);
  rb_str_resize(vals_bufstr, 0);
  return Qnil;
}

static VALUE
kvstore_push(int argc, VALUE *argv, VALUE obj)
{
  KVStoreHandle handle;
  mx_uint num_keys;
  int str_key_p, priority = 0;
  VALUE key, value, opts;
  VALUE keys_bufstr, vals_bufstr, gc_guard;
  NDArrayHandle *vals_cptr;

  rb_scan_args(argc, argv, "2:", &key, &value, &opts);
  if (!NIL_P(opts)) {
    static ID keywords[1];
    VALUE kwargs[1];
    if (!keywords[0]) {
      keywords[0] = rb_intern("priority");
    }
    rb_get_kwargs(opts, keywords, 0, 1, kwargs);
    if (kwargs[0] != Qundef) {
      priority = NUM2INT(kwargs[0]);
    }
  }

  str_key_p = convert_key_value(key, value, &num_keys, &keys_bufstr, &vals_bufstr, &gc_guard);
  vals_cptr = (NDArrayHandle *)RSTRING_PTR(vals_bufstr);

  handle = mxnet_kvstore_get_handle(obj);
  if (str_key_p) {
    char const **keys_cptr = (char const **)RSTRING_PTR(keys_bufstr);
    CHECK_CALL(MXNET_API(MXKVStorePushEx)(handle, num_keys, keys_cptr, vals_cptr, priority));
  }
  else {
    int *keys_cptr = (int *)RSTRING_PTR(keys_bufstr);
    CHECK_CALL(MXNET_API(MXKVStorePush)(handle, num_keys, keys_cptr, vals_cptr, priority));
  }

  rb_str_resize(keys_bufstr, 0);
  rb_str_resize(vals_bufstr, 0);
  return Qnil;
}

static VALUE
kvstore_pull(int argc, VALUE *argv, VALUE obj)
{
  KVStoreHandle handle;
  mx_uint num_keys;
  int str_key_p, priority = 0;
  VALUE key, out = Qundef, opts;
  VALUE keys_bufstr, vals_bufstr, gc_guard;
  NDArrayHandle *vals_cptr;

  rb_scan_args(argc, argv, "1:", &key, &opts);
  if (!NIL_P(opts)) {
    static ID keywords[2];
    VALUE kwargs[2];
    if (!keywords[0]) {
      keywords[0] = rb_intern("out");
      keywords[1] = rb_intern("priority");
    }
    rb_get_kwargs(opts, keywords, 1, 1, kwargs);
    out = kwargs[0];
    if (kwargs[1] != Qundef) {
      priority = NUM2INT(kwargs[0]);
    }
  }
  if (out == Qundef) {
    rb_raise(rb_eArgError, ":out keyword argument must be supplied");
  }

  str_key_p = convert_key_value(key, out, &num_keys, &keys_bufstr, &vals_bufstr, &gc_guard);
  vals_cptr = (NDArrayHandle *)RSTRING_PTR(vals_bufstr);

  handle = mxnet_kvstore_get_handle(obj);
  if (str_key_p) {
    char const **keys_cptr = (char const **)RSTRING_PTR(keys_bufstr);
    CHECK_CALL(MXNET_API(MXKVStorePullEx)(handle, num_keys, keys_cptr, vals_cptr, priority));
  }
  else {
    int *keys_cptr = (int *)RSTRING_PTR(keys_bufstr);
    CHECK_CALL(MXNET_API(MXKVStorePull)(handle, num_keys, keys_cptr, vals_cptr, priority));
  }

  rb_str_resize(keys_bufstr, 0);
  rb_str_resize(vals_bufstr, 0);
  return Qnil;
}

static int
extract_hash_params_i(VALUE key, VALUE val, VALUE arg)
{
  VALUE *args = (VALUE *)arg;
  char const **key_cptr = (char const **)args[0];
  char const **val_cptr = (char const **)args[1];
  VALUE gc_guard = args[2];

  if (RB_TYPE_P(key, T_SYMBOL)) {
    key = rb_sym_to_s(key);
    rb_ary_push(gc_guard, key);
  }
  if (RB_TYPE_P(val, T_SYMBOL)) {
    val = rb_sym_to_s(val);
    rb_ary_push(gc_guard, val);
  }
  *key_cptr = StringValueCStr(key);
  *val_cptr = StringValueCStr(val);
  ++key_cptr;
  ++val_cptr;
  return ST_CONTINUE;
}

static void
extract_hash_params(VALUE params, mx_uint *num_params, VALUE *keys_bufstr, VALUE *vals_bufstr, VALUE *gc_guard)
{
  long n;
  VALUE args[3];

  params = rb_check_hash_type(params);
  n = RHASH_SIZE(params);
#if SIZEOF_LONG > SIZEOF_INT
  if (n > UINT_MAX) {
    rb_raise(rb_eArgError, "too much parameters given");
  }
#endif
  *num_params = (mx_uint)n;

  args[0] = (VALUE)RSTRING_PTR(keys_bufstr);
  args[1] = (VALUE)RSTRING_PTR(vals_bufstr);
  args[2] = *gc_guard = rb_ary_new();

  rb_hash_foreach(params, extract_hash_params_i, (VALUE)args);
}

static VALUE
kvstore_set_gradient_compression(VALUE obj, VALUE compression_params)
{
  KVStoreHandle handle;
  VALUE kv_type;
  char const *kv_type_cstr;
  VALUE keys_bufstr, vals_bufstr, gc_guard;
  mx_uint num_params;
  char const **keys_cptr, **vals_cptr;

  kv_type = kvstore_get_type(obj);
  kv_type_cstr = StringValueCStr(kv_type);
  if (strncmp(kv_type_cstr, "device", 6) != 0 && strncmp(kv_type_cstr, "dist", 4) != 0) {
    rb_raise(rb_eRuntimeError, "Gradient compression is not supported for this type of kvstore");
  }

  extract_hash_params(compression_params, &num_params, &keys_bufstr, &vals_bufstr, &gc_guard);

  handle = mxnet_kvstore_get_handle(obj);
  keys_cptr = (char const **)RSTRING_PTR(keys_bufstr);
  vals_cptr = (char const **)RSTRING_PTR(vals_bufstr);
  CHECK_CALL(MXNET_API(MXKVStoreSetGradientCompression)(handle, num_params, keys_cptr, vals_cptr));
  return Qnil;
}

static VALUE
kvstore_set_optimizer(VALUE obj, VALUE optimizer)
{

}

static void
kvstore_updater_caller(VALUE key, NDArrayHandle recv_nd, NDArrayHandle local_nd, VALUE obj)
{
  static ID id_updater;
  KVStoreHandle handle;
  VALUE updater;

  if (id_updater == 0) {
    id_updater = rb_intern("updater");
  }

  updater = rb_ivar_get(obj, id_updater);

  if (!NIL_P(updater)) {
    VALUE recv, local;

    recv = mxnet_ndarray_new(recv_nd);
    local = mxnet_ndarray_new(local_nd);

    rb_funcall(updater, rb_intern("call"), 3, key, recv, local);
  }
}

static void
kvstore_int_updater_caller(int key_cint, NDArrayHandle recv_nd, NDArrayHandle local_nd, void *handle)
{
  VALUE key, obj;
  key = INT2NUM(key_cint);
  obj = (VALUE)handle;
  kvstore_updater_caller(key, recv_nd, local_nd, obj);
}

static void
kvstore_str_updater_caller(char const *key_cstr, NDArrayHandle recv_nd, NDArrayHandle local_nd, void *handle)
{
  VALUE key, obj;
  key = rb_str_new_static(key_cstr, strlen(key_cstr));
  obj = (VALUE)handle;
  kvstore_updater_caller(key, recv_nd, local_nd, obj);
}

static VALUE
kvstore_set_updater(VALUE obj, VALUE updater)
{
  static ID id_updater;
  KVStoreHandle handle;

  if (id_updater == 0) {
    id_updater = rb_intern("updater");
  }

  rb_ivar_set(obj, id_updater, updater);

  handle = mxnet_kvstore_get_handle(obj);
  MXNET_API(MXKVStoreSetUpdaterEx)(handle, kvstore_int_updater_caller, kvstore_str_updater_caller, obj);
}

static VALUE
kvstore_save_optimizer_states(int argc, VALUE *argv, VALUE obj)
{}

static VALUE
kvstore_load_optimizer_states(VALUE obj, VALUE fname)
{}

void
mxnet_init_kvstore(void)
{
  VALUE cKVStore;

  cKVStore = rb_const_get_at(mxnet_mMXNet, rb_intern("KVStore"));

  rb_define_alloc_func(cKVStore, kvstore_allocate);
  rb_define_method(cKVStore, "initialize", kvstore_initialize, -1);
  rb_define_method(cKVStore, "==", kvstore_eq, 1);
  rb_define_method(cKVStore, "type", kvstore_get_type, 0);
  /* TODO: rb_define_method(cKVStore, "rank", kvstore_get_rank, 0); */
  /* TODO: rb_define_method(cKVStore, "num_workers", kvstore_get_num_workers, 0); */
  rb_define_method(cKVStore, "init", kvstore_init, 2);
  rb_define_method(cKVStore, "push", kvstore_push, -1);
  rb_define_method(cKVStore, "pull", kvstore_pull, -1);
  rb_define_method(cKVStore, "gradient_compression=", kvstore_set_gradient_compression, 1);
  rb_define_method(cKVStore, "optimizer=", kvstore_set_optimizer, 1);
  rb_define_method(cKVStore, "updater=", kvstore_set_updater, 1);
  /* TODO: rb_define_method(cKVStore, "barrier", kvstore_barrier, 0); */
  /* TODO: rb_define_method(cKVStore, "send_command_to_servers", kvstore_send_command_to_servers, 2); */
  rb_define_method(cKVStore, "save_optimizer_states", kvstore_save_optimizer_states, -1);
  rb_define_method(cKVStore, "load_optimizer_states", kvstore_load_optimizer_states, 1);

  mxnet_cKVStore = cKVStore;
}
