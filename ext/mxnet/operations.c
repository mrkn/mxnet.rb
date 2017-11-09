#include "mxnet_internal.h"

VALUE mxnet_mNDArrayOps;

VALUE mxnet_sOpInfo;
VALUE mxnet_sOpArgInfo;

static ID id_handles;
static ID id_descriptions;

static VALUE
get_m_description(VALUE mod, VALUE name)
{
  VALUE hash;
  hash = rb_ivar_get(mxnet_mNDArrayOps, id_handles);
  StringValue(name);
  return rb_hash_aref(hash, rb_to_symbol(name));
}

static void
register_handle(VALUE mod, char const* name, void *handle)
{
  VALUE hash = rb_ivar_get(mod, id_handles);
  rb_hash_aset(hash, ID2SYM(rb_intern(name)), PTR2NUM(handle));
}

static void
register_description(VALUE mod, char const *name, VALUE description)
{
  VALUE hash = rb_ivar_get(mod, id_handles);
  rb_hash_aset(hash, ID2SYM(rb_intern(name)), description);
}

static VALUE
op_arg_info_new(char const *name, char const *type_info, char const *description)
{
  return rb_struct_new(mxnet_sOpArgInfo, ID2SYM(rb_intern(name)), rb_str_new2(type_info), rb_str_new2(description), 0);
}

static VALUE
op_info_new(char const *name, char const *description, mx_uint num_args,
            char const **arg_names, char const **arg_type_infos,
            char const **arg_descriptions, char const *key_var_num_args,
            char const *return_type)
{
  mx_uint i;
  VALUE args;

  args = rb_ary_new_capa(num_args);
  for (i = 0; i < num_args; ++i) {
    rb_ary_push(args, op_arg_info_new(arg_names[i], arg_type_infos[i], arg_descriptions[i]));
  }

  return rb_struct_new(mxnet_sOpInfo,
    ID2SYM(rb_intern(name)),
    rb_str_new2(description),
    args,
    key_var_num_args ? ID2SYM(rb_intern(key_var_num_args)) : Qnil,
    return_type ? rb_str_new2(return_type) : Qnil,
    0);
}

static void
setup_operation(VALUE mod, VALUE name)
{
  void *op_handle;
  char const *real_name, *description, **arg_names, **arg_type_infos, **arg_descriptions, *key_var_num_args, *return_type;
  mx_uint num_args;
  VALUE op_info;

  CHECK_CALL(MXNET_API(NNGetOpHandle)(RSTRING_PTR(name), &op_handle)); /* check handle availability just in case */

  CHECK_CALL(MXNET_API(MXSymbolGetAtomicSymbolInfo)(
      op_handle, &real_name, &description,
      &num_args, &arg_names, &arg_type_infos, &arg_descriptions,
      &key_var_num_args, &return_type));

  register_handle(mod, real_name, op_handle);

  op_info = op_info_new(real_name, description, num_args, arg_names, arg_type_infos,
                        arg_descriptions, key_var_num_args, return_type);
  register_description(mod, real_name, op_info);

  rb_funcall(mxnet_mUtils, rb_intern("define_operation_delegator"), 3, mod, PTR2NUM(op_handle), op_info);
}

static VALUE
list_all_op_names(void)
{
  mx_uint size, i;
  char const** op_names;
  VALUE ary;

  CHECK_CALL(MXNET_API(MXListAllOpNames)(&size, &op_names));
  ary = rb_ary_new_capa((long)size);
  for (i = 0; i < size; ++i) {
    rb_ary_push(ary, rb_str_new2(op_names[i]));
  }

  return ary;
}

void
mxnet_init_operations(void)
{
  VALUE mOps;
  long i;
  VALUE op_names;

  mOps = rb_define_module_under(mxnet_cNDArray, "Ops");

  mxnet_sOpInfo = rb_const_get_at(mxnet_mMXNet, rb_intern("OpInfo"));
  mxnet_sOpArgInfo = rb_const_get_at(mxnet_mMXNet, rb_intern("OpArgInfo"));

  id_handles = rb_intern("handles");
  id_descriptions = rb_intern("descriptions");

  rb_ivar_set(mOps, id_handles, rb_hash_new());
  rb_ivar_set(mOps, id_descriptions, rb_hash_new());

  op_names = list_all_op_names();
  for (i = 0; i < RARRAY_LEN(op_names); ++i) {
    setup_operation(mOps, RARRAY_AREF(op_names, i));
  }

  rb_define_module_function(mOps, "description", get_m_description, 1);

  mxnet_mNDArrayOps = mOps;
}
