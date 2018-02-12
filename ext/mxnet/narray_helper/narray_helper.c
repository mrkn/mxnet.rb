#include "../mxnet_internal.h"

#include <numo/narray.h>
#include <limits.h>

static VALUE
ndarray_to_narray(VALUE obj)
{
  NDArrayHandle handle;
  mx_uint mx_ndim;
  mx_uint const* mx_shape;
  int mx_dtype_id, na_ndim, i;
  VALUE nary_type, nary, na_shape_str;
  size_t *na_shape, na_size;
  char *na_ptr;

  handle = mxnet_get_handle(obj);
  CHECK_CALL(MXNET_API(MXNDArrayGetShape)(handle, &mx_ndim, &mx_shape));
  CHECK_CALL(MXNET_API(MXNDArrayGetDType)(handle, &mx_dtype_id));

  if (INT_MAX < mx_ndim) {
    rb_raise(rb_eRuntimeError, "The number of dimensions is too large for Numo::NArray");
  }
  na_ndim = (int)mx_ndim;

  switch (mx_dtype_id) {
    case kFloat32:
      nary_type = numo_cSFloat;
      break;
    case kFloat64:
      nary_type = numo_cDFloat;
      break;
    case kFloat16:
      nary_type = numo_cSFloat;
      break;
    case kUint8:
      nary_type = numo_cUInt8;
      break;
    case kInt32:
      nary_type = numo_cInt32;
      break;
    case kInt8:
      nary_type = numo_cInt8;
      break;
    case kInt64:
      nary_type = numo_cInt64;
      break;
    default:
      rb_raise(rb_eRuntimeError, "Unknown dtype of MXNet::NDArray (%d)", mx_dtype_id);
  }

  na_shape_str = rb_str_tmp_new(sizeof(size_t) * na_ndim);
  na_shape = (size_t *)RSTRING_PTR(na_shape_str);
  for (i = 0; i < na_ndim; ++i) {
    na_shape[i] = mx_shape[i];
  }
  nary = nary_new(nary_type, na_ndim, na_shape);
  na_ptr = nary_get_pointer_for_write(nary);
  na_size = RNARRAY_SIZE(nary);

  CHECK_CALL(MXNET_API(MXNDArraySyncCopyToCPU)(handle, na_ptr, na_size));

  return nary;
}

static VALUE
m_sync_copyfrom(VALUE mod, VALUE nd_obj, VALUE nary)
{
  NDArrayHandle handle;
  narray_t *na;
  void *data;
  size_t size;

  mxnet_check_ndarray(nd_obj);

  if (!RTEST(nary_check_contiguous(nary))) {
    nary = nary_dup(nary);
  }

  GetNArray(nary, na);
  data = NA_DATA_PTR(na);
  size = NA_SIZE(na);

  handle = mxnet_get_handle(nd_obj);
  CHECK_CALL(MXNET_API(MXNDArraySyncCopyFromCPU)(handle, data, size));

  return nd_obj;
}

void
Init_narray_helper(void)
{
  VALUE mHelper;

  rb_undef_method(mxnet_cNDArray, "to_narray");
  rb_define_method(mxnet_cNDArray, "to_narray", ndarray_to_narray, 0);

  mHelper = rb_define_module_under(mxnet_mMXNet, "NArrayHelper");
  rb_define_module_function(mHelper, "sync_copyfrom", m_sync_copyfrom, 2);
}
