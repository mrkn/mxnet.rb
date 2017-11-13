#include "mxnet_internal.h"

static VALUE
random_m_set_seed(VALUE mod, VALUE seed_state)
{
  seed_state = rb_check_to_int(seed_state);
  CHECK_CALL(MXNET_API(MXRandomSeed)(NUM2INT(seed_state)));
  return Qnil;
}

void
mxnet_init_random(void)
{
  VALUE mRandom;

  mRandom = rb_define_module_under(mxnet_mMXNet, "Random");
  rb_define_module_function(mRandom, "seed", random_m_set_seed, 1);
}
