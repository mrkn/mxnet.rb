#include "mxnet_internal.h"

/* Set status to recording/not recording.
 * When recording, graph will be constructed for gradient computation.
 *
 * @param is_recording [true, false]
 * @return [true, false] The previous state before this set.
 */
static VALUE
autograd_s_set_recording(VALUE mod, VALUE is_recording)
{
  int prev;
  CHECK_CALL(MXNET_API(MXAutogradSetIsRecording)(RTEST(is_recording), &prev));
  return prev ? Qtrue : Qfalse;
}

/* Set status to training/predicting.  This affects ctx.is_train in operator
 * running context.  For example, Dropout will drop inputs randomly when
 * train_mode is true while simply passing through if train_mode is false.
 *
 * @param train_mode [true, false]
 * @return [true, false] The previous state before this set.
 */
static VALUE
autograd_s_set_training(VALUE mod, VALUE train_mode)
{
  int prev;
  CHECK_CALL(MXNET_API(MXAutogradSetIsTraining)(RTEST(train_mode), &prev));
  return prev ? Qtrue : Qfalse;
}

static VALUE
autograd_s_recording_p(VALUE mod)
{
  bool curr;
  CHECK_CALL(MXNET_API(MXAutogradIsRecording)(&curr));
  return curr ? Qtrue : Qfalse;
}

static VALUE
autograd_s_training_p(VALUE mod)
{
  bool curr;
  CHECK_CALL(MXNET_API(MXAutogradIsTraining)(&curr));
  return curr ? Qtrue : Qfalse;
}

void
mxnet_init_autograd(void)
{
  VALUE mAutograd;

  mAutograd = rb_const_get_at(mxnet_mMXNet, rb_intern("Autograd"));
  rb_define_singleton_method(mAutograd, "set_recording", autograd_s_set_recording, 1);
  rb_define_singleton_method(mAutograd, "set_training", autograd_s_set_training, 1);
  rb_define_singleton_method(mAutograd, "recording?", autograd_s_recording_p, 0);
  rb_define_singleton_method(mAutograd, "training?", autograd_s_training_p, 0);
}
