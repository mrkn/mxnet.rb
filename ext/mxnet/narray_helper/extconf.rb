require 'mkmf'

require 'numo/narray'
$LOAD_PATH.each do |x|
  if File.exist? File.join(x, 'numo/numo/narray.h')
    $INCFLAGS = "-I#{x}/numo " + $INCFLAGS
    break
  end
end

create_makefile('mxnet/narray_helper')
