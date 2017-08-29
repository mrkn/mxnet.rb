require 'mkmf'

headers = []
if have_header('stdint.h')
  headers << 'stdint.h'
end
have_type('int8_t', headers)
have_type('int16_t', headers)
have_type('int32_t', headers)
have_type('int64_t', headers)

create_makefile('mxnet')
