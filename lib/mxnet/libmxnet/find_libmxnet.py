import os
import platform
import sys

if sys.version_info.major >= 3:
    from importlib import util as u
    spec = u.find_spec('mxnet')
    if spec is None:
        sys.exit(1)
    mxnet_dir = os.path.dirname(spec.origin)
else:
    import imp
    try:
        spec = imp.find_module('mxnet')
        mxnet_dir = spec[1]
    except ImportError:
        sys.exit(1)

# The following algorithm is almost same as mxnet.libinfo.find_lib_path
curr_path = mxnet_dir
api_path = os.path.join(curr_path, '../../lib/')
cmake_build_path = os.path.join(curr_path, '../../build/Release/')
dll_path = [curr_path, api_path, cmake_build_path]
if os.name == 'nt':
    dll_path.append(os.path.join(curr_path, '../../build'))
    vs_configuration = 'Release'
    dll_path.append(os.path.join(curr_path, '../../build', vs_configuration))
    if platform.architecture()[0] == '64bit':
        dll_path.append(os.path.join(curr_path, '../../windows/x64', vs_configuration))
    else:
        dll_path.append(os.path.join(curr_path, '../../windows', vs_configuration))
elif os.name == 'posix' and os.environ.get('LD_LIBRARY_PATH', None):
    dll_path.extend([p.strip() for p in os.environ['LD_LIBRARY_PATH'].split(':')])

if os.name == 'nt':
    os.environ['PATH'] = mxnet_dir + ';' + os.environ['PATH']
    dll_path = [os.path.join(p, 'libmxnet.dll') for p in dll_path]
elif platform.system() == 'Darwin':
    dll_path = [os.path.join(p, 'libmxnet.dylib') for p in dll_path] + \
               [os.path.join(p, 'libmxnet.so')    for p in dll_path]
else:
    dll_path.append('../../../')
    dll_path = [os.path.join(p, 'libmxnet.so') for p in dll_path]

lib_path = [p for p in dll_path if os.path.exists(p) and os.path.isfile(p)]
if len(lib_path) > 0:
    print(lib_path[0])
