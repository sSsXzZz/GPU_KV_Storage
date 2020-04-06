import os
import ycm_core

DIR_OF_THIS_SCRIPT = os.path.abspath( os.path.dirname( __file__ ) )

includes = ['-I/usr/local/cuda-10.2/include/']

common = ['-std=c++14',
          '-DUSE_CLANG_COMPLETER']
cpp_flags = ['-x', 'c++',]

# http://llvm.org/docs/CompileCudaWithLLVM.html
cuda_flags = ['-x', 'cuda', '--cuda-gpu-arch=sm_61']

def Settings(**kwargs):
  filename = kwargs['filename']

  compile_flags = cpp_flags
  if filename.endswith('.cu'):
    compile_flags = cuda_flags
  compile_flags.extend(common)
  compile_flags.extend(includes)

  return {
    'flags': compile_flags,
    'include_paths_relative_to_dir': DIR_OF_THIS_SCRIPT
  }
