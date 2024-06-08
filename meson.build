project('pyseldon', 'cpp', 
    default_options: ['cpp_std=c++20'])

args = []
deps = []

# todo change the repository name in wrap
seldon_subproj = subproject('seldon', default_options: ['build_tests=false', 
                                                        'build_exe=false'])
seldon_dep = seldon_subproj.get_variable('seldon_static_dep')

deps += [seldon_dep]

py_dep = import('python').find_installation('python', modules: ['numpy']).dependency()
deps += py_dep 

# todo: might require numpy dependency

pyb11_dep = [
  py.dependency(),
  dependency('pybind11')
]
deps += [pyb11_dep]

# ask: name of the module
py.extension_module(
  'seldoncore',
  sources : [
    'python_bindings/bindings.cpp'
  ],
  dependencies: deps,
  cpp_args : args,
  install: true,
  subdir: 'pyseldon/'
)

py.install_sources(
  [
    'pyseldon/__init__.py',
  ],
  pure: false, # install next to compiled extension
  subdir: 'pyseldon'
)