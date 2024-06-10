project('pyseldon', 'cpp', 
    default_options: ['cpp_std=c++20'])

args = ['-fPIC']
deps = []


cppc = meson.get_compiler('cpp')

# todo change the repository name in wrap
seldon_subproj = subproject('seldon', default_options: ['default_library=static','build_tests=false', 
                                                        'build_exe=false'])
seldon_dep = seldon_subproj.get_variable('seldon_static_dep')

deps += [seldon_dep]

py = import('python').find_installation('python')
deps += py.dependency() 

# todo: might require numpy dependency
fmt_dep = dependency('fmt')
deps+= fmt_dep

pyb11_dep = [
  py.dependency(),
  dependency('pybind11')
]
deps += [pyb11_dep]

incdir = include_directories('subprojects/seldon/include')

# ask: name of the module
py.extension_module(
  'seldoncore',
  sources : [
    'python_bindings/bindings.cpp'
  ],
  include_directories: incdir,
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