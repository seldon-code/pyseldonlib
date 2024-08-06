project('pyseldon', 'cpp', 
    default_options: ['cpp_std=c++20', 'warning_level=3'])

_args = ['-fPIC']
_deps = []


cppc = meson.get_compiler('cpp')

# TODO(USER-DK)- change the repo name in git wrap
seldon_subproj = subproject('seldon', default_options: ['default_library=static','build_tests=false', 
                                                        'build_exe=false'])
seldon_dep = seldon_subproj.get_variable('seldon_static_dep')

_deps += seldon_dep

py = import('python').find_installation('python')
_deps += py.dependency()

# todo: might require numpy dependency
fmt_dep = dependency('fmt')
_deps += fmt_dep

pyb11_dep = [
  py.dependency(),
  dependency('pybind11')
]
_deps += pyb11_dep

# ask: name of the module
py.extension_module(
  'seldoncore',
  sources : [
    'python_bindings/bindings.cpp'
  ],
  dependencies: _deps,
  cpp_args : _args,
  install: true,
  subdir: 'bindings/',
)

py.install_sources(
   [
    'pyseldon/__init__.py',
    'pyseldon/DeGrootModel.py',
    'pyseldon/DeffuantModel.py',
    'pyseldon/DeffuantVectorModel.py',
    'pyseldon/ActivityDrivenModel.py',
    'pyseldon/InertialModel.py',
  ],
  pure: false, # install next to compiled extension
  subdir: 'pyseldon'
)