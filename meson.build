project('pyseldonlib', 'cpp', 
    version: '1.0.0',
    default_options: ['cpp_std=c++20', 'warning_level=3'])

_deps = []

cppc = meson.get_compiler('cpp')
seldon_subproj = subproject('seldon', default_options: ['default_library=static','build_tests=false', 
                                                        'build_exe=false'])
seldon_dep = seldon_subproj.get_variable('seldon_static_dep')

_deps += seldon_dep

py = import('python').find_installation('python')
_deps += py.dependency()

pyb11_dep = [
  py.dependency(),
  dependency('pybind11')
]
_deps += pyb11_dep

py.extension_module(
  'seldoncore',
  sources : [
    'python_bindings/bindings.cpp'
  ],
  dependencies: _deps,
  install: true,
  subdir: 'bindings/',
)

py.install_sources(
   [
    'pyseldonlib/__init__.py',
    'pyseldonlib/_basemodel.py',
    'pyseldonlib/DeGrootModel.py',
    'pyseldonlib/DeffuantModel.py',
    'pyseldonlib/DeffuantVectorModel.py',
    'pyseldonlib/ActivityDrivenModel.py',
    'pyseldonlib/InertialModel.py',
    'pyseldonlib/network.py',
    'pyseldonlib/_othersettings.py',
    'pyseldonlib/_run_simulation.py',
    'pyseldonlib/utils.py',

  ],
  pure: false, # do not install next to compiled extension
  subdir: 'pyseldonlib'
)
