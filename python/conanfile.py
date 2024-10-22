from conan import ConanFile

class SardinePythonConan(ConanFile):
    name = 'baldr-python'
    version = '1.0.0'
    license = ''

    settings = 'os', 'compiler', 'build_type', 'arch'

    default_options = {
        # 'emu/*:cuda'   : False, # Should not be necessary, but it is.
        'emu/*:python' : True, # Should not be necessary, but it is.
    }

    requires = [
        'emu/1.0.0',
        'baldr/1.0.0'
        # 'pybind11/2.10.4',
    ]

    generators = 'CMakeDeps'
