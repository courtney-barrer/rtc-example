from conan import ConanFile
from conan.tools.files import copy

class SardinePythonConan(ConanFile):
    name = 'baldr-python'
    version = '1.0.0'
    license = ''

    settings = 'os', 'compiler', 'build_type', 'arch'

    default_options = {
        # 'emu/*:cuda'   : False, # Should not be necessary, but it is.
        'boost/*:namespace' : 'myboost',
        # 'emu/*:python' : True, # Should not be necessary, but it is.
    }

    requires = [
        'baldr/1.0.0',
        'boost/1.84.0',
        'fmt/11.0.0',
    ]



    def layout(self):
        self.folders.generators = "generators"

    generators = 'CMakeDeps'

    def generate(self):
        for dep in self.dependencies.values():
            for libdir in dep.cpp_info.libdirs:
                copy(self, "*.so*", libdir, self.build_folder)
