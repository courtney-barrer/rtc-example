from conan import ConanFile
from conan.tools.cmake import cmake_layout
from conan.tools.files import copy

class SardinePythonConan(ConanFile):
    name = 'sardine-python'
    version = '1.0.0'
    license = ''

    settings = 'os', 'compiler', 'build_type', 'arch'

    default_options = {
        # 'emu/*:cuda'   : False, # Should not be necessary, but it is.
        'emu/*:python' : True, # Should not be necessary, but it is.
    }

    requires = [
        'sardine/1.0.0', # keep the version number in sync with the C++ version.
        'emu/1.0.0',
        # 'pybind11/2.10.4',
    ]

    def layout(self):
        self.folders.generators = "generators"

    generators = 'CMakeDeps'

    def generate(self):
        print(f"BUILDFOLDER: {self.build_folder=}")
        for dep in self.dependencies.values():
            for libdir in dep.cpp_info.libdirs:
                copy(self, "*.so", libdir, self.build_folder)
