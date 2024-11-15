from conan import ConanFile
from conan.tools.cmake import CMake, cmake_layout, CMakeToolchain
from conan.tools.files import copy

class sardineConan(ConanFile):
    name = 'sardine-milk'
    version = '1.0.0'
    license = ''

    settings = 'os', 'compiler', 'build_type', 'arch'

    exports_sources = 'CMakeLists.txt', 'include/*', 'src/*', 'test/*'

    options = {
        'cuda': [True, False],
        'python_module': [True, False],
    }

    default_options = {
        'cuda': False,
        'python_module': False,
        'milk/*:max_semaphore': "1",
    }

    def requirements(self):
        # Order matters here. We want to enforce the cuda option for emu
        self.requires('emu/1.0.0', transitive_headers=True, options={'cuda' : self.options.cuda})
        self.requires('milk/20240906.0.0', options={'cuda' : self.options.cuda})
        self.requires('sardine/1.0.0', transitive_headers=True, transitive_libs=True)

        self.test_requires('gtest/1.13.0')

    def layout(self):
        if self.options.python_module:
            # Using conan as CMAKE_PROJECT_TOP_LEVEL_INCLUDES cmake_layout does not work
            # We don't want to pollute the build folder with conan. We put everything in "generators"
            self.folders.generators = "generators"
        else:
            # Otherwise, we use the default cmake layout
            cmake_layout(self)

    generators = 'CMakeDeps'

    def generate(self):
        tc = CMakeToolchain(self)

        tc.variables['sardine_build_cuda'] = self.options.cuda

        tc.generate()

        if self.options.python_module:
            for libdir in self.dependencies['milk'].cpp_info.libdirs:
                copy(self, "*.so*", libdir, self.build_folder)


    def build(self):
        cmake = CMake(self)

        cmake.configure()
        cmake.build()

        cmake.test()

    def package(self):
        cmake = CMake(self)
        cmake.install()

    def package_info(self):
        self.cpp_info.libs = ['sardinemilk']
