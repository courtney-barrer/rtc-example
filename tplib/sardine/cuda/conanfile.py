from conan import ConanFile
from conan.tools.cmake import CMake, cmake_layout, CMakeToolchain
from conan.tools.files import copy

class sardineConan(ConanFile):
    name = 'sardine-cuda'
    version = '1.0.0'
    license = ''

    settings = 'os', 'compiler', 'build_type', 'arch'

    exports_sources = 'CMakeLists.txt', 'include/*', 'src/*', 'test/*'

    options = {
        'python_module': [True, False],
    }

    default_options = {
        'python_module': False,
        'emu/*:cuda': True,
    }

    def requirements(self):
        self.requires('sardine/1.0.0', transitive_headers=True)
        self.requires('emu/1.0.0', transitive_headers=True)
        # self.requires('boost/1.84.0', transitive_headers=True)
        # self.requires('fmt/11.0.0', transitive_headers=True)

        self.test_requires('gtest/1.13.0')

    def layout(self):
        if self.options.python_module:
            # Using conan as CMAKE_PROJECT_TOP_LEVEL_INCLUDES cmake_layout does not work
            # We don't want to pollute the build folder with conan. We put everything in "generators"
            self.folders.generators = "generators"
        else:
            # Otherwise, we use the default cmake layout
            cmake_layout(self)

    generators = 'CMakeDeps', 'CMakeToolchain'

    # def generate(self):
    #     if self.options.python_module:
    #         for dep in self.dependencies.values():
    #             for libdir in dep.cpp_info.libdirs:
    #                 copy(self, "*.so*", libdir, self.build_folder)

    def build(self):
        cmake = CMake(self)

        cmake.configure()
        cmake.build()

        cmake.test()

    def package(self):
        cmake = CMake(self)
        cmake.install()

    def package_info(self):
        self.cpp_info.libs = ['sardinecuda']
