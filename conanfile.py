from conan import ConanFile
from conan.tools.cmake import CMakeToolchain, CMake, cmake_layout
from conan.tools.files import copy

class BaldrConan(ConanFile):
    name = 'baldr'
    version = '1.0.0'
    license = ''

    implements = ["auto_shared_fpic"]

    options = {
        'shared': [True, False],
        'fPIC': [True, False],
        'python_module': [True, False],
    }

    exports_sources = 'CMakeLists.txt', 'include/*', 'src/*', 'test/*', 'tplib/*'

    settings = 'os', 'compiler', 'build_type', 'arch'

    default_options = {
        'shared': True,
        'fPIC':   True,
        'python_module': False,
        'boost/*:namespace': 'me_boost'
    }

    def requirements(self):
        self.requires('sardine/1.0.0', transitive_headers=True, transitive_libs=True)
        self.requires('sardine-milk/1.0.0')

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

        tc.generate()

        if self.options.python_module:
            for dep in self.dependencies.values():
                for libdir in dep.cpp_info.libdirs:
                    copy(self, "*.so*", libdir, self.build_folder)



    def build(self):
        cmake = CMake(self)

        cmake.configure()
        cmake.build()

    def package(self):
        cmake = CMake(self)
        cmake.install()

    def package_info(self):
        self.cpp_info.libs = ['baldr']
