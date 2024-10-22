from conan import ConanFile
from conan.tools.cmake import CMake, cmake_layout

class BaldrConan(ConanFile):
    name = 'baldr'
    version = '1.0.0'
    license = ''

    options = {
        'shared': [True, False],
        'fPIC': [True, False],
    }

    exports_sources = 'CMakeLists.txt', 'include/*', 'src/*', 'test/*'

    settings = 'os', 'compiler', 'build_type', 'arch'

    default_options = {
        'shared': False,
        'fPIC': True,
    }

    def requirements(self):
        self.requires('sardine/1.0.0', transitive_headers=True)
        self.requires('boost/1.84.0', transitive_headers=True)

        self.test_requires('gtest/1.13.0')

    def layout(self):
        cmake_layout(self)

    generators = 'CMakeDeps', 'CMakeToolchain'

    def build(self):
        cmake = CMake(self)

        cmake.configure()
        cmake.build()

    def package(self):
        cmake = CMake(self)
        cmake.install()

    def package_info(self):
        self.cpp_info.libs = ['baldr']
