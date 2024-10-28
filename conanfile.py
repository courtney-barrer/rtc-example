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
        'python': [True, False],
    }

    exports_sources = 'CMakeLists.txt', 'include/*', 'src/*', 'test/*', 'tplib/*'

    settings = 'os', 'compiler', 'build_type', 'arch'

    default_options = {
        'shared': True,
        'fPIC': True,
        'boost/*:namespace' : 'myboost',
        # 'boost/*:shared' : True,
        'python'     : True,
    }

    def requirements(self):
        self.requires('fmt/11.0.0', transitive_headers=True)
        self.requires('boost/1.84.0', transitive_headers=True)
        self.requires('ms-gsl/4.0.0', transitive_headers=True)
        self.requires('mdspan/0.6.0', transitive_headers=True)
        self.requires('tl-expected/1.1.0', transitive_headers=True)
        self.requires('tl-optional/1.1.0', transitive_headers=True)
        self.requires('half/2.2.0', transitive_headers=True)
        self.requires('dlpack/1.0', transitive_headers=True)
        self.requires('range-v3/0.12.0', transitive_headers=True)

        # We keep the python package as a set of header. No deps.
        if self.options.python:
            self.requires('pybind11/2.13.1', transitive_headers=True)\

        self.test_requires('gtest/1.13.0')


    def layout(self):
        cmake_layout(self)

        self.cpp.source.components['emu'].includedirs = ['tplib/emu/include/core']
        self.cpp.build.components['emu'].libdirs = self.cpp.build.libdirs

        self.cpp.source.components['sardine'].includedirs = ['tplib/sardine/include']
        self.cpp.build.components['sardine'].libdirs = self.cpp.build.libdirs

        self.cpp.source.components['baldr'].includedirs = ['include']
        self.cpp.build.components['baldr'].libdirs = self.cpp.build.libdirs

        if self.options.python:
            self.cpp.source.components['emu-python'].includedirs = ['tplib/emu/include/python']


    generators = 'CMakeDeps'

    def generate(self):
        tc = CMakeToolchain(self)

        tc.variables['emu_build_python'] = self.options.python
        tc.variables['emu_boost_namespace'] = self.dependencies['boost'].options.namespace

        tc.generate()

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
        self.cpp_info.components['emu'].libs = ['emucore']
        self.cpp_info.components['emu'].requires = [
            'fmt::fmt',
            'boost::boost',
            'ms-gsl::_ms-gsl',
            'tl-expected::expected',
            'tl-optional::optional',
            'mdspan::mdspan',
            'half::half',
            'dlpack::dlpack',
            'range-v3::range-v3'

        ]

        self.cpp_info.components['emu'].defines = ['EMU_BOOST_NAMESPACE={}'.format(self.dependencies['boost'].options.namespace)]

        self.cpp_info.components['sardine'].libs = ['sardine']
        self.cpp_info.components['sardine'].requires = [
            'emu',
            'boost::url',
            'boost::json',
        ]

        self.cpp_info.components['baldr'].libs = ['baldr']
        self.cpp_info.components['baldr'].requires = [
            'sardine'
        ]

        if self.options.python:
            self.cpp_info.components['emu-python'].bindirs = []
            self.cpp_info.components['emu-python'].libdirs = []
            self.cpp_info.components['emu-python'].requires = ['emu', 'pybind11::pybind11']
