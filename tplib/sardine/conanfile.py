from conan import ConanFile
from conan.tools.cmake import CMake, cmake_layout, CMakeToolchain

class sardineConan(ConanFile):
    name = 'sardine'
    version = '1.0.0'
    license = ''

    settings = 'os', 'compiler', 'build_type', 'arch'

    exports_sources = 'CMakeLists.txt', 'include/*', 'src/*', 'test/*'

    implements = ["auto_shared_fpic"]

    options = {
        'cuda': [True, False],
        'milk': [True, False],
        'shared': [True, False],
        'fPIC': [True, False],
    }

    default_options = {
        'cuda': False,
        'milk': False,
        'shared': True,
        'fPIC': True,
        'milk/*:max_semaphore': "1",
    }

    def requirements(self):
        self.requires('emu/1.0.0', transitive_headers=True, options={'cuda' : self.options.cuda})

        if self.options.milk:
            self.requires('milk/20240906.0.0', options={'cuda' : self.options.cuda})

        self.test_requires('gtest/1.13.0')

    def layout(self):
        cmake_layout(self)

    generators = 'CMakeDeps'

    def generate(self):
        print(f"BUILDFOLDER: {self.build_folder=}")

        tc = CMakeToolchain(self)

        tc.variables['sardine_build_cuda'] = self.options.cuda
        tc.variables['sardine_build_milk'] = self.options.milk


        tc.generate()

    def build(self):
        cmake = CMake(self)

        cmake.configure()
        cmake.build()

        cmake.test()

    def package(self):
        cmake = CMake(self)
        cmake.install()

    def package_info(self):
        self.cpp_info.libs = ['sardine']

        if self.options.cuda:
            self.cpp_info.defines = ['SARDINE_CUDA']

            if not self.options.shared:
                # linker by default will not keep sardine_cuda_converter because it is not used explicitly.
                self.cpp_info.exelinkflags = ['-Wl,-u,sardine_cuda_converter']
