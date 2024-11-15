default: install

conan_opt := "--options=boost/*:namespace=me_boost"

configure:
    # curl -sS https://raw.githubusercontent.com/raplonu/cosmic-center-index/master/install.sh | bash -s -- --skip-install
    @conan editable remove tplib/emu
    conan create tplib/emu -b missing {{conan_opt}}
    @conan editable remove tplib/sardine
    conan create tplib/sardine -b missing {{conan_opt}}
    @conan editable remove tplib/sardine/milk
    conan create tplib/sardine/milk -b missing {{conan_opt}}

    pip install tplib/sardine --config-settings=cmake.define.CONAN_ARGS={{conan_opt}}
    pip install tplib/sardine/milk --config-settings=cmake.define.CONAN_ARGS={{conan_opt}}

configure-dev:
    conan build tplib/emu -b missing {{conan_opt}}
    @conan editable add tplib/emu
    conan build tplib/sardine -b missing {{conan_opt}}
    @conan editable add tplib/sardine
    conan build tplib/sardine/milk -b missing {{conan_opt}}
    @conan editable add tplib/sardine/milk

    pip install --no-build-isolation -e tplib/sardine --config-settings=cmake.define.CONAN_ARGS={{conan_opt}}
    pip install --no-build-isolation -e tplib/sardine/milk --config-settings=cmake.define.CONAN_ARGS={{conan_opt}}

# Install rtc in developer mode (editable) in conan and pip
dev *args:
    just cpp-dev {{args}}
    just python-dev

# Install rtc in conan cache and in pip
install *args:
    just cpp-install {{args}}
    just python-install

# Install sardine
cpp-install *args:
    # unregister just in case or else consumer will continue to use the editable version
    just unregister

    conan create . -b missing {{args}}

# Install sardine-python
python-install *args:
    pip install .  {{args}}

cpp-dev *args:
    just register
    conan build . -b missing {{args}}
    @# copy compile_commands.json from either debug of release folder
    -@cp build/*/compile_commands.json build/

python-dev *args:
    pip install --no-build-isolation -ve . {{args}}

build build_type="release":
    just cpp-build {{build_type}}
    just python-build

cpp-build build_type="release":
    cmake --build --preset "conan-{{build_type}}"

python-build:
    pip install --no-build-isolation -ve .

# Run tests
test build_type="release":
    just cpp-test {{build_type}}

cpp-test build_type="release":
    ctest --preset conan-{{build_type}}

# Register sardine as editable in conan
@register:
    conan editable add .

# Unregister sardine as editable in conan
@unregister:
    conan editable remove .

@clean:
    just unregister
    rm -rf					                        \
        build				                        \
        CMakeUserPresets.json                       \
        tplib/emu/build                             \
        tplib/emu/CMakeUserPresets.json             \
        tplib/sardine/build                         \
        tplib/sardine/CMakeUserPresets.json         \
        tplib/sardine/milk/build                    \
        tplib/sardine/milk/CMakeUserPresets.json
