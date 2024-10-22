default: install

configure:
    # Configure conan and adds additional deps
    curl -sS https://raw.githubusercontent.com/raplonu/cosmic-center-index/master/install.sh | bash -s -- --skip-install
    just tplib/emu/dev -o python=True
    just tplib/sardine/dev

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

# Install rtc in developer mode (editable) in conan and pip
dev *args:
    just cpp-dev {{args}}
    just python-dev

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
    pip install --no-build-isolation -Ceditable.rebuild=true -ve .

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
    rm -rf						        \
        build					        \
        CMakeUserPresets.json
