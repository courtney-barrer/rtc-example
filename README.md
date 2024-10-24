# RTC example

This project is an example of how to use a RTC written in C++ from python.

## Requirements

- gcc 11 or later
- python 3.10 or later
- [just](https://just.systems/) (optional), utility to run the commands in the `justfile`

## Configure

```bash
    # install conan and configure it
    curl -sS https://raw.githubusercontent.com/raplonu/cosmic-center-index/master/install.sh | bash -s
```

## Build

```bash
# Build the C++ library and the cli application
conan build .
# Register the library in conan
conan editable .
# Compile the python module (it will use the library register with conan)
pip install .
```

## Test

Simple test:

```bash
ipython -i script_test.py
```

And in another terminal:

```bash
./build/Release/baldr_main --config camera_config.json
```
