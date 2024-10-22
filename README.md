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
    # install the two embeded dependencies: emu and sardine
    conan create tplib/emu -b missing
    conan create tplib/sardine -b missing
    pip install tplib/sardine
```

## Build

```bash
conan build .
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
