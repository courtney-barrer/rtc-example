default: install

# Install the package
install *args:
    pip install .  {{args}}

# Install the package in editable mode
build *args:
    pip install --no-build-isolation -Ceditable.rebuild=true -ve . {{args}}
