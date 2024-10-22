

FROM ubuntu:20.04

# Set noninteractive mode for apt-get
ENV DEBIAN_FRONTEND=noninteractive

# [Optional] Uncomment this section to install additional packages.
RUN apt-get update \
 && apt-get install -y --no-install-recommends \
    build-essential \
    curl wget git gdb \
    python3-dev python3-pip \
    software-properties-common \
 && add-apt-repository ppa:ubuntu-toolchain-r/test \
 && apt-get update \
 && apt-get install -y --no-install-recommends \
    gcc-11 g++-11 \
 && apt-get clean \
 && rm -rf /var/lib/apt/lists/*

RUN update-alternatives --install /usr/bin/g++ g++ /usr/bin/g++-11 100 \
 && update-alternatives --install /usr/bin/gcc gcc /usr/bin/gcc-11 100


# Download and install Miniconda3
RUN wget https://repo.anaconda.com/miniconda/Miniconda3-py312_24.7.1-0-Linux-x86_64.sh -O /tmp/miniconda.sh \
    && chmod +x /tmp/miniconda.sh \
    && /tmp/miniconda.sh -b -p /opt/miniconda3 \
    && rm /tmp/miniconda.sh

# Add Miniconda to PATH
ENV PATH="/opt/miniconda3/bin:${PATH}"

# Install Rust using rustup
RUN curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y

# Add Rust to the PATH
ENV PATH="/root/.cargo/bin:${PATH}"

RUN cargo install just

# # coreutils build-essential software-properties-common pkg-config

# RUN add-apt-repository ppa:fish-shell/release-3

# RUN apt-get update \
#  && DEBIAN_FRONTEND=noninteractive apt-get -y install --no-install-recommends \
#     fish \
#  && rm -rf /var/lib/apt/lists/*

# # RUN curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y
# # ENV PATH="/root/.cargo/bin:${PATH}"

# # RUN curl -sS https://starship.rs/install.sh | sh -s -- --yes
# # RUN mkdir -p /root/.config/fish
# # RUN echo "starship init fish | source" >> /root/.config/fish/config.fish

# # RUN cargo install just

RUN python3 -m pip install -U --disable-pip-version-check --no-cache-dir cmake ninja conan ipython scikit_build_core numpy

RUN curl -sS https://raw.githubusercontent.com/raplonu/cosmic-center-index/master/install.sh | bash -s -- --skip-install \
 && conan install -b missing --requires boost/1.84.0 --requires fmt/11.0.0 --requires ms-gsl/4.0.0 --requires pybind11/2.13.1
