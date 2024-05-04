#!/bin/bash

CONDA_PREFIX=./venv

# Deactivate and remove existing environment
eval "$(conda shell.bash deactivate)"
conda env remove --prefix $CONDA_PREFIX

# Flags to C++ compiler (needed to install python-snappy package)
export CPPFLAGS="-I/opt/homebrew/include -L/opt/homebrew/lib"

# Create new environment
conda env create --prefix $CONDA_PREFIX --file environment.yml
conda env update --prefix $CONDA_PREFIX --file environment.local.yml
