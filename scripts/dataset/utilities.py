""" Module that contains all constants necessary to run the scripts.

Constants are separated into classes in order to make the module more readable and maintainable.
"""
import os
import sys
from pathlib import Path


class System:
    """ Constants related to the running system. """
    PY_INTERPRETER = sys.executable


class Paths:
    """ Constants that represent useful file system paths.

     This class use pathlib library to declare the paths.
     """

    # ------------ COMMON ------------
    ROOT_DIR = Path(os.path.dirname(os.path.abspath(__file__))).parent.parent
    SCRIPTS_DIR = ROOT_DIR / 'scripts'

    # ----------- RESOURCES -----------
    OUTPUT_DIR = ROOT_DIR / 'output'
    DATASETS_DIR = OUTPUT_DIR / 'datasets'
    RAW_DATASETS_DIR = DATASETS_DIR / 'raw'
    CANONICAL_DATASETS_DIR = DATASETS_DIR / 'canonical'
    GENERATED_DATASETS_DIR = DATASETS_DIR / 'generated'
    REPRESENTATION_DATASETS_DIR = DATASETS_DIR / 'representation'

    # ------------ DATASET -----------
    DATASET_SCRIPTS_DIR = SCRIPTS_DIR / 'dataset'
    DATASET_CONFIG_DIR = DATASET_SCRIPTS_DIR / 'config'
    DATASET_IMPORTER_CONFIG_DIR = DATASET_CONFIG_DIR / 'importer'
    DATASET_CANONICALIZER_CONFIG_DIR = DATASET_CONFIG_DIR / 'canonicalizer'
    DATASET_GENERATOR_CONFIG_DIR = DATASET_CONFIG_DIR / 'generator'
    DATASET_METRICS_CONFIG_DIR = DATASET_CONFIG_DIR / 'metrics'
    DATASET_REPRESENTATION_CONFIG_DIR = DATASET_CONFIG_DIR / 'representation'
