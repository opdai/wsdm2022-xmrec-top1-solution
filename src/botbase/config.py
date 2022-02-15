import os

BASE_DIR_abs = os.path.dirname(
    os.path.dirname(os.path.dirname(
        os.path.realpath(__file__))))  # same level as lib

BASE_DIR_cwd = os.getcwd()
