import os


class Config:
    # To run on dev environment without Intel Graphics
    IGNORE_CUDA = bool(os.environ.get('IGNORE_CUDA'))
