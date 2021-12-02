import logging
import sys
from . import config

logging.basicConfig(stream=sys.stdout,
                    format=config.LOG_FORMAT)

from .evaluation import Evaluation
from . import io
from . import helpers

__all__ = ['Evaluation', 'io', 'helpers']

__version__ = '1.5.0'
