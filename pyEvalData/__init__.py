import logging
import sys
from . import config

logging.basicConfig(stream=sys.stdout,
                    format=config.LOG_FORMAT)

from .evaluation import Evaluation
from . import io

__all__ = ['Evaluation', 'io']

__version__ = '1.3.7'
