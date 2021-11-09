import logging
import sys
from . import config

logging.basicConfig(stream=sys.stdout,
                    level=config.LOG_LEVEL,
                    format=config.LOG_FORMAT)

from .evaluation import Evaluation

__all__ = ['Evaluation']
