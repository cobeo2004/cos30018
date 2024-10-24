from .FredHelper import FredHelper
from .env import Env, env
from .IndexHelper import IndexHelper

FredInstance = FredHelper()
IndexInstance = IndexHelper()
__all__ = ["FredHelper", "env", "IndexHelper", "FredInstance", "IndexInstance", "Env"]
