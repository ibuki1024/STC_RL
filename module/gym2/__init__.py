import distutils.version
import os
import sys
import warnings

from gym2 import error
from gym2.version import VERSION as __version__

from gym2.core import Env, GoalEnv, Wrapper, ObservationWrapper, ActionWrapper, RewardWrapper
from gym2.spaces import Space
from gym2.envs import make, spec, register
from gym2 import logger
from gym2 import vector

__all__ = ["Env", "Space", "Wrapper", "make", "spec", "register"]
