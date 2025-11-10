__version__ = '0.1'

# TensorFlow 2.x / Keras 3 compatibility fixes
import numpy as np
import tensorflow as tf

# NumPy compatibility fixes
if not hasattr(np, "int"):
    np.int = np.int64
if not hasattr(np, "float"):
    np.float = np.float64

# TensorFlow compatibility fixes
tf.compat.v1.disable_v2_behavior()
if not hasattr(tf, "GraphKeys"):
    tf.GraphKeys = tf.compat.v1.GraphKeys
if not hasattr(tf, 'layers'):
    tf.layers = tf.compat.v1.layers
if not hasattr(tf, "log"):
    tf.log = tf.math.log
if not hasattr(tf, "log1p"):
    tf.log1p = tf.math.log1p
if not hasattr(tf.train, "AdamOptimizer"):
    tf.train.AdamOptimizer = tf.compat.v1.train.AdamOptimizer
if not hasattr(tf, "reset_default_graph"):
    tf.reset_default_graph = tf.compat.v1.reset_default_graph
if not hasattr(tf, "variable_scope"):
    tf.variable_scope = tf.compat.v1.variable_scope
if not hasattr(tf, "VariableScope"):
    tf.VariableScope = tf.compat.v1.VariableScope
if not hasattr(tf, "get_variable"):
    tf.get_variable = tf.compat.v1.get_variable
if not hasattr(tf, "placeholder"):
    tf.placeholder = tf.compat.v1.placeholder
if not hasattr(tf, "get_default_graph"):
    tf.get_default_graph = tf.compat.v1.get_default_graph
if not hasattr(tf, "random_normal"):
    tf.random_normal = tf.compat.v1.random_normal
if not hasattr(tf, "get_collection"):
    tf.get_collection = tf.compat.v1.get_collection
if not hasattr(tf, "check_numerics"):
    tf.check_numerics = tf.debugging.check_numerics
if not hasattr(tf.summary, "merge"):
    tf.summary.merge = tf.compat.v1.summary.merge
if not hasattr(tf.summary, "histogram"):
    tf.summary.histogram = tf.compat.v1.summary.histogram
if not hasattr(tf.summary, "scalar"):
    tf.summary.scalar = tf.compat.v1.summary.scalar
if not hasattr(tf, "no_op"):
    tf.no_op = tf.compat.v1.no_op
if not hasattr(tf, "variables_initializer"):
    tf.variables_initializer = tf.compat.v1.variables_initializer
if not hasattr(tf, "Session"):
    tf.Session = tf.compat.v1.Session
if not hasattr(tf, "get_default_session"):
    tf.get_default_session = tf.compat.v1.get_default_session
if not hasattr(tf, "global_variables"):
    tf.global_variables = tf.compat.v1.global_variables
if not hasattr(tf, "is_variable_initialized"):
    tf.is_variable_initialized = tf.compat.v1.is_variable_initialized
if not hasattr(tf.train, "Saver"):
    tf.train.Saver = tf.compat.v1.train.Saver

from .augmentation import *
from .model import *
from .prediction import *
from .preprocessing import *
from .reconstruction import *
from .training import *
from .utils import *

__all__ = ['Donut', 'DonutPredictor', 'DonutTrainer']
