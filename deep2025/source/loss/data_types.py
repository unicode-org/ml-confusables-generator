# -----------------------------------------------------------------------------
# Code adapted from TensorFlow Addons under the Apache 2.0 License
# Source: https://github.com/tensorflow/addons
# -----------------------------------------------------------------------------

from typing import Union, List
import tensorflow as tf
from packaging.version import Version
import numpy as np

# Find KerasTensor.
if Version(tf.__version__).release >= Version("2.16").release:
    # Determine if loading keras 2 or 3.
    if (
        hasattr(tf.keras, "version")
        and Version(tf.keras.version()).release >= Version("3.0").release
    ):
        from keras import KerasTensor
    else:
        from tf_keras.src.engine.keras_tensor import KerasTensor
elif Version(tf.__version__).release >= Version("2.13").release:
    from tf.keras.src.engine.keras_tensor import KerasTensor
elif Version(tf.__version__).release >= Version("2.5").release:
    from tf.keras.engine.keras_tensor import KerasTensor
else:
    from tensorflow.python.keras.engine.keras_tensor import KerasTensor
    
Number = Union[
    float,
    int,
    np.float16,
    np.float32,
    np.float64,
    np.int8,
    np.int16,
    np.int32,
    np.int64,
    np.uint8,
    np.uint16,
    np.uint32,
    np.uint64,
]
TensorLike = Union[
    List[Union[Number, list]],
    tuple,
    Number,
    np.ndarray,
    tf.Tensor,
    tf.SparseTensor,
    tf.Variable,
    KerasTensor,
]

FloatTensorLike = Union[tf.Tensor, float, np.float16, np.float32, np.float64]