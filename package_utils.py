# 3rd party dependencies
import tensorflow as tf


def validate_for_keras3():
    tf_major = int(tf.__version__.split(".", maxsplit=1)[0])
    tf_minor = int(tf.__version__.split(".", maxsplit=-1)[1])

    # tf_keras is a must dependency after tf 2.16
    if tf_major == 1 or (tf_major == 2 and tf_minor < 16):
        return

    try:
        import tf_keras

    except ImportError as err:
        # you may consider to install that package here
        raise ValueError(
            f"You have tensorflow {tf.__version__} and this requires "
            "tf-keras package. Please run `pip install tf-keras` "
            "or downgrade your tensorflow."
        ) from err
