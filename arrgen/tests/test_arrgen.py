import numpy as np
from arrgen import uniform_array, normal_array, constant_array

SEED = 42
SHAPE = [2, 2]


def test_uniform_array():
    min_val = 0.0
    max_val = 1.0
    expected = np.array([[0.52655741, 0.54272521], [0.6364651, 0.40590176]])
    x = uniform_array(SEED, SHAPE, min_val, max_val)
    np.testing.assert_array_almost_equal(
        x,
        expected,
        decimal=8,
        err_msg="uniform_array output does not match expected values",
    )


def test_normal_array():
    mean = 0.0
    std_dev = 1.0
    expected = np.array([[0.06942792, 0.13293812], [0.26257636, -0.22530088]])
    x = normal_array(SEED, SHAPE, mean, std_dev)
    np.testing.assert_array_almost_equal(
        x,
        expected,
        decimal=8,
        err_msg="normal_array output does not match expected values",
    )


def test_constant_array():
    value = 0.0
    expected = np.array([[0.0, 0.0], [0.0, 0.0]])
    x = constant_array(SHAPE, value)
    np.testing.assert_array_almost_equal(
        x,
        expected,
        decimal=8,
        err_msg="constant_array output does not match expected values",
    )
