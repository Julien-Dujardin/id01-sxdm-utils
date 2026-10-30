"""Color normalization regressions."""

import unittest
import numpy as np
from sxdm.plot.utils import make_hsv


class TestHSV(unittest.TestCase):
    def test_integer_magnitude(self):
        image = make_hsv(np.array([[0, 2]]), np.zeros((1, 2)))
        np.testing.assert_allclose(image, [[[1, 1, 1], [1, 0, 0]]])

    def test_zero_magnitude(self):
        with np.errstate(divide="raise", invalid="raise"):
            image = make_hsv(np.zeros((2, 2)), np.zeros((2, 2)))
        np.testing.assert_allclose(image, np.ones((2, 2, 3)))

    def test_constant_stretched_magnitude(self):
        with np.errstate(divide="raise", invalid="raise"):
            image = make_hsv(np.full((2, 2), 3.0), np.zeros((2, 2)), stretch=True)
        np.testing.assert_allclose(image, np.ones((2, 2, 3)))
