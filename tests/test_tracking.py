#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Tests for the Tracker class.

Created on Mon Apr 24 10:57:01 2017

@author: yosef
"""

import unittest
from math import isclose

from openptv_python.parameters import (
    TrackPar,
)
from openptv_python.track import pos3d_in_bounds, predict, search_volume_center_moving


class TestPredict(unittest.TestCase):
    """Test the predict function."""

    def test_predict(self):
        """Test the predict function."""
        prev_pos = [1.1, 0.6]
        curr_pos = [2.0, -0.8]
        result = [2.9, -2.2]
        EPS = 1e-9  # You can adjust this epsilon value as needed

        c = [0.0, 0.0]
        predict(prev_pos, curr_pos, c)

        self.assertTrue(
            isclose(c[0], result[0], rel_tol=EPS), f"Expected 2.9 but found {c[0]}"
        )
        self.assertTrue(
            isclose(c[1], result[1], rel_tol=EPS), f"Expected -2.2 but found {c[1]}"
        )


class TestSearchVolumeCenterMoving(unittest.TestCase):
    """Test the search_volume_center_moving function."""

    def test_search_volume_center_moving(self):
        """Test the search_volume_center_moving function."""
        prev_pos = [1.1, 0.6, 0.1]
        curr_pos = [2.0, -0.8, 0.2]
        result = [2.9, -2.2, 0.3]
        EPS = 1e-9  # You can adjust this epsilon value as needed

        c = [0.0, 0.0, 0.0]
        search_volume_center_moving(prev_pos, curr_pos, c)

        self.assertTrue(
            isclose(c[0], result[0], rel_tol=EPS), f"Expected 2.9 but found {c[0]}"
        )
        self.assertTrue(
            isclose(c[1], result[1], rel_tol=EPS), f"Expected -2.2 but found {c[1]}"
        )
        self.assertTrue(
            isclose(c[2], result[2], rel_tol=EPS), f"Expected 0.3 but found {c[2]}"
        )


class TestPos3dInBounds(unittest.TestCase):
    """Test the pos3d_in_bounds function."""

    def test_pos3d_in_bounds(self):
        """Test the pos3d_in_bounds function."""
        inside = [1.0, -1.0, 0.0]
        outside = [2.0, -0.8, 2.1]

        bounds = TrackPar(
            0.4, 120, 2.0, -2.0, 2.0, -2.0, 2.0, -2.0, 0.0, 0.0, 0.0, 0.0, 1
        )

        result = pos3d_in_bounds(inside, bounds)

        self.assertEqual(result, 1, "Expected True but found %s" % result)

        result = pos3d_in_bounds(outside, bounds)

        self.assertEqual(result, 0, "Expected False but found %s" % result)


if __name__ == "__main__":
    unittest.main()
