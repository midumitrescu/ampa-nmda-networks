"""Tests for ExtendedDict: attribute-style access and immutability."""
import unittest

try:
    from utils import ExtendedDict
except ImportError:
    from src.utils import ExtendedDict


class TestExtendedDict(unittest.TestCase):
    def test_attribute_access(self):
        d = ExtendedDict({"a": 1, "b": 2})
        self.assertEqual(d.a, 1)
        self.assertEqual(d.b, 2)
        self.assertEqual(d["a"], 1)
        self.assertEqual(d.get("c", 99), 99)

    def test_immutable_item_assignment_raises(self):
        d = ExtendedDict({"a": 1, "b": 2})
        with self.assertRaises(TypeError) as ctx:
            d["a"] = 3
        self.assertIn("immutable", str(ctx.exception))

    def test_immutable_attribute_assignment_raises(self):
        d = ExtendedDict({"a": 1, "b": 2})
        with self.assertRaises(TypeError) as ctx:
            d.a = 3
        self.assertIn("immutable", str(ctx.exception))

    def test_immutable_new_key_raises(self):
        d = ExtendedDict({"a": 1})
        with self.assertRaises(TypeError):
            d["c"] = 3

    def test_read_after_construction(self):
        d = ExtendedDict({"x": 10})
        self.assertEqual(d.x, 10)
        self.assertEqual(d["x"], 10)
