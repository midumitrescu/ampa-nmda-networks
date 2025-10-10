import unittest

from Setup import verify_setup


class MyTestCase(unittest.TestCase):
    def test_setup(self):
        self.assertEqual("works", verify_setup())


if __name__ == '__main__':
    unittest.main()
