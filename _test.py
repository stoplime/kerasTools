import unittest

import os
import json
import datatool
import model
import numpy as np

class TestStringMethods(unittest.TestCase):

    def test_upper(self):
        self.assertEqual('foo'.upper(), 'FOO')

    def test_isupper(self):
        self.assertTrue('FOO'.isupper())
        self.assertFalse('Foo'.isupper())

    def test_split(self):
        s = 'hello world'
        self.assertEqual(s.split(), ['hello', 'world'])
        # check that s.split fails when the separator is not a string
        with self.assertRaises(TypeError):
            s.split(2)

    def test_datatool(self):
        d = datatool.dataTool(os.path.join(os.getcwd(), "test_images"), os.path.join(os.getcwd(), "test_labels.json"))
        self.assertTrue(all(d.get_label_per_batch(0) == np.array([123, 234, 345])))

if __name__ == '__main__':
    unittest.main()
