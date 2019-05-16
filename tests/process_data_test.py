import unittest
import pandas as pd

import sys
sys.path.append('../data')

from process_data import load_data


class ProcessDataTestCase(unittest.TestCase):
    """Tests for methods defined in process_data.py"""

    def test_load_data(self):
        """Does the load_data function return a pandas data frame ?"""
        df: pd.DataFrame = load_data("../data/disaster_messages.csv", "../data/disaster_categories.csv")
        self.assertIsNotNone(df)


def suite():
    suite = unittest.TestSuite()
    suite = unittest.defaultTestLoader.loadTestsFromTestCase(ProcessDataTestCase)
    return suite


if __name__ == '__main__':
    runner = unittest.TextTestRunner()
    runner.run(suite())

