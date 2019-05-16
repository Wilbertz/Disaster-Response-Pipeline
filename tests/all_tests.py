import unittest

from process_data_test import ProcessDataTestCase
from train_classifier_test import TrainClassifierTestCase


def suite():
    """Create a test suite with all unit tests in solution"""
    test_suite = unittest.TestSuite()
    test_suite.addTest(unittest.defaultTestLoader.loadTestsFromTestCase(ProcessDataTestCase))
    test_suite.addTest(unittest.defaultTestLoader.loadTestsFromTestCase(TrainClassifierTestCase))

    return test_suite


if __name__ == '__main__':
    """Run all unit tests in solution"""
    runner = unittest.TextTestRunner()
    runner.run(suite())
