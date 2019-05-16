import unittest

from process_data_test import ProcessDataTestCase


def suite():
    return unittest.defaultTestLoader.loadTestsFromTestCase(ProcessDataTestCase)


if __name__ == '__main__':
    runner = unittest.TextTestRunner()
    runner.run(suite())
