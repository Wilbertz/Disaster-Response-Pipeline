import unittest

from process_data_test import ProcessDataTestCase

def suite():
    suite = unittest.TestSuite()
    #suite.addTest(ProcessDataTestCase())
    suite = unittest.defaultTestLoader.loadTestsFromTestCase(ProcessDataTestCase)
    return suite

if __name__ == '__main__':
    runner = unittest.TextTestRunner()
    runner.run(suite())
