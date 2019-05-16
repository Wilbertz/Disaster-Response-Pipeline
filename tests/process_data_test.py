"""process_data_test: Unit tests for process data."""

__author__ = "Harald Wilbertz"
__version__ = "1.0.0"

import os
import sys
import unittest
import pandas as pd
from sqlalchemy import create_engine


sys.path.append('../data')
from process_data import load_data, clean_data, save_data


class ProcessDataTestCase(unittest.TestCase):
    """Tests for methods defined in process_data.py"""

    UNIT_TEST_DB = "UnitTest_DisasterResponse.sqlite"

    @classmethod
    def setUpClass(cls):
        """
        Delete an already existing database file. The database file is not deleted
        after running all tests in order the enable further analysis.
        """
        if os.path.isfile(cls.UNIT_TEST_DB):
            os.remove(cls.UNIT_TEST_DB)

    def test_load_data(self):
        """Does the load_data function return an valid pandas data frame ?"""
        # Arrange

        # Act
        df = load_data("unittest_disaster_messages.csv", "unittest_disaster_categories.csv")

        # Assert
        self.assertIsNotNone(df)
        # An inner join should be done, ignoring the second row in disaster messages.
        self.assertEqual(df.shape, (1, 5))

    def test_clean_data(self):
        """Does the clean data function produce the desired result ?"""
        # Arrange
        df_input = load_data("unittest_disaster_messages.csv", "unittest_disaster_categories.csv")

        # Act
        df_result = clean_data(df_input)

        # Assert
        self.assertIsNotNone(df_result)
        self.assertEqual(df_result.shape, (1, 40))
        self.assertEqual((df_result['id'][0]), 2)
        self.assertEqual((df_result['message'][0]), 'Weather update - a cold front from Cuba that could pass over Haiti')
        self.assertEqual((df_result['original'][0]), 'Un front froid se retrouve sur Cuba ce matin. Il pourrait traverser Haiti demain. Des averses de pluie isolee sont encore prevues sur notre region ce soi')
        self.assertEqual((df_result['genre'][0]), 'direct')
        self.assertEqual((df_result['related'][0]), 1)
        self.assertEqual((df_result['request'][0]), 0)
        self.assertEqual((df_result['offer'][0]), 0)
        self.assertEqual((df_result['aid_related'][0]), 0)
        self.assertEqual((df_result['medical_help'][0]), 0)
        self.assertEqual((df_result['medical_products'][0]), 0)
        self.assertEqual((df_result['search_and_rescue'][0]), 0)
        self.assertEqual((df_result['security'][0]), 0)
        self.assertEqual((df_result['military'][0]), 0)
        self.assertEqual((df_result['child_alone'][0]), 0)
        self.assertEqual((df_result['water'][0]), 0)
        self.assertEqual((df_result['food'][0]), 0)
        self.assertEqual((df_result['shelter'][0]), 0)
        self.assertEqual((df_result['clothing'][0]), 0)
        self.assertEqual((df_result['money'][0]), 0)
        self.assertEqual((df_result['missing_people'][0]), 0)
        self.assertEqual((df_result['refugees'][0]), 0)
        self.assertEqual((df_result['death'][0]), 0)
        self.assertEqual((df_result['other_aid'][0]), 0)
        self.assertEqual((df_result['infrastructure_related'][0]), 0)
        self.assertEqual((df_result['transport'][0]), 0)
        self.assertEqual((df_result['buildings'][0]), 0)
        self.assertEqual((df_result['electricity'][0]), 0)
        self.assertEqual((df_result['tools'][0]), 0)
        self.assertEqual((df_result['hospitals'][0]), 0)
        self.assertEqual((df_result['shops'][0]), 0)
        self.assertEqual((df_result['aid_centers'][0]), 0)
        self.assertEqual((df_result['other_infrastructure'][0]), 0)
        self.assertEqual((df_result['weather_related'][0]), 0)
        self.assertEqual((df_result['floods'][0]), 0)
        self.assertEqual((df_result['storm'][0]), 0)
        self.assertEqual((df_result['fire'][0]), 0)
        self.assertEqual((df_result['earthquake'][0]), 0)
        self.assertEqual((df_result['cold'][0]), 0)
        self.assertEqual((df_result['other_weather'][0]), 0)
        self.assertEqual((df_result['direct_report'][0]), 0)

    def test_save_data(self):
        """Does save_data persist the data in the database ?"""
        # Arrange
        df_input = load_data("unittest_disaster_messages.csv", "unittest_disaster_categories.csv")
        df_result = clean_data(df_input)

        # Act
        save_data(df_result, self.UNIT_TEST_DB)
        # Assert
        engine = create_engine('sqlite:///' + self.UNIT_TEST_DB)
        df = pd.read_sql_table('Messages', engine)

        self.assertIsNotNone(df)
        self.assertEqual(df.shape, (1, 40))


def suite():
    """Define the test suit used by the test runner."""
    return unittest.defaultTestLoader.loadTestsFromTestCase(ProcessDataTestCase)


if __name__ == '__main__':
    """Run the unit tests."""
    runner = unittest.TextTestRunner()
    runner.run(suite())
