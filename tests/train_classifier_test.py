"""train_classifier_test: Unit tests for training a classifier."""

__author__ = "Harald Wilbertz"
__version__ = "1.0.0"

import os
import sys
import unittest

sys.path.append('../models')
from train_classifier import tokenize


class TrainClassifierTestCase(unittest.TestCase):
    """Tests for methods defined in train classifier.py"""

    def test_tokenize_should_convert_to_lowercase(self):
        """Does the tokenize function convert to lowercase ?"""
        # Arrange
        text = "Please, we need tents and water. We are in Silo, Thank you!"

        # Act
        result = tokenize(text)

        # Assert
        self.assertIsNotNone(result)
        self.assertFalse('Silo' in result)
        self.assertTrue('silo' in result)

    def test_tokenize_should_remove_punctuation(self):
        """Does the tokenize function remove punctuation ?"""
        # Arrange
        text = "Please, we need tents and water. We are in Silo, Thank you! My name is Dr. Who"

        # Act
        result = tokenize(text)

        # Assert
        self.assertIsNotNone(result)
        for token in result:
            self.assertNotIn(".", token)
            self.assertNotIn(",", token)
            self.assertNotIn("!", token)
            self.assertNotIn("?", token)

    def test_tokenize_should_lemmatize_text(self):
        """Does the tokenize function lemmatize words ?"""
        # Arrange
        text = "Please, we need tents and water. We are in Silo, Thank you!"

        # Act
        result = tokenize(text)

        # Assert
        self.assertTrue('tent' in result)

    def test_tokenize_should_remove_stopwords(self):
        """Does the tokenize function remove stop words ?"""
        # Arrange
        text = "Please, we need tents and water. We are in Silo, Thank you!"

        # Act
        result = tokenize(text)

        # Assert
        self.assertFalse('we' in result)
        self.assertFalse('you' in result)
        self.assertFalse('and' in result)
        self.assertFalse('in' in result)


def suite():
    """Define the test suit used by the test runner."""
    return unittest.defaultTestLoader.loadTestsFromTestCase(TrainClassifierTestCase)


if __name__ == '__main__':
    """Run the unit tests."""
    runner = unittest.TextTestRunner()
    runner.run(suite())