"""process_data: Python methods for loading, cleaning and saving disaster response data."""

__author__ = "Harald Wilbertz"
__version__ = "1.0.0"

import sys
import logging
import pandas as pd
from sqlalchemy import create_engine


def load_data(messages_filepath: str, categories_filepath: str) -> pd.DataFrame:
    """
       Load the message and categories data from 2 csv file and merge them into
       an pandas data frame. The id column is used for joining the data.

       arguments:
           message_filepath (string): The full path to the messages csv file.
           categories_filepath (string): The full path to the categories csv file.
       returns:
           data (pandas data frame) : The message data combined with the corresponding categories.
    """
    logging.info(
        'load_data started, messages_filepath {}, categories_filepath{}'.format(messages_filepath, categories_filepath))
    messages = pd.read_csv(messages_filepath)
    categories = pd.read_csv(categories_filepath)
    df = pd.merge(messages, categories, on='id')

    return df


def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """
       Cleans disaster response data. This includes the following steps:

       - Expand the categories data into several columns.
            - Use the all but the last 2 characters as column names
            - Transform the last character into a numeric values
              and use this number as the new value.
            - Change all value of 2 to 1
        - Remove duplicates

       arguments:
           df (pandas data frame): The data frame to be cleaned.
       returns:
           df (pandas data frame) : The cleaned data frame.
    """
    logging.info('clean_data started.')

    categories = df['categories'].str.split(pat=';', expand=True)
    row = categories.iloc[0]
    category_colnames = row.apply(lambda x: x[:-2])
    categories.columns = category_colnames

    for column in categories:
        # Use the last character of the string as the new column value
        categories[column] = categories[column].apply(lambda x: x[:][-1])

        # change all column values to numeric
        categories[column] = pd.to_numeric(categories[column])

    # Change category 2 values to 1
    categories = categories.replace(2, 1)

    df.drop('categories', axis=1, inplace=True)
    df = pd.concat([df, categories], axis=1)

    df.drop_duplicates(inplace=True)

    return df


def save_data(df: pd.DataFrame, database_filename: str) -> None:
    """
       Save a pandas data frame into a SQLite database. Any existing
       database file will be overwritten.

       arguments:
           df (pandas data frame): The data frame to be persisted into the database.
           database_filename (string): The full path to SQLite database.
       returns:
           None
    """
    logging.info('save_data started.')
    engine = create_engine('sqlite:///' + database_filename)
    df.to_sql('Messages', engine, index=False, if_exists='replace', chunksize=500)


def log_and_print(log_message: str) -> None:
    """
        This method outputs its given string on the console and
        as a log message with info log level.

        arguments:
           log_message (string): The string to be logged.
       returns:
           None
    """
    print(log_message)
    logging.info(log_message)


def main() -> None:
    """
        This is the main entry point into the process data program. Running
        the program loads data from the messages and categories csv file and
        persists this data into a SQLite database. Any existing database file
        will be overwritten. Log messages are written both to the console and
        a dedicated log file named process_data.log.

        The following sequence of actions is executed:
        - Load data csv files.
        - Clean the data.
        - Save the data into a SQLite database.

        command line arguments:
            messages_filepath (string): The full path to the messages csv file.
            categories_filepath (string): The full path to the categories csv file.
            database_filepath (string) : The full path to the SQLite database.
        returns:
           None
    """
    logging.basicConfig(
        filename='process_data.log',
        format='%(asctime)s : %(levelname)s : %(message)s',
        level=logging.INFO)
    logging.info('Started with arguments: {}'.format(sys.argv))

    if len(sys.argv) == 4:
        try:
            messages_filepath, categories_filepath, database_filepath = sys.argv[1:]

            log_and_print('Loading data...\n    MESSAGES: {}\n    CATEGORIES: {}'
                          .format(messages_filepath, categories_filepath))
            df: pd.DataFrame = load_data(messages_filepath, categories_filepath)

            log_and_print('Cleaning data...')
            df = clean_data(df)

            log_and_print('Saving data...\n    DATABASE: {}'.format(database_filepath))
            save_data(df, database_filepath)

            log_and_print('Cleaned data saved to database!')

        except FileNotFoundError as error:
            logging.error(error)
            print(error)

    else:
        log_and_print('Please provide the filepaths of the messages and categories '
                      'datasets as the first and second argument respectively, as '
                      'well as the filepath of the database to save the cleaned data '
                      'to as the third argument. \n\nExample: python process_data.py '
                      'disaster_messages.csv disaster_categories.csv '
                      'DisasterResponse.db')

    logging.info('Finished\n')


if __name__ == '__main__':
    main()
