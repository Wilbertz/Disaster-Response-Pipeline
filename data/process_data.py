import sys
import os
import logging
import pandas as pd


def load_data(messages_filepath: str, categories_filepath: str) -> pd.DataFrame:
    logging.info('load_data started, messages_filepath {}, categories_filepath{}'.format(messages_filepath, categories_filepath))


def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    logging.info('clean_data started.')


def save_data(df: pd.DataFrame, database_filename: str):
    logging.info('save_data started.')


def check_for_existing_filepath(filepath: str) -> bool:
    """This method checks for the existence of the given file.
    In case the file does not exist, an exception is thrown.
    """
    return os.path.isfile(filepath)


def log_and_print(log_message: str):
    """This message outputs its given string on the console and
    as a log message with info level."""
    print(log_message)
    logging.info(log_message)


def main():
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
            df = load_data(messages_filepath, categories_filepath)

            log_and_print('Cleaning data...')
            df = clean_data(df)

            log_and_print('Saving data...\n    DATABASE: {}'.format(database_filepath))
            save_data(df, database_filepath)

            log_and_print('Cleaned data saved to database!')

        except FileNotFoundError as error:
            logging.error(error)
            print(error)
    
    else:
        log_and_print('Please provide the filepaths of the messages and categories '\
              'datasets as the first and second argument respectively, as '\
              'well as the filepath of the database to save the cleaned data '\
              'to as the third argument. \n\nExample: python process_data.py '\
              'disaster_messages.csv disaster_categories.csv '\
              'DisasterResponse.db')

    logging.info('Finished')

if __name__ == '__main__':
    main()