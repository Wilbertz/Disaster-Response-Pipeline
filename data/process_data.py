import sys
import os
import logging
import pandas as pd
from sqlalchemy import create_engine


def load_data(messages_filepath: str, categories_filepath: str) -> pd.DataFrame:
    logging.info(
        'load_data started, messages_filepath {}, categories_filepath{}'.format(messages_filepath, categories_filepath))
    messages = pd.read_csv(messages_filepath)
    categories = pd.read_csv(categories_filepath)
    df = pd.merge(messages, categories, on='id')

    return df


def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    logging.info('clean_data started.')

    categories = df['categories'].str.split(pat=';', expand=True)
    row = categories.iloc[0]
    category_colnames = row.apply(lambda x: x[:-2])
    categories.columns = category_colnames

    for column in categories:
        # set each value to be the last character of the string
        categories[column] = categories[column].apply(lambda x: x[:][-1])

        # convert column from string to numeric
        categories[column] = pd.to_numeric(categories[column])

        # replace all values that equals 2 to be encoded as 1
    categories = categories.replace(2, 1)

    df.drop('categories', axis=1, inplace=True)
    df = pd.concat([df, categories], axis=1)
    df.drop_duplicates(inplace=True)

    return df


def save_data(df: pd.DataFrame, database_filename: str) -> None:
    logging.info('save_data started.')
    engine = create_engine('sqlite:///' + database_filename)
    df.to_sql('Messages', engine, index=False, if_exists='replace', chunksize=500)


def check_for_existing_filepath(filepath: str) -> bool:
    """This method checks for the existence of the given file.
    In case the file does not exist, an exception is thrown.
    """
    return os.path.isfile(filepath)


def log_and_print(log_message: str) -> None:
    """This message outputs its given string on the console and
    as a log message with info level."""
    print(log_message)
    logging.info(log_message)


def main() -> None:
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
