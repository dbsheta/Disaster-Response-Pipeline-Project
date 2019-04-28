import sys

import pandas as pd
from sqlalchemy import create_engine


def load_data(messages_filepath, categories_filepath):
    """
    Load message and category data into datasets
    :param messages_filepath: path to messages data
    :param categories_filepath: path to categories data
    :return: pandas dataframe
    """
    messages = pd.read_csv(messages_filepath)
    categories = pd.read_csv(categories_filepath)
    df = messages.merge(categories, on='id', how='left')
    return df


def clean_data(df):
    """
    Clean the dataset
    :param df: dataset to clean
    :return: cleaned dataset
    """
    categories = df.categories.str.split(';', expand=True)
    row = categories.iloc[0]
    category_colnames = row.str.split('-').str.get(0)
    categories.columns = category_colnames
    for column in categories:
        # set each value to be the last character of the string
        categories[column] = categories[column].str.get(-1)

        # convert column from string to numeric
        categories[column] = pd.to_numeric(categories[column])
    df = df.drop(['categories'], axis=1)
    df = pd.concat([df, categories], axis=1)
    df = df.drop_duplicates()
    df.related = df.related.replace(2, 1)
    df = df.drop(['child_alone'], axis=1)
    return df


def save_data(df, database_filename):
    """
    Save dataset into sqlite
    :param df: dataframe
    :param database_filename: database name
    """
    engine = create_engine(f'sqlite:///{database_filename}')
    df.to_sql('DisasterResponse', engine, index=False)


def main():
    if len(sys.argv) == 4:

        messages_filepath, categories_filepath, database_filepath = sys.argv[1:]

        print('Loading data...\n    MESSAGES: {}\n    CATEGORIES: {}'
              .format(messages_filepath, categories_filepath))
        df = load_data(messages_filepath, categories_filepath)

        print('Cleaning data...')
        df = clean_data(df)

        print('Saving data...\n    DATABASE: {}'.format(database_filepath))
        save_data(df, database_filepath)

        print('Cleaned data saved to database!')

    else:
        print('Please provide the filepaths of the messages and categories ' \
              'datasets as the first and second argument respectively, as ' \
              'well as the filepath of the database to save the cleaned data ' \
              'to as the third argument. \n\nExample: python process_data.py ' \
              'disaster_messages.csv disaster_categories.csv ' \
              'DisasterResponse.db')


if __name__ == '__main__':
    main()
