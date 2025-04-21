import pandas as pd

def read_csv_date_index(infile, ncol=None):
    """ read a CSV file into a dataframe and set index to type
    datetime.date """
    df = pd.read_csv(infile, parse_dates=['Date'])
    df.set_index('Date', inplace=True)
    df.index = df.index.date
    if ncol is not None:
        df = df.iloc[:, :ncol]
    return df
