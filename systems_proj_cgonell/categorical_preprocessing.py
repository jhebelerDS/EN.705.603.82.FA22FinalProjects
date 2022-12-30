import pandas as pd


class Categorical_Preprocessing:
    def __init__(self):
        self.df = None

    def __getitem__(self, item):
        item = self.df
        return item

    def read_data(self, file_name: str):
        read = pd.read_csv(file_name)
        self.df = pd.DataFrame(read)
        return self.df

    def find_missing(self):
        missing = self.df.isnull().sum()
        return missing

    def data_type(self):
        data_type = self.df.dtypes
        return print(data_type)

    def missing_mean(self, col_name):
        self.df[col_name] = self.df[col_name].fillna(self.df[col_name].mean())
        return self.df

    def unique_values(self):
        unique_vals = {column: len(self.df[column].unique())
                       for column in self.df.columns if self.df.dtypes[column] == 'object'}
        return unique_vals

    def unique_col_vals(self, col_name):
        values_col = self.df[col_name].unique()
        return values_col

    def replace_mapping(self, col_name, mapping):
        self.df[col_name] = self.df[col_name].replace(mapping)
        return self.df[col_name]

    def boolean_values(self):
        for column in self.df.columns:
            if self.df.dtypes[column] == 'bool':
                self.df[column] = self.df[column].astype(int)
        return self.df

    def onehot_encode(self, columns, prefixes):
        df = self.df.copy()
        for column, prefix in zip(columns, prefixes):
            dummies = pd.get_dummies(df[column], prefix=prefix)
            df = pd.concat([df, dummies], axis=1)
            df = df.drop(column, axis=1)
        return df

    def rename_col(self, col_name):
        self.df = self.df.rename(col_name, axis=1)
        return self.df

    def drop_columns(self, col_name):
        self.df = self.df.drop(col_name, axis=1)
        return self.df

    def dummies(self, col_name):
        dummies = pd.get_dummies(self.df[col_name])
        self.df = pd.concat([self.df, dummies], axis='columns')

        return self.df

    def copy(self):
        df = self.df.copy()
        return df

    def numeric(self, col_name):
        self.df[col_name] = pd.to_numeric(self.df[col_name], errors='coerce')
        return self.df

    def convert_int(self):
        for column in self.df.columns:
            if self.df.dtypes[column] == 'uint8':
                self.df[column] = self.df[column].astype(int)
        return self.df

    def remaining_non_numeric(self):
        return print("Remaining non-numeric columns:", (self.df.dtypes == 'object').sum())
