from sklearn.base import BaseEstimator
from sklearn.feature_extraction import DictVectorizer
from sklearn.preprocessing import StandardScaler
import pandas as pd


class MyVectorizer(BaseEstimator):
    def __init__(self):
        self._scaler = StandardScaler()
        self._vectorizer = DictVectorizer(sparse=False)
        self._numerical_features = [
            "age",
            "annual_salary",
            "credit_card_debt",
            "net_worth",
            "population",
            "area_sq_mi",
            "pop_density_per_sq_mi",
            "coastline_coastarea_ratio",
            "gdp__per_capita",
            "birthrate",
            "deathrate"

        ]

    def fit(self, df):
        self._scaler.fit(df[self._numerical_features].values)
        row_dicts = df.to_dict(orient='records')
        self._vectorizer.fit(row_dicts)

    def transform(self, df):
        df_ = df.copy()
        df_.loc[:, self._numerical_features] = self._scaler.transform(df[self._numerical_features].values)
        row_dicts = df_.to_dict(orient='records')
        return self._vectorizer.transform(row_dicts)

    def fit_transform(self, df):
        self.fit(df)
        return self.transform(df)

    def get_feature_names(self):
        return self._vectorizer.get_feature_names_out()


class DataPreparation(BaseEstimator):
    def __init__(self, vectorizer, countries_df):
        self._vectorizer = vectorizer
        self._countries_df = countries_df

    def fit(self):
        pass

    def transform(self, value):
        if type(value) == dict:
            df_ = pd.DataFrame.from_dict({k: [v] for k, v in value.items()})
        else:
            df_ = value.copy()

        df_["customer_email_suffix"] = df_.customer_email.str.split('@').apply(lambda x: x[1][x[1].find('.'):])
        df_.country = df_.country \
            .str.replace("Bahamas", "Bahamas, The") \
            .str.replace("Central African Republic", "Central African Rep.") \
            .str.replace("Falkland Islands", "Argentina") \
            .str.replace("Macao", "Macau") \
            .str.replace("Micronesia", "Micronesia, Fed. St.") \
            .str.replace("Montenegro", "Serbia") \
            .str.replace("Trinidad and Tobago", "Trinidad & Tobago") \
            .str.replace("United Kingdom (Great Britain)", "United Kingdom", regex=False) \
            .str.replace("United States Minor Outlying Islands", "United States") \
            .str.replace("Viet Nam", "Vietnam") \
            .str.replace("Virgin Islands, British", "United Kingdom") \
            .str.replace("Virgin Islands, United States", "United States") \
            .str.replace("Bosnia and Herzegovina", "Bosnia & Herzegovina")

        df_.loc[~df_.country.isin(self._countries_df.country), "country"] = "UNKNOWN"

        df_ = df_.merge(self._countries_df, on='country')
        columns_to_drop = [
            "customer_name",
            "customer_email",
            "country"
        ]
        df_.drop(columns_to_drop, axis=1, inplace=True)

        return self._vectorizer.transform(df_)