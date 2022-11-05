import pickle

from catboost import CatBoostRegressor
from loguru import logger
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
import pandas as pd

from vectorizers import MyVectorizer, DataPreparation

SEED = 42
DATA_PATH = "data/processed/main_data.csv"
COUNTRIES_DATA_PATH = "data/processed/countries_data.csv"
MODEL_FILE = "artefacts/catboostregressor.bin"
VECTORIZER_PATH = "artefacts/vectorizer.pkl"


def load_data() -> (pd.DataFrame, pd.Series):
    df = pd.read_csv(DATA_PATH)
    y = df.car_purchase_amount
    df.drop(["car_purchase_amount"], axis=1, inplace=True)
    return df, y


def get_model() -> CatBoostRegressor:
    catboost_params = {'depth': 3, 'iterations': 200, 'learning_rate': 0.1}
    return CatBoostRegressor(**catboost_params)


def main():
    logger.info("Loading data")

    df, y = load_data()

    df_train, df_test, y_train, y_test = train_test_split(df, y, test_size=0.2, random_state=SEED, shuffle=True)

    logger.info("Vectorizing data")

    vectorizer = MyVectorizer()
    X_train = vectorizer.fit_transform(df_train)
    X_test = vectorizer.transform(df_test)

    logger.info("Training model")
    model = get_model()
    model.fit(X_train, y_train, verbose=False)

    logger.info("Evaluating model")
    train_error = mean_squared_error(y_train, model.predict(X_train)) ** 0.5
    test_error = mean_squared_error(y_test, model.predict(X_test)) ** 0.5
    logger.info(f"Train error: {train_error}")
    logger.info(f"Test error: {test_error}")

    logger.info("Prepare vectorizer for inference")
    dataPreparation = DataPreparation(
        vectorizer=vectorizer,
        countries_df=pd.read_csv(COUNTRIES_DATA_PATH)
    )

    logger.info("Saving model")
    model.save_model(MODEL_FILE)
    with open(VECTORIZER_PATH, "wb") as pkl_file:
        pickle.dump(dataPreparation, pkl_file)

    logger.success("Done")


if __name__ == "__main__":
    main()
