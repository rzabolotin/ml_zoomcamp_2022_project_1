import pickle
from catboost import CatBoostRegressor
from loguru import logger
from pprint import pformat

VECTORIZER_FILE = "artefacts/vectorizer.pkl"
MODEL_FILE = "artefacts/catboostregressor.bin"

with open(VECTORIZER_FILE, "rb") as f:
    vectorizer = pickle.load(f)

model = CatBoostRegressor()
model.load_model(MODEL_FILE)

sample_data = {"customer_name": "Martina Avila",
               "customer_email": "cubilia.Curae.Phasellus@quisaccumsanconvallis.edu",
               "country": "Bulgaria",
               "gender": 0,
               "age": 42,
               "annual_salary": 62812,
               "credit_card_debt": 11609.5,
               "net_worth": 238961.2}


def main():
    logger.info("Starting prediction")
    logger.info(f"Sample data: {pformat(sample_data)}")
    sample_data_vectorized = vectorizer.transform(sample_data)
    prediction = model.predict(sample_data_vectorized)
    logger.success(f"Predicted car purchase amount: {prediction[0]:.2f}")


if __name__ == '__main__':
    main()
