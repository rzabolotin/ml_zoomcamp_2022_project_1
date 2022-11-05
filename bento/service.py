import bentoml
from bentoml.io import JSON
from pydantic import BaseModel

BENTO_MODEL_NAME = "what_price_catboost:latest"

model_ref = bentoml.catboost.get(BENTO_MODEL_NAME)
vectorizer = model_ref.custom_objects["vectorizer"]
my_runner = model_ref.to_runner()

svc = bentoml.Service("what_price", runners=[my_runner])


class CustomerData(BaseModel):
    customer_name: str
    customer_email: str
    country: str
    gender: int
    age: float
    annual_salary: float
    credit_card_debt: float
    net_worth: float


@svc.api(input=JSON(pydantic_model=CustomerData), output=JSON())
async def predict(input_dict: CustomerData) -> dict:
    input_data = vectorizer.transform(input_dict.dict())
    prediction = await my_runner.predict.async_run(input_data)
    return {
        "customer": input_dict.customer_name,
        "possible_car_price": round(prediction[0],2)
    }
