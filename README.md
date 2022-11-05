# Project: At what price will you buy a car? 
Project created at cohort 2022 of ML Zoomcamp course.

The solved problem is a regression problem. We try to predict the amount of money that a customer willing to spend on a car, knowing some features of a customer.  
Knowledge of this will help car dealers to make better decisions on whom and how to sell their cars.

During the EDA and model creation I realized that the dataset is synthetic and the target variable is a linear combination of the features. And the model can be described as a linear combination of features.  

But I still decided to create a model, and make all steps of the project

# Sources of data
In this project, I used the data from the [ANN - Car Sales Price Prediction](https://www.kaggle.com/datasets/yashpaloswal/ann-car-sales-price-prediction) dataset on Kaggle.

**Dataset description:**  
As a vehicle salesperson, you would like to create a model that can estimate the overall amount that consumers would spend given the following characteristics:
- customer name
- customer email
- country
- gender
- age
- annual salary
- credit card debt
- and net worth

While EDA I decided to enrich the dataset with the information about country. For this I used the [Countries of the world](https://www.kaggle.com/datasets/fernandol/countries-of-the-world) from kaggle.  
I added the following features:
- Region
- Population
- Area (sq. mi.)
- Pop. Density (per sq. mi.)
- Coastline (coast/area ratio)
- GDP ($ per capita)
- Birthrate
- Deathrate

# Project structure:
- [notebooks](notebooks) - Folder with notebooks
  - [EDA](<notebooks/1.%20EDA%20&%20data%20preparation.ipynb>) - Exploratory data analysis and data preparation
  - [Model selection](<notebooks/2.%20Model%20training.ipynb>) - Model creation and selection
- [scripts](scripts) - Folder with scripts
  - [data preparation](scripts/1_data_preparation.py) - Script for data preparation
  - [model training](scripts/2_model_training.py) - Script for model training
  - [model evaluation](scripts/3_model_evaluation.py) - Script for model evaluation
- [data](data) - Folder with data
  - [raw](data/raw) - Folder with raw data
  - [processed](data/processed) - Folder with processed data (filled by notebooks/scripts)
- [artifacts](artefacts) - Folder with artifacts of the project (model & vectorizer)
- [bento](bento) - Folder with bentoML service
- [docker](docker) - Folder with docker files
- [README.md](README.md) - Project description
- [pipenv](Pipfile) - Pipenv file with project dependencies

# How to run the project:
1. Clone the repository
2. Install the dependencies
```bash
pipenv install
```
3. Prepare the data for training
```bash
pipenv run python scripts/1_data_preparation.py
```
3. Train catboost model
```bash
pipenv run python scripts/2_model_training.py
```
4. Run sample prediction
```bash
pipenv run python scripts/3_model_evaluation.py
```
5. Run local bentoml service
```bash
cd bento
pipenv run bentoml serve --production
```
Then you can test API on [http://localhost:3000](http://localhost:3000)
or by curl
```bash
curl -X 'POST' \
  'http://localhost:3000/predict' \
  -H 'accept: application/json' \
  -H 'Content-Type: application/json' \
  -d '{"customer_name": "Martina Avila",
               "customer_email": "cubilia.Curae.Phasellus@quisaccumsanconvallis.edu",
               "country": "Bulgaria",
               "gender": 0,
               "age": 42,
               "annual_salary": 62812,
               "credit_card_debt": 11609.5,
               "net_worth": 238961.2}'
```

# Containerization

The project is containerized with Bentoml.  
To build the container, run the following command:
```bash
cd bento
pipenv run bentoml build
pipenv run bentoml containerize what_price:latest
```
To run the container, run the following command:
```bash
docker run -it --rm -p 3000:3000 what_price:uqfzsys5isn3caav
```

Then you can test API on [http://localhost:3000](http://localhost:3000)

The Dockerfile created by bentoml is located in the [docker folder](docker).


# Used libraries & tools
- [pandas](https://pandas.pydata.org/)
- [numpy](https://numpy.org/)
- [matplotlib](https://matplotlib.org/)
- [seaborn](https://seaborn.pydata.org/)
- [scikit-learn](https://scikit-learn.org/stable/)
- [catboost](https://catboost.ai/)
- [BentoML](https://bentoml.org/)
- [Dockers](https://www.docker.com/)
- [Pipenv](https://pypi.org/project/pipenv/)
