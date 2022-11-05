import pandas as pd
from loguru import logger

MAIN_DATA_PATH = "../data/raw/car_purchasing.csv"
COUNTRIES_DATA_PATH = "../data/raw/countries_of_the_world.csv"

MAIN_DATA_OUTPUT_PATH = "../data/processed/main_data.csv"
COUNTRIES_DATA_OUTPUT_PATH = "../data/processed/countries_data.csv"


def load_main_data(countries_df: pd.DataFrame) -> pd.DataFrame:
    logger.info("Loading raw customer's data")
    df = pd.read_csv(MAIN_DATA_PATH, encoding='ISO-8859-1')

    df.columns = df.columns.str.replace(" ", "_") \
        .str.replace("-", "") \
        .str.lower()

    # extract email domain
    df["customer_email_suffix"] = df.customer_email.str.split('@').apply(lambda x: x[1][x[1].find('.'):])

    # enrich with countries data
    df.country = df.country \
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

    df.loc[~df.country.isin(countries_df.country), "country"] = "UNKNOWN"

    logger.info("Merging with countries data")
    df = df.merge(countries_df, how='left', on='country')

    # drop unnecessary columns
    columns_to_drop = [
        "customer_name",
        "customer_email",
        "country"
    ]

    df.drop(columns_to_drop, axis=1, inplace=True)

    return df


def load_countries_data() -> pd.DataFrame:
    logger.info("Loading raw countries data")
    countries_df = pd.read_csv(COUNTRIES_DATA_PATH, decimal=",")

    # Select only columns we need
    countries_columns = [
        "Country",
        "Region",
        "Population",
        "Area (sq. mi.)",
        "Pop. Density (per sq. mi.)",
        "Coastline (coast/area ratio)",
        "GDP ($ per capita)",
        "Birthrate",
        "Deathrate"

    ]
    countries_df = countries_df[countries_columns]

    countries_df.columns = countries_df.columns \
        .str.replace(" ", "_") \
        .str.replace("[-|(|)|\.|//|\$|\%]", "", regex=True) \
        .str.lower()

    # Fill missing values & strip whitespaces
    countries_df.fillna(0, inplace=True)
    countries_df.country = countries_df.country.str.strip()

    # Add row for UNKNOWN country
    unknown_country = pd.DataFrame.from_dict({
        "country": ["UNKNOWN"],
        "region": ["UNKNOWN"],
        "population": [0],
        "area_sq_mi": [0],
        "pop_density_per_sq_mi": [0],
        "coastline_coastarea_ratio": [0],
        "gdp__per_capita": [0],
        "birthrate": [0],
        "deathrate": [0],
    }, orient="columns")
    countries_df = pd.concat([countries_df, unknown_country])

    return countries_df


def main():
    countries_df = load_countries_data()
    main_df = load_main_data(countries_df)

    logger.info("Saving processed data")
    main_df.to_csv(MAIN_DATA_OUTPUT_PATH, index=False)
    logger.info(f"Saved main data to {MAIN_DATA_OUTPUT_PATH}")
    countries_df.to_csv(COUNTRIES_DATA_OUTPUT_PATH, index=False)
    logger.info(f"Saved counties data to {COUNTRIES_DATA_OUTPUT_PATH}")
    logger.success("Data saved successfully")


if __name__ == "__main__":
    main()
