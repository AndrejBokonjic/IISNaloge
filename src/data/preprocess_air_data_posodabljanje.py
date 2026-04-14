import os

import numpy as np
import pandas as pd
from lxml import etree as ET


def preprocess_air_data():
    # Open XML file
    with open("data/raw/air/air_data.xml", "rb") as file:
        tree = ET.parse(file)
        root = tree.getroot()

    # Extract and print data
    print(f"Version: {root.attrib['verzija']}")
    print(f"Source: {root.find('vir').text}")
    print(f"Suggested Capture: {root.find('predlagan_zajem').text}")
    print(f"Suggested Capture Period: {root.find('predlagan_zajem_perioda').text}")
    print(f"Preparation Date: {root.find('datum_priprave').text}")

    sifra_vals = set(tree.xpath('//postaja/@sifra'))

    postaja_elements = tree.xpath(f'//postaja[@sifra="E410"]')

    # Initialize an empty DataFrame
    columns = ["date_to", "PM10", "PM2.5"]
    df = pd.DataFrame(columns=columns)

    # Check if csv file already exists
    if os.path.exists("data/preprocessed/air/E410.csv"):
        df = pd.read_csv("data/preprocessed/air/E410.csv")

    # Convert the XML data to a DataFrame
    for postaja in postaja_elements:
        date_to = postaja.find('datum_do').text
        pm10 = postaja.find('pm10').text if postaja.find('pm10') is not None else np.nan
        pm2_5 = postaja.find('pm2.5').text if postaja.find('pm2.5') is not None else np.nan

        # Append the data as a new row in the DataFrame
        df = pd.concat([df, pd.DataFrame([[date_to, pm10, pm2_5]], columns=columns)], ignore_index=True)

    # Filter unique "datum_do" values
    df = df.drop_duplicates(subset=["date_to"])

    # Sort the DataFrame by the "date_to" column
    df = df.sort_values(by="date_to")

    # Replace string values
    df = df.replace("", np.nan)
    df = df.replace("<1", 1)
    df = df.replace("<2", 2)

    # Save the DataFrame to a CSV file
    df.to_csv("data/preprocessed/air/E410.csv", index=False)


if __name__ == "__main__":
    preprocess_air_data()