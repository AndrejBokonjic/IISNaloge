import numpy as np
import pandas as pd
from lxml import etree as ET
from pathlib import Path

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

    # Get all station codes
    sifra_vals = set(tree.xpath('//postaja/@sifra'))

    # ensure output folder exists
    Path("data/preprocessed/air").mkdir(parents=True, exist_ok=True)

    # loop through all stations
    for sifra in sifra_vals:
        postaja_elements = tree.xpath(f'//postaja[@sifra="{sifra}"]')

        rows = []

        for postaja in postaja_elements:
            date_to = postaja.find('datum_do').text
            pm10 = postaja.find('pm10').text if postaja.find('pm10') is not None else np.nan
            pm2_5 = postaja.find('pm2.5').text if postaja.find('pm2.5') is not None else np.nan

            rows.append([date_to, pm10, pm2_5])

        df = pd.DataFrame(rows, columns=["date_to", "pm10", "pm2_5"])

        df = df.drop_duplicates(subset=["date_to"])

        # Sort
        df = df.sort_values(by="date_to")

        # Clean values
        df = df.replace("", np.nan)
        df = df.replace("<1", 1)
        df = df.replace("<2", 2)

        # Save separate CSV per station
        output_path = f"data/preprocessed/air/{sifra}.csv"
        df.to_csv(output_path, index=False)

        print(f"Saved: {output_path}")


if __name__ == "__main__":
    preprocess_air_data()