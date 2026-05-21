import os
import sys
from pathlib import Path

import pandas as pd
from evidently import Report
from evidently.presets.dataset_stats import DataSummaryPreset
from evidently.presets.drift import DataDriftPreset

# Poišči vsa merilna mesta
preprocessed_dir = Path("data/preprocessed/air")
stations = [f.stem for f in preprocessed_dir.glob("*.csv")]

if not stations:
    print("Ni najdenih merilnih mest!")
    sys.exit(1)

print(f"Testiranje merilnih mest: {stations}")

Path("reports").mkdir(parents=True, exist_ok=True)

all_passed = True

for station in stations:
    print(f"\n--- Testiram postajo: {station} ---")

    current = pd.read_csv(f"data/preprocessed/air/{station}.csv")
    reference_path = f"data/reference/air/{station}.csv"

    if not os.path.exists(reference_path):
        print(f"Referenčna datoteka ne obstaja. Kopiram trenutne podatke v {reference_path}.")
        os.makedirs(os.path.dirname(reference_path), exist_ok=True)
        current.to_csv(reference_path, index=False)

    reference = pd.read_csv(reference_path)

    # Odstrani datume
    del reference["date_to"]
    del current["date_to"]

    # Odstrani stolpce ki so popolnoma prazni v referenci ali trenutnih podatkih
    empty_cols = [col for col in current.columns
                  if current[col].isna().all() or reference[col].isna().all()]
    if empty_cols:
        print(f"  Preskakujem prazne stolpce: {empty_cols}")
        current = current.drop(columns=empty_cols)
        reference = reference.drop(columns=empty_cols)

    if current.empty or len(current.columns) == 0:
        print(f"  ⚠️  Postaja {station} nima uporabnih podatkov, preskakujem.")
        continue

    report = Report([
            DataSummaryPreset(),
            DataDriftPreset(),
        ],
        include_tests=True
    )

    result = report.run(reference_data=reference, current_data=current)
    result.save_html(f"reports/data_testing_report_{station}.html")

    result_dict = result.dict()
    station_passed = True
    if "tests" in result_dict:
        for test in result_dict["tests"]:
            if "status" in test and test["status"] != "SUCCESS":
                station_passed = False
                break

    if not station_passed:
        print(f"❌ Testi neuspešni za postajo {station}.")
        all_passed = False
    else:
        print(f"✅ Testi uspešni za postajo {station}.")
        os.remove(reference_path)
        current = pd.read_csv(f"data/preprocessed/air/{station}.csv")
        current.to_csv(reference_path, index=False)

if not all_passed:
    print("\n❌ Nekateri testi so bili neuspešni!")
    sys.exit(1)
else:
    print("\n✅ Vsi testi so uspešni!")
    sys.exit(0)