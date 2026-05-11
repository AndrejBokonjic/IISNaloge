"""
Zaženi enkrat lokalno za inicializacijo GE datasource, expectations suite in checkpoint.
Pogoj: data/preprocessed/air/*.csv datoteke morajo že obstajati.

Zagon: uv run python gx/setup_ge.py
"""

import json
from pathlib import Path

import great_expectations as gx

context = gx.get_context()

# Poišči vsa merilna mesta
preprocessed_dir = Path("data/preprocessed/air")
stations = [f.stem for f in preprocessed_dir.glob("*.csv")]

if not stations:
    print("Ni najdenih CSV datotek v data/preprocessed/air/. Najprej poženi DVC pipeline.")
    exit(1)

print(f"Najdena merilna mesta: {stations}")

for station in stations:
    print(f"\n--- Nastavljam GE za postajo: {station} ---")

    datasource_name = f"air_quality_{station}"
    data_asset_name = f"air_quality_data_{station}"
    expectation_suite_name = f"air_quality_suite_{station}"
    checkpoint_name = f"air_quality_checkpoint_{station}"

    # 1. Ustvari datasource
    try:
        datasource = context.sources.add_pandas_filesystem(
            name=datasource_name,
            base_directory="data"
        )
    except Exception:
        datasource = context.get_datasource(datasource_name)
        print(f"  Datasource {datasource_name} že obstaja.")

    # 2. Dodaj data asset
    try:
        data_asset = datasource.add_csv_asset(
            name=data_asset_name,
            batching_regex=rf"preprocessed/air/{station}\.csv"
        )
    except Exception:
        data_asset = datasource.get_asset(data_asset_name)
        print(f"  Asset {data_asset_name} že obstaja.")

    # 3. Ustvari expectations suite
    expectation_suite = context.add_or_update_expectation_suite(
        expectation_suite_name=expectation_suite_name
    )

    # 4. Ustvari validator in poženi data assistant
    batch_request = data_asset.build_batch_request()
    validator = context.get_validator(
        batch_request=batch_request,
        expectation_suite_name=expectation_suite_name
    )

    print(f"  Poganjam OnboardingDataAssistant za {station}...")
    data_assistant_result = context.assistants.onboarding.run(
        validator=validator,
        exclude_column_names=[]
    )

    # 5. Shrani samodejno generirano suite
    expectation_suite = data_assistant_result.get_expectation_suite()
    context.save_expectation_suite(
        expectation_suite=expectation_suite,
        expectation_suite_name=expectation_suite_name
    )

    # 6. Prepiši z ročno prilagojenimi expectations (smiselna pravila)
    suite_path = Path(f"gx/expectations/{expectation_suite_name}.json")
    if suite_path.exists():
        with open(suite_path, "r") as f:
            existing = json.load(f)

        # Posodobi z ročno definiranimi expectations
        manual_expectations = [
            {
                "expectation_type": "expect_table_row_count_to_be_between",
                "kwargs": {"max_value": None, "min_value": 1},
                "meta": {}
            },
            {
                "expectation_type": "expect_table_columns_to_match_set",
                "kwargs": {
                    "column_set": ["date_to", "pm2_5", "pm10"],
                    "exact_match": None
                },
                "meta": {}
            },
            {
                "expectation_type": "expect_column_values_to_be_unique",
                "kwargs": {"column": "date_to"},
                "meta": {}
            },
            {
                "expectation_type": "expect_column_values_to_not_be_null",
                "kwargs": {"column": "date_to"},
                "meta": {}
            },
            {
                "expectation_type": "expect_column_value_lengths_to_be_between",
                "kwargs": {
                    "column": "date_to",
                    "max_value": 20,
                    "min_value": 10,
                    "mostly": 1.0,
                    "strict_max": False,
                    "strict_min": False
                },
                "meta": {}
            },
            {
                "expectation_type": "expect_column_values_to_be_between",
                "kwargs": {
                    "column": "pm10",
                    "min_value": 0,
                    "max_value": 300,
                    "strict_min": True,
                    "strict_max": False,
                    "mostly": 0.95
                },
                "meta": {}
            },
            {
                "expectation_type": "expect_column_values_to_be_between",
                "kwargs": {
                    "column": "pm2_5",
                    "min_value": 0,
                    "max_value": 300,
                    "strict_min": True,
                    "strict_max": False,
                    "mostly": 0.95
                },
                "meta": {}
            },
        ]

        existing["expectations"] = manual_expectations
        with open(suite_path, "w") as f:
            json.dump(existing, f, indent=2)
        print(f"  ✅ Expectations suite posodobljen z ročnimi pravili.")

    # 7. Ustvari checkpoint
    context.add_or_update_checkpoint(
        name=checkpoint_name,
        validations=[
            {
                "batch_request": batch_request,
                "expectation_suite_name": expectation_suite_name
            }
        ],
    )
    print(f"  ✅ Checkpoint {checkpoint_name} ustvarjen.")

print("\n✅ Setup zaključen za vsa merilna mesta!")
print("Zdaj lahko poženeš: uv run python gx/run_checkpoint.py")