import sys
from pathlib import Path

import great_expectations as gx
import yaml

params = yaml.safe_load(open("params.yaml"))["preprocess"]
station = params["station"]

context = gx.get_context()

# Poišči vsa merilna mesta (vse CSV datoteke v preprocessed/air/)
preprocessed_dir = Path("data/preprocessed/air")
stations = [f.stem for f in preprocessed_dir.glob("*.csv")]

if not stations:
    print("Ni najdenih merilnih mest!")
    sys.exit(1)

print(f"Validiranje merilnih mest: {stations}")

all_passed = True

for station in stations:
    datasource_name = f"air_quality_{station}"
    data_asset_name = f"air_quality_data_{station}"
    expectation_suite_name = f"air_quality_suite_{station}"
    checkpoint_name = f"air_quality_checkpoint_{station}"

    try:
        checkpoint = context.get_checkpoint(checkpoint_name)
    except Exception:
        print(f"Checkpoint za {station} ne obstaja, preskakujem...")
        continue

    run_id = f"air_quality_run_{station}"
    checkpoint_result = checkpoint.run(run_id=run_id)

    context.build_data_docs()

    if checkpoint_result["success"]:
        print(f"✅ Validacija uspešna za postajo {station}!")
    else:
        print(f"❌ Validacija neuspešna za postajo {station}!")
        all_passed = False

if all_passed:
    print("✅ Vse validacije so uspešne!")
    sys.exit(0)
else:
    print("❌ Nekatere validacije so bile neuspešne!")
    sys.exit(1)