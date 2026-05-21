"""
Zagon validacije za vsa merilna mesta.
Zagon: uv run python gx/run_checkpoint.py
"""

import sys
from pathlib import Path
import great_expectations as gx

gx_dir = Path(__file__).parent.parent.parent / "gx"
context = gx.get_context(context_root_dir=str(gx_dir))

# Poišči vsa merilna mesta
preprocessed_dir = Path(__file__).parent.parent / "data/preprocessed/air"
stations = [f.stem for f in preprocessed_dir.glob("*.csv")]

if not stations:
    print("Ni najdenih merilnih mest!")
    sys.exit(1)

print(f"Validiranje merilnih mest: {stations}")

all_passed = True

for station in stations:
    checkpoint_name = f"air_quality_checkpoint_{station}"

    try:
        checkpoint = context.get_checkpoint(checkpoint_name)
    except Exception:
        print(f"⚠️  Checkpoint za {station} ne obstaja. Najprej poženi setup_ge.py.")
        all_passed = False
        continue

    run_id = f"air_quality_run_{station}"
    checkpoint_result = checkpoint.run(run_id=run_id)

    if checkpoint_result["success"]:
        print(f"✅ Validacija uspešna za postajo {station}!")
    else:
        print(f"❌ Validacija neuspešna za postajo {station}!")
        all_passed = False

# Generiraj poročilo
context.build_data_docs()
print("\n📄 Poročilo generirano v: gx/uncommitted/data_docs/local_site/index.html")

if all_passed:
    print("✅ Vse validacije so uspešne!")
    sys.exit(0)
else:
    print("❌ Nekatere validacije so bile neuspešne!")
    sys.exit(1)