import os

temp_scripts = [
    "test_urls.py",
    "check_data.py",
    "force_clean.py",
    "verify_dataset.py",
    "analyze_skips.py",
    "check_duplicates.py",
    "check_duplicate_details.py",
    "check_state.py",
]

for f in temp_scripts:
    if os.path.exists(f):
        os.remove(f)
        print(f"Removed: {f}")

print("Cleanup complete")