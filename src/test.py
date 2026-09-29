import os
import sys
import platform
from importlib.metadata import version, PackageNotFoundError

import pandas as pd

ARTIFACTS_DIR = r"D:\MovieReccomendationSystem\artifacts"

# ---------- 1. CSV check ----------
print("=" * 60)
print("1. movies_slim_v2.csv")
print("=" * 60)
df = pd.read_csv(os.path.join(ARTIFACTS_DIR, "movies_slim_v2.csv"))
pd.set_option("display.width", 250)
pd.set_option("display.max_columns", None)
pd.set_option("display.max_colwidth", 60)

print("Shape:", df.shape)
print("Columns:", df.columns.tolist())
print("\nhead(3):")
print(df.head(3).to_string())
print("\nMissing values:")
print(df.isna().sum().to_string())
print("\nposter_path missing count:", df["poster_path"].isna().sum())
print("Duplicate labels:", df["label"].duplicated().sum())

print("\nChip titles present in dataset?")
for t in ["Inception", "The Dark Knight", "Interstellar", "Spirited Away"]:
    m = df[df["title"].str.lower() == t.lower()]
    print(f"  {t:18s} -> {m['label'].tolist()[:3]}")

# ---------- 2. Versions ----------
print("\n" + "=" * 60)
print("2. Versions")
print("=" * 60)
print("Python:", sys.version.split()[0], "|", platform.platform())
for pkg in ["scikit-learn", "scipy", "numpy", "pandas", "joblib", "streamlit", "requests"]:
    try:
        print(f"{pkg:14s} {version(pkg)}")
    except PackageNotFoundError:
        print(f"{pkg:14s} NOT INSTALLED")

# ---------- 3. File sizes ----------
print("\n" + "=" * 60)
print("3. File sizes (MB)")
print("=" * 60)
for f in sorted(os.listdir(ARTIFACTS_DIR)):
    size = os.path.getsize(os.path.join(ARTIFACTS_DIR, f)) / (1024 * 1024)
    print(f"{f:28s} {size:8.2f} MB")