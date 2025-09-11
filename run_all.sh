#!/usr/bin/env bash
# run_all.sh - convenience script to generate data, train models
set -euo pipefail

python generate_data.py
python preprocess_train.py --input data/agriculture_suitability.csv --out_dir models

echo "Done. To run the app: streamlit run app.py"
