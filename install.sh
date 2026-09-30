#!/usr/bin/env bash
# Install the unified `thalkak` conda environment.
set -eo pipefail

# Make `conda activate` available in this non-interactive shell.
eval "$(conda shell.bash hook)"

# 1. Create env + install conda/pip deps
conda env create -f environment.yml
conda activate thalkak

# 2. --no-deps installs (boltz / chai / protenix / opendde / esm)
pip install --no-deps -r requirements-nodeps.txt

# 3. Patch alphafold-colabfold templates.py: kalign realignment first,
# fuzzy regex match only as fallback.
patch -p1 -d "$CONDA_PREFIX/lib/python3.12/site-packages" \
    < MSA/script/patches/alphafold_templates_kalign_first.patch

echo
echo "Done. Activate with:  conda activate thalkak"
