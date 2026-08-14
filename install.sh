#!/usr/bin/env bash
# Install the unified `thalkak` conda environment.
set -eo pipefail

# Make `conda activate` available in this non-interactive shell.
eval "$(conda shell.bash hook)"

# 1. Create env + install conda/pip deps
conda env create -f environment.yml
conda activate thalkak

# 2. --no-deps installs (boltz / chai / protenix)
pip install --no-deps -r requirements-nodeps.txt

# 3. Pin the Biohub transformers fork to the ESMFold2-matched commit.
# esm installs transformers @ main transitively, so re-pin it here
# (--force-reinstall: both commits report version 4.57.6, so pip would
# otherwise consider the requirement already satisfied and skip).
pip install --no-deps --force-reinstall \
    "transformers @ git+https://github.com/Biohub/transformers.git@ef32577f55da19a4989cd7b22e004dc43a4998cb"

# 4. Patch alphafold-colabfold templates.py: kalign realignment first,
# fuzzy regex match only as fallback.
patch -p1 -d "$CONDA_PREFIX/lib/python3.12/site-packages" \
    < MSA/script/patches/alphafold_templates_kalign_first.patch

echo
echo "Done. Activate with:  conda activate thalkak"
