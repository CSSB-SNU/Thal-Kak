"""Force-field stack assembly for all-atom OpenMM relaxation (method ``openmm``).

Adapted from OpenMMDL (``openmmdl/openmmdl_simulation/scripts/forcefield_water.py``,
https://github.com/wolberlab/OpenMMDL), used under the MIT License:

    Copyright (c) 2024 Valerij Talagayev, Yu Chen, Niklas Piet Doering &
    Leon Obendorf (Wolber lab)

    Permission is hereby granted, free of charge, to any person obtaining a copy
    of this software and associated documentation files (the "Software"), to
    deal in the Software without restriction, including without limitation the
    rights to use, copy, modify, merge, publish, distribute, sublicense, and/or
    sell copies of the Software, and to permit persons to whom the Software is
    furnished to do so, subject to the following conditions:

    The above copyright notice and this permission notice shall be included in
    all copies or substantial portions of the Software.

    THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
    IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
    FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
    AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
    LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
    OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
    SOFTWARE.
"""

from openmm import app as openmm_app

# Main all-atom bundle: protein ff19SB + DNA OL21 + RNA OL3 + lipid21.
FF_MAIN = "amber19-all.xml"
# Carbohydrates (N-glycan core etc.); requires GLYCAM residue naming.
FF_GLYCAN = "amber14/GLYCAM_06j-1.xml"
# Monatomic ion templates. Loading only contributes templates -- never water.
FF_IONS = "amber14/tip3p.xml"

IMPLICIT_SOLVENT_FF = {
    "obc2": "implicit/obc2.xml",
    "gbn2": "implicit/gbn2.xml",
    "none": None,
}


def forcefield_files(implicit_solvent):
    """Return the ordered list of XML force-field files for the given solvent."""
    if implicit_solvent not in IMPLICIT_SOLVENT_FF:
        raise ValueError(
            f"unknown implicit_solvent {implicit_solvent!r} "
            f"(choose from {sorted(IMPLICIT_SOLVENT_FF)})"
        )
    files = [FF_MAIN, FF_GLYCAN, FF_IONS]
    if IMPLICIT_SOLVENT_FF[implicit_solvent] is not None:
        files.append(IMPLICIT_SOLVENT_FF[implicit_solvent])
    return files


def make_forcefield(implicit_solvent, template_generators=None):
    """Build an OpenMM ForceField with the all-atom bundle.

    Args:
        implicit_solvent: ``obc2`` | ``gbn2`` | ``none``.
        template_generators: optional iterable of small-molecule template
            generators (GAFF/SMIRNOFF) to register for ligands. Added by the
            ligand phase; ``None`` in the polymer/ion-only path.

    Returns:
        openmm.app.ForceField
    """
    ff = openmm_app.ForceField(*forcefield_files(implicit_solvent))
    for gen in template_generators or []:
        ff.registerTemplateGenerator(gen)
    return ff
