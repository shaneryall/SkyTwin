# Third-party licenses

Beyond the Rust standard library, Skytwin uses these crates at run time, directly or, where
marked, through `sgp4`:

| Crate | License | Use |
| --- | --- | --- |
| `sgp4` | MIT | SGP4 propagation of TLEs |
| `chrono` (through `sgp4`) | MIT or Apache-2.0 | TLE epochs |
| `num-traits` (through `sgp4`) | MIT or Apache-2.0 | `libm` maths for SGP4 |
| `fayde-catalog`, `fayde-core` | GPL-3.0-or-later | Star catalogues |
| `libm` | MIT | Portable maths |
| `serde`, `serde_json` | MIT or Apache-2.0 | Scene and truth files |
| `sha2` (CLI only) | MIT or Apache-2.0 | Hashes in the provenance file |

[`crates/skytwin-py/THIRD-PARTY-LICENSES.txt`](crates/skytwin-py/THIRD-PARTY-LICENSES.txt) lists
every crate compiled into the command and the Python package, with its licence text; the wheel
carries it. `scripts/third_party_licenses.py` writes it from `Cargo.lock`, and continuous
integration checks that it is current. It also checks every crate in `Cargo.lock` against the
licenses that `deny.toml` allows, all of which are compatible with GPL-3.0-or-later.

The Python bindings (`crates/skytwin-py`) also use `pyo3` (MIT or Apache-2.0). The Python
packages have these run-time dependencies:

| Package | License | Used by |
| --- | --- | --- |
| `numpy` | BSD-3-Clause | `skytwin`, `skytwin-calibrate` |
| `scipy` | BSD-3-Clause | `skytwin-calibrate` |
| `astropy` | BSD-3-Clause | `skytwin-calibrate` |

## Data

`profiles/cassiope-fai-nir.json` holds values measured from the DUST
dataset: V. Suthakar, P. Kunalakantha, R. S. K. Lee and G. Sohn, "An
On-Orbit Star Tracker Benchmark for Resident Space Object Detection and
Attitude Estimation", *Scientific Data* (2026),
doi:[10.1038/s41597-026-07736-9](https://doi.org/10.1038/s41597-026-07736-9);
data at doi:[10.5281/zenodo.20255672](https://doi.org/10.5281/zenodo.20255672),
licensed [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/). The
images were taken by the Fast Auroral Imager on CASSIOPE. No DUST images
are included here.

`profiles/astrianet-pl16803.json` holds values measured from J. Feuge-Miller,
D. Kucharski, S. Iyer and M. Jah, "ASTRIANet Data for PyCIS", Texas Data
Repository (2021),
doi:[10.18738/T8/GV0ASD](https://doi.org/10.18738/T8/GV0ASD), released
under [CC0 1.0](https://creativecommons.org/publicdomain/zero/1.0/). No
ASTRIANet images are included here.

`profiles/ckoirama-cdk24.json` holds values measured from the Ckoirama
Observatory frames of the BlueWalker 3 observing campaign organised by the
IAU Centre for the Protection of the Dark and Quiet Sky from Satellite
Constellation Interference: S. Nandakumar et al., "The high optical
brightness of the BlueWalker 3 satellite", *Nature* 623, 938–941 (2023),
doi:[10.1038/s41586-023-06672-7](https://doi.org/10.1038/s41586-023-06672-7);
data at doi:[10.5281/zenodo.8102655](https://doi.org/10.5281/zenodo.8102655),
licensed [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/). No
campaign images are included here.

`profiles/wise-w1.json` holds values measured from WISE All-Sky
single-exposure (Level 1b) W1 frames. This makes use of data products from
the Wide-field Infrared Survey Explorer, which is a joint project of the
University of California, Los Angeles, and the Jet Propulsion
Laboratory/California Institute of Technology, funded by the National
Aeronautics and Space Administration (E. L. Wright et al., *AJ* 140,
1868–1881 (2010),
doi:[10.1088/0004-6256/140/6/1868](https://doi.org/10.1088/0004-6256/140/6/1868)),
and of the NASA/IPAC Infrared Science Archive, operated by the Jet
Propulsion Laboratory, California Institute of Technology, under contract
with NASA. No WISE images are included here.

The learned profiles' photometry is calibrated against Gaia DR3 (Gaia
Collaboration, A. Vallenari et al., *A&A* 674, A1 (2023),
doi:[10.1051/0004-6361/202243940](https://doi.org/10.1051/0004-6361/202243940)),
from the European Space Agency mission Gaia, processed by the Gaia Data
Processing and Analysis Consortium. No catalogue rows are included here.
