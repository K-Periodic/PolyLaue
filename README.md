# PolyLaue

![PolyLaue](https://github.com/user-attachments/assets/ad01aa2a-69ee-466e-acba-0079acf86e2b)

PolyLaue is a tool for visualizing and analyzing scans of Laue X-ray
diffraction patterns. It is developed in collaboration with the High-Pressure
Collaborative Access Team (HPCAT) at the Advanced Photon Source (APS) of
Argonne National Laboratory.

## Status of this fork

This experimental fork adds a **Grain Orientation and Cubic CSL** module. It
calculates the raw misorientation, the minimum cubic disorientation, rotation
axes in laboratory and crystal coordinates, and distances to exact cubic
coincidence-site-lattice rotations.

The CSL result classifies a three-dimensional lattice orientation
relationship. It does not determine the grain-boundary plane and, by itself,
does not prove that a boundary is coherent or that a twin is present. Letter
suffixes such as `a`, `b`, and `c` are deterministic internal variant labels;
use Sigma, ideal angle, and ideal axis together when comparing with a
published table. The calculation assumes that both grains represent the same
cubic phase and parent lattice; it does not verify phase identity.

The conda-forge package named `polylaue` is the official upstream release and
does **not** contain this fork's additions. Conda-forge is used below for the
dependencies, while this fork is installed from its Git repository or from a
wheel attached to a release. Do not install conda-forge's `polylaue` package
in the same environment.

## Install from a GitHub release

This is the recommended installation method. Install Miniforge or Anaconda,
then download the `.whl` file and `environment.yml` from the desired release's
**Assets** section. Do not use GitHub's automatically generated “Source code
(zip)” archive: it omits the Git metadata used to calculate the package
version.

Open a macOS/Linux terminal or Windows Miniforge/Anaconda Prompt in the
directory containing both downloaded files, then run:

```bash
conda env create -f environment.yml
conda activate polylaue-csl
python -m pip install --no-deps DOWNLOADED-WHEEL-FILENAME.whl
polylaue
```

Replace `DOWNLOADED-WHEEL-FILENAME.whl` with the exact name of the downloaded
wheel.

## Launch PolyLaue

For each new terminal session, activate the environment and launch the program:

```bash
conda activate polylaue-csl
polylaue
```

Load a reflections file containing at least two indexed grains of the same
cubic phase, then select **Indexing → Find Grain Orientation**.

## Install from a tagged source checkout

On this repository's GitHub page, select **Code** and copy the HTTPS clone URL.
Replace the example URL below with that URL and replace `RELEASE-TAG` with the
tag shown on the release page:

```bash
git clone https://github.com/K-Periodic/PolyLaue.git
cd PolyLaue
git checkout RELEASE-TAG
conda env create -f environment.yml
conda activate polylaue-csl
python -m pip install --no-deps --no-build-isolation .
polylaue
```

For development, install the checkout in editable mode instead:

```bash
python -m pip install --no-deps --no-build-isolation -e .
```

## Verify the installation

```bash
python -c "import importlib.metadata as m, polylaue; print(polylaue.__file__); print(m.version('polylaue'))"
python -m pip check
conda list polylaue
```

The version must match the selected release tag and must not be `0.0.0`.
`conda list` should identify PolyLaue as installed through pip rather than from
conda-forge.

On macOS or Linux, locate the command with:

```bash
command -v polylaue
```

On Windows:

```powershell
where.exe polylaue
```

## Avoid conflicts with official PolyLaue

Use separate environments, for example:

- `polylaue-csl` for this fork;
- `polylaue-official` for the conda-forge release.

Never run `conda install polylaue` or `conda update polylaue` inside
`polylaue-csl`. If both variants were installed into one environment, recreate
the environment; uninstalling one over the other can leave shared files in an
inconsistent state.

To install the official upstream release instead, follow the instructions in
the [official PolyLaue repository](https://github.com/PolyLaue/PolyLaue).
