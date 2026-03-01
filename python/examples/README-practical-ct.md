# Practical 2D CT Pipeline

`practical_ct_pipeline.py` runs a manuscript-oriented 2D CT workflow:

1. load a real slice (Walnut `.mat` or PCIR DICOM zip),
2. normalize and resize to a configurable square phantom input (`--image-size`, default `200`),
3. run `line_integral --sinogram` in MFA and trapezoid modes,
4. reconstruct with `scikit-image` (`iradon`, optional `iradon_sart`),
5. write figures and metrics JSON for manuscript use.

## Package Dependencies
- numpy
- scipy
- scikit-image
- matplotlib
- pydicom

## Python dependencies

```bash
python3 -m pip install -r mfa/python/examples/requirements-practical-ct.txt
```

## Build the line integral executable

```bash
cmake -S mfa -B mfa/build -DCMAKE_BUILD_TYPE=Release
cmake --build mfa/build --target line_integral -j
```

## Run

Run only Walnut:

```bash
python3 mfa/python/examples/practical_ct_pipeline.py walnut \
  --download \
  --line-integral-binary mfa/build/examples/eval/line_integral
```

Run Walnut then PCIR in manuscript order:

```bash
python3 mfa/python/examples/practical_ct_pipeline.py manuscript \
  --download \
  --line-integral-binary mfa/build/examples/eval/line_integral
```

Use `--skip-forward` to only prepare input slices without running MFA/trapezoid forward projection.

## Output layout

Default outputs are written under `plots/practical_ct/`:

- `plots/practical_ct/<dataset>/prepared/` prepared slice and metadata,
- `plots/practical_ct/<dataset>/mfa_forward/` MFA forward-projection logs and sinogram files,
- `plots/practical_ct/<dataset>/trapezoid_forward/` trapezoid forward-projection logs and sinogram files,
- `plots/practical_ct/<dataset>/<dataset>_comparison.png` panel figure,
- `plots/practical_ct/<dataset>/summary.json` dataset metrics summary,
- `plots/practical_ct/manuscript_summary.json` combined Walnut+PCIR summary.
