# M12 comprehensive lab — 0.4.0 first-run fixes

First pilot of `notebooks/nflbowl/M12_COMPREHENSIVE_EXPERIMENTAL_LAB_COLAB.ipynb`: Colab Tesla T4, clean bundle
fingerprint `4867be08f8191b3a` (77,125 sequences, 510 athletes), nflbowl 0.4.0, science hash `0281883accece2f0`,
`run_mode='pilot'`, `run_training=False`. B1–B4, G3 and H1 completed; C1, C4 and D3 failed.

These fixes were written by reading the code only. Neither the tests nor the notebook have been run. They are not
verified until the corrected code runs in Colab.

## Failures and fixes

| Stage | Error | Cause | Fix |
|---|---|---|---|
| C1 | `AttributeError: 'function' object has no attribute 'collapse_indicators'` | `nflbowl/evaluation/__init__.py` re-exports the function `domain_diagnostics`. That binding shadows the submodule of the same name, so `from ..evaluation import domain_diagnostics as dd` gets the function. | `lab/geometry.py`: `basic_geometry` and `domain_geometry` import `collapse_indicators` / `domain_diagnostics` from `..evaluation.domain_diagnostics`. |
| D3 | `AttributeError: 'function' object has no attribute 'domain_diagnostics'` | Same pattern in `Experiment.compare_inductive`. | `api.py`: imports `domain_diagnostics` and `positive_pair_coverage` from `.evaluation.domain_diagnostics`. |
| C4 | Polars `SchemaError: Float32 incompatible with Float64` | PCA coordinates are float64; UMAP returns float32; `pl.concat(how="vertical")` is strict. | New `geometry.projection_frame` (pure, testable). `projection_coordinates` delegates to it. `x`/`y` are cast to Float64 in both frames, `method`/`representation`/`fit_scope` are String, `fold` is Int64 (it was Int32 from `pl.lit(int)`). Metadata columns keep the bundle-meta dtypes. |

The public export shape of `nflbowl.evaluation` is unchanged, and nothing is patched at runtime.

Other module-style imports in the package (`evaluation.paired_inductive`, `models.registry`, `data.splits`,
`visualization.plots`) name real submodules that are not shadowed, so they are correct. A regression test now scans the
package for this pattern.

`tests/nflbowl/test_nflbowl_paired_inductive.py` had the same broken import (`dd.grouped_domain_auc` would also have
failed). It now binds the submodule with `importlib.import_module("nflbowl.evaluation.domain_diagnostics")`.

### Edge-case behaviour added (degenerate inputs only; science settings unchanged)

- `basic_geometry`: fewer than 3 finite held-out rows → scalars `dims`, `n_rows` (+ whatever `collapse_indicators` returns) and empty arrays, instead of an SVD / sampling error. `run_geometry` skips empty spectrum / norm frames.
- `projection_frame`: fewer than 3 sampled rows → an empty frame with the full schema. A 1-D embedding → `y = 0`. UMAP (still `n_neighbors=30`, `min_dist=0.2`) is fitted only when the sample has more than 30 rows. Real folds have thousands of rows, so pilot / standard outputs are unaffected.
- No cache key, science configuration or checkpoint namespace changes. C4 failed before writing, so nothing stale is cached.

## Tests added (written, not executed)

All of them are in `tests/nflbowl/test_nflbowl_lab.py`. They use the synthetic conftest bundle on CPU, with no warehouse and no GPU. UMAP is faked by a module that returns float32, the dtype that triggered the C4 failure.

- `test_no_module_style_import_of_a_reexported_function`: static scan of the package.
- `basic_geometry`: a held-out fold with a non-finite row; 0 and 2 rows.
- `domain_geometry`: a planted domain shift; a low-sample row set (NaN, no error).
- `projection_frame`: PCA only; PCA + UMAP (one schema, strict concat across calls, domain balance); 0 and 2 rows; ≤ 30 rows (UMAP skipped); a 1-D embedding.
- `projection_coordinates`: through a stub experiment (held-out rows of the requested fold).
- `Experiment.compare_inductive` without training:
  - F2 / F4 missing on every fold (status `missing`, F0 diagnostics plus training-split coverage rows); `on_missing='raise'`; a reference that is missing everywhere.
  - A synthetic F4 available on fold 0 only (unavailable fold 1, paired set restricted to fold 0, F4 domain diagnostics).

Run them with `pytest tests/nflbowl/test_nflbowl_lab.py tests/nflbowl/test_nflbowl_paired_inductive.py -q`.

## Notebook cells to paste (not applied to the .ipynb)

### Replace cell `setup-1`

```python
# Setup 1: install nflbowl from GitHub with ONE controlled install. Colab's CUDA build of PyTorch is never replaced:
# torch is not an nflbowl dependency, and the package itself is installed with --no-deps (nothing else is upgraded).
# Before the final multi-fold run, pin GIT_REF to the exact tested commit SHA, e.g. GIT_REF = '0123abcd...'.
import sys
from importlib.metadata import PackageNotFoundError, version

from packaging.requirements import Requirement

GIT_REF = 'main'
PKG_URL = f'git+https://github.com/mybenkhadda/NFL_bowl.git@{GIT_REF}'
if 'nflbowl' in sys.modules:
    print('WARNING: nflbowl is already imported in this runtime. After this cell: Runtime > Restart session, then continue with Setup 2.')

# nflbowl 0.4.0 core dependencies (pyproject.toml) + the lab extras (UMAP projections, HTML reports).
# Only requirements that are absent or outside their range are installed; satisfied Colab packages are left as they are.
REQUIREMENTS = ['numpy>=1.26,<3', 'polars>=1.10', 'pyarrow>=15', 'psutil>=5.9', 'pyyaml>=6.0', 'matplotlib>=3.8', 'duckdb>=1.1',
                'umap-learn>=0.5', 'markdown>=3.5']

def _unmet(reqs):
    out = []
    for r in map(Requirement, reqs):
        try:
            if not r.specifier.contains(version(r.name), prereleases=True):
                out.append(str(r))
        except PackageNotFoundError:
            out.append(str(r))
    return out

def _torch_version():
    try:
        return version('torch')
    except PackageNotFoundError:
        return None

TORCH_BEFORE = _torch_version()
MISSING = _unmet(REQUIREMENTS)
# --force-reinstall: the version string stays 0.4.0 across commits, so pip would otherwise keep an older 0.4.0 build.
%pip install -q --no-deps --force-reinstall {PKG_URL}
if MISSING:
    print('installing missing / out-of-range requirements:', MISSING)
    _quoted = ' '.join(f"'{r}'" for r in MISSING)
    %pip install -q {_quoted}
else:
    print('all nflbowl requirements and lab extras already satisfied; nothing else installed')
TORCH_AFTER = _torch_version()
assert TORCH_AFTER == TORCH_BEFORE, f'PyTorch changed during install ({TORCH_BEFORE} -> {TORCH_AFTER}): restore the Colab CUDA build before continuing'
print('nflbowl installed from', GIT_REF, '| torch', TORCH_AFTER, '(unchanged)')
```

`GIT_REF`, the repository URL, `CFG['bundle_dir']`, the experiment name (checkpoint namespace) and every science key stay
as they are in Setup 2.

### Insert a new cell after `setup-2` (before `setup-3`)

```python
# Setup 2b: preflight API compatibility (fails fast if the installed nflbowl predates the 0.4.0 first-run fixes) + scope warnings
import inspect

import nflbowl.api as _api
from nflbowl.evaluation.domain_diagnostics import collapse_indicators, positive_pair_coverage  # noqa: F401  (import check)

_problems = []
for _mod in (geometry, _api):
    if 'import domain_diagnostics as' in inspect.getsource(_mod):
        _problems.append(f'{_mod.__name__}: module-style import of the re-exported domain_diagnostics function (C1 / D3 AttributeError)')
if not hasattr(geometry, 'projection_frame'):
    _problems.append('nflbowl.lab.geometry has no projection_frame (C4 Float32/Float64 SchemaError fix)')
else:
    _rng = np.random.default_rng(0)
    _meta = pl.DataFrame({'idx': np.arange(40, dtype=np.int64), 'nfl_id': np.repeat(np.arange(10, dtype=np.int64), 4),
                          'domain': ['combine', 'nfl'] * 20, 'kind': ['forty', 'braking'] * 20, 'position_group': ['WR'] * 40})
    _E = _rng.normal(size=(40, 4)).astype(np.float32)
    _g, _ = geometry.basic_geometry(_E, _meta, np.arange(40), seed=0)
    _co = geometry.projection_frame(_E, _meta, np.arange(40), 'preflight', 0, seed=0, use_umap=False)
    if _co.schema['x'] != pl.Float64 or _co.schema['y'] != pl.Float64 or 'effective_rank' not in _g:
        _problems.append('geometry smoke check returned an unexpected schema')
if _problems:
    raise RuntimeError('installed nflbowl is not compatible with this notebook:\n  - ' + '\n  - '.join(_problems)
                       + '\nRe-run Setup 1 (pin GIT_REF to the fixed commit), restart the session, then Setup 2.')
print('preflight API check: ok')

if not cfg.run_training:
    print('WARNING: run_training=False. No encoder (F4, F2) is trained inductively in this session. An encoder enters D3 only '
          'if Phase 2 adopts a compatible inductive checkpoint. Otherwise D3 compares F0 and the random encoder only: '
          'that retrieval is DESCRIPTIVE and is not evidence about F4 generalization.')
print('CAVEATS: a B4 pass (no exact-duplicate group spans athletes or folds) rules out one leakage route; it is not proof '
      'of domain-invariant learning, because valid-frame count and duration alone separate the domains (AUC about 0.90 '
      'in the pilot). H1 biomechanical tables are exploratory; metrics with negative repeatability must not be read as '
      'athlete traits.')
```

## Scientific caveats for reports

- **B4.** 160 groups of exact duplicate sequences, none spanning athletes or folds. This closes one leakage route. Valid-frame count and duration alone discriminate the domains at AUC ≈ 0.90, so domain separability has trivial shortcuts, and passing B4 says nothing about domain invariance.
- **D3 pilot.** F0 and random on 79 held-out athletes (one fold, no inductive F4) is descriptive only.
- **H1.** The biomechanical tables are exploratory, and some metrics have negative repeatability.
