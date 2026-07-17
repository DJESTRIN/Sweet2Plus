# Sweet2Plus Repository Reorganization Proposal

**Status:** Proposal only — no code has been moved yet. This document is meant to
be reviewed by the lab, then executed as a follow-up change (or series of PRs).

**Author:** Copilot CLI review, 2026-07-17

## 1. How the repo looks today

```
Sweet2Plus/                      <- git repo root
├── Sweet2Plus/                  <- the actual Python package
│   ├── core/                    behavior parsing, suite2p wrapper, quick_run/quickgui, correlation & stim analysis
│   ├── cloud/                   two shell scripts for running CAA on a cluster
│   ├── decoders/                GNN / decoder models (+ a copy of NetworkArchitectures.py)
│   ├── denoise/                 DeepCAD wrapper + destreak
│   ├── graphics/                plotting helpers
│   ├── pose_estimation/         DLC_parse.py and DLC_parse_KWJ.py (near-duplicates)
│   ├── signalclassifier/        MLP classifier code **+ committed X.npy (23 MB) and best_model_weights.pth (19 MB)**
│   ├── statistics/               R + Python stats, includes glms/ and markdown/ subpackages
│   ├── utils/                   logger, parallel helper, folder renaming
│   └── SynapticWeightModeling/  a second, separate top-level package (not under Sweet2Plus/) with its own copy of NetworkArchitectures.py
├── portreader_gui/              standalone GUI app + pseudo_comdata/ (12 CSVs, several MB each, sample serial data)
├── arduino/                     .ino firmware for sens/sync/tmt_assay rigs
├── tests/                       one smoke test (import-only)
├── my_data/, my_figs/, figures/ scratch output dirs — gitignored now, but old files were committed before the ignore rules existed
├── images/                      README images (fine, small, meant to be committed)
├── best_model_weights.pth (root), training_metrics.xlsx (root)  <- stray copies of generated artifacts
├── requirements.txt, suite2p310_requirements.txt (binary conda export, not real text), Dockerfile, setup.py
└── README.md, ToDoList.txt
```

## 2. Problems identified

### 2.1 Repo bloat from committed binary/data artifacts
Several large generated files are tracked in git history and will stay in every
clone forever unless history is rewritten:
- `Sweet2Plus/signalclassifier/X.npy` — 23.4 MB
- `Sweet2Plus/signalclassifier/best_model_weights.pth` — 19.0 MB
- `best_model_weights.pth` (root, duplicate of the one above) — 19.0 MB
- `my_figs/solo_model.pth` — 6.8 MB (inside a directory `.gitignore` now excludes, but it's already committed so the rule doesn't remove it)
- `portreader_gui/pseudo_comdata/*.csv` — five files 1–4 MB each (sample serial logs, not source code)
- `training_metrics.xlsx` (root)

None of these belong in version control. They bloat every clone/fetch and have
nothing to do with reviewing source code changes.

### 2.2 Naming-convention inconsistency (corrected from initial review)
**Correction:** an earlier draft of this document incorrectly stated that
`SynapticWeightModeling/` lived outside the `Sweet2Plus` package namespace.
It does not — it was already a proper subpackage at
`Sweet2Plus/Sweet2Plus/SynapticWeightModeling/` with its own `__init__.py`,
importable as `Sweet2Plus.SynapticWeightModeling`, and already covered by
`tests/test_imports.py`'s `pkgutil.walk_packages` discovery. The only real
issue was a **naming-convention mismatch**: it used PascalCase while every
sibling subpackage (`decoders`, `signalclassifier`, `pose_estimation`, etc.)
uses snake_case. This has been fixed by renaming it to `weight_modeling/`
(see §4, PR2, completed).
Its `NetworkArchitectures.py` was also checked against
`Sweet2Plus/decoders/NetworkArchitectures.py` — they are **not duplicates**;
they define entirely different model classes for different purposes (GNN
decoders vs. per-neuron LSTM/transformer weight-modeling architectures). The
shared filename is a coincidence and is harmless since each lives in its own
subpackage namespace, so no further de-duplication is needed.

### 2.3 Inconsistent/unclear submodule boundaries
- `core/` mixes several unrelated concerns: suite2p wrapping (`customs2p.py`,
  `core.py`), behavior/serial-log parsing (`behavior.py`), object
  serialization (`SaveLoadObjs.py`), and two different analysis scripts
  (`CorrelativeActivityAnalysis.py`, `StimulationAnalysis.py`) that look more
  like `statistics/`-level analyses than "core" infrastructure.
- `pose_estimation/` has `DLC_parse.py` and `DLC_parse_KWJ.py` — an unclear
  naming convention (are personal-initial suffixes meant to be temporary
  forks, alternate configs, or dead code?). New contributors can't tell which
  one is canonical.
- `cloud/` only contains two `.sh` scripts for launching `CorrelativeActivityAnalysis.py`
  on a cluster — arguably these belong next to the analysis they invoke
  (`core/` or a `scripts/` folder), not as a package importable via `import Sweet2Plus.cloud`.
- Top-level non-Python assets (`portreader_gui/`, `arduino/`) are siblings of
  the Python package, which is fine, but they mix a Python GUI app, hardware
  firmware, and a Python library in one repo without a README pointing out
  that they're three different kinds of projects.

### 2.4 Naming / packaging inconsistencies
- `requirements.txt` is a normal pip file; `suite2p310_requirements.txt` is
  actually a binary/UTF-16 conda environment export (not usable via
  `pip install -r`). This is confusing without a comment explaining which to
  use when.
- `setup.py` still has a placeholder description (`'A short description of
  your project'`) instead of a real one.
- Mixed casing conventions across folders: `SynapticWeightModeling`
  (PascalCase) vs `signalclassifier`, `pose_estimation`, `decoders`
  (lower/snake_case). Picking one convention would make the package easier to
  navigate.

### 2.5 Hardcoded, non-portable paths
Example: `Sweet2Plus/core/quick_run.py` parses paths with
`path.split('\\')`, which silently breaks on non-Windows systems (the lab
should decide whether this matters given the intended runtime environment —
worth flagging even though this proposal is about structure, not logic).

### 2.6 Test coverage
Only one test file (`tests/test_imports.py`) exists, and it explicitly notes
it only checks import-time correctness, not runtime behavior, because there is
no synthetic test data in the repo. A reorganized repo is a good opportunity
to add a `tests/data/` (small, synthetic, git-trackable) fixture set per
subpackage.

## 3. Proposed new structure

```
Sweet2Plus/
├── src/Sweet2Plus/                  # (optional) move package under src/ layout to avoid import-path ambiguity
│   ├── __init__.py
│   ├── core/                        # ONLY suite2p object construction + serialization
│   │   ├── customs2p.py
│   │   ├── core.py
│   │   ├── SaveLoadObjs.py
│   │   └── behavior.py
│   ├── analysis/                    # renamed/moved from core/: study-level analyses
│   │   ├── correlative_activity.py  # was CorrelativeActivityAnalysis.py
│   │   └── stimulation_analysis.py  # was StimulationAnalysis.py
│   ├── cli/                         # was quick_run.py / quickgui.py — user entry points
│   ├── cluster_scripts/             # was cloud/ — .sh launch scripts, clearly named as ops scripts not a package
│   ├── decoders/                    # GNN/decoder models, single canonical NetworkArchitectures.py
│   ├── weight_modeling/             # was SynapticWeightModeling/ — renamed only (it was already
│   │                                  a proper Sweet2Plus subpackage); NetworkArchitectures.py here
│   │                                  is distinct in content/purpose from decoders/NetworkArchitectures.py
│   │                                  and stays separate (DONE — see status table below)
│   ├── denoise/
│   ├── graphics/
│   ├── pose_estimation/             # DLC_parse.py stays; DLC_parse_KWJ.py either merged in
│   │                                  via config/parameters, renamed to reflect its actual
│   │                                  purpose, or removed if superseded
│   ├── signalclassifier/            # code only; X.npy / best_model_weights.pth moved out (see below)
│   ├── statistics/
│   │   ├── glms/
│   │   └── markdown/
│   └── utils/
├── apps/
│   └── portreader_gui/              # unchanged internally, just grouped as "a standalone app" alongside future GUIs
├── firmware/
│   └── arduino/                     # was arduino/, renamed to clarify it's device firmware, not analysis code
├── docs/
│   └── images/                      # was images/, README figures
├── tests/
│   ├── test_imports.py
│   └── data/                        # small synthetic fixtures for future real unit tests
├── assets/                          # NEW: canonical home for model weights/checkpoints if they must live in-repo,
│   │                                  otherwise these should move to Git LFS, cloud storage, or a release asset
│   └── README.md                    # explains where to download large artifacts instead of committing them
├── requirements.txt
├── environment-suite2p310.yml       # renamed from suite2p310_requirements.txt to reflect that it's a conda env file, not pip
├── Dockerfile
├── setup.py                         # filled-in description/classifiers
├── README.md
└── ToDoList.txt                     # or migrate to GitHub Issues/Projects for trackable, assignable tasks
```

### Key structural changes summarized
1. **[DONE] Rename `SynapticWeightModeling/` to `weight_modeling/`** for
   snake_case consistency with sibling subpackages. (It was already inside
   the `Sweet2Plus` package and already tested/packaged like everything
   else — see §2.2 correction above.) Its `NetworkArchitectures.py` is
   distinct in content from `decoders/NetworkArchitectures.py`, so no
   de-duplication was needed; both stay as-is.
2. **Split `core/` into `core/` (suite2p + serialization plumbing) and
   `analysis/` (study-specific analyses)** so "core" actually means core
   infrastructure, not a catch-all.
3. **Move `cloud/` shell scripts to `cluster_scripts/`** (or under
   `scripts/`), not imported as a Python package.
4. **Remove committed binary artifacts from git** (`X.npy`,
   `best_model_weights.pth` x2, `solo_model.pth`, `training_metrics.xlsx`,
   large `pseudo_comdata/*.csv`). Options, in order of preference:
   - Store on a lab shared drive / cloud bucket, and document the download
     step in `assets/README.md` or `README.md`.
   - If they must stay in git, use **Git LFS** so clones stay small.
   - Regenerate at test/run time to keep only truly minimal synthetic
     fixtures under `tests/data/`.
   This should also include a **git history cleanup** (e.g. `git filter-repo`)
   once the team agrees, since `.gitignore` alone does not remove
   already-tracked files or shrink `.git` history.
5. **Rename `arduino/` → `firmware/arduino/`** and `images/` → `docs/images/`
   to make the repo's top level self-explanatory at a glance (Python package,
   apps, firmware, docs, tests).
6. **Resolve `DLC_parse.py` vs `DLC_parse_KWJ.py`** — pick one canonical
   implementation with configurable parameters, or clearly document why both
   exist (e.g. rename to reflect actual behavioral difference, not initials).
7. **Fix `suite2p310_requirements.txt`** — rename to indicate it's a Conda
   export (`environment-suite2p310.yml`), not a pip requirements file.
8. **Fill in `setup.py` metadata** (real description, classifiers, license
   field consistent with README's MIT mention).
9. Consider moving `ToDoList.txt` to GitHub Issues so tasks are assignable,
   commentable, and closeable, keeping the repo itself free of a manually
   maintained task list.

## 4. Suggested migration order (safe, incremental PRs)

**Progress so far (this session):**
- [DONE] PR 1 (partial): removed root-level `best_model_weights.pth`
  duplicate and untracked `my_figs/` scratch output from git.
- [DONE] PR 2 (revised scope): renamed `SynapticWeightModeling/` →
  `weight_modeling/` for naming consistency (see §2.2 correction — it was
  already a proper subpackage, so no namespace merge was actually needed).
- [DONE] PR 3: split `core/` into `core/` (suite2p wrapper + serialization
  plumbing) and `analysis/` (correlative-activity and stimulation
  analyses); renamed `cloud/` → `cluster_scripts/`. Updated hardcoded HPC
  paths in `CAA.sh`, `parallel_CAA.sh`, and `utils/parallel_helper.py`.
- [DONE] PR 4: renamed `arduino/` → `firmware/arduino/` and `images/` →
  `docs/images/`; updated README image URLs and repo-layout diagram.

1. **PR 1 — cleanup only, no moves:** remove root-level stray duplicate
   artifacts (`best_model_weights.pth`) since it is an exact duplicate of a
   file already inside the package. Add a documented `.gitattributes`/LFS
   plan for the remaining large tracked binaries.
2. **PR 2 — rename `SynapticWeightModeling` → `weight_modeling`** for
   snake_case consistency with sibling subpackages (no namespace change
   needed — it was already part of `Sweet2Plus`).
3. **PR 3 — split `core/` into `core/` + `analysis/`,** move `cloud/` to
   `cluster_scripts/`. Update all internal imports and any shell scripts
   referencing old paths.
4. **PR 4 — rename top-level non-package folders** (`arduino/` →
   `firmware/arduino/`, `images/` → `docs/images/`, update README image
   links).
5. **PR 5 — extract large binary artifacts from git tracking** (requires
   coordinating with anyone who has scripts pointing at the current paths;
   decide on LFS vs external storage; optionally rewrite history in a
   separate, clearly-communicated step since it changes commit hashes).
6. **PR 6 — polish packaging metadata** (`setup.py`, rename conda file,
   clarify `requirements.txt` vs conda env file usage in README).

Each PR should re-run `pytest` (the import smoke test) and a manual `pip
install -e .` to confirm packaging still works after each move.

## 5. Open questions for the lab (need a decision before executing)

- Are `DLC_parse.py` and `DLC_parse_KWJ.py` both still in active use? If one
  is obsolete, which one?
- Should large model weights/data live in Git LFS, a shared lab drive, or a
  cloud bucket? This affects how `assets/` is set up and whether existing git
  history should be rewritten to shrink the repo.
- Should `portreader_gui` and `arduino` remain in this repo at all, or would
  they be better as separate repos (they're logically independent of the
  two-photon analysis package)?
