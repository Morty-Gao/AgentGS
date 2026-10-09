# Data preparation

`data/raw/` stores raw downloads, source records, and temporary processing files. Prepared results go in `data/`: one file for a single-file dataset, and a directory for a multi-file dataset. Processing starts from the raw cache. If an old release is missing, the available current release is used. The actual source and hash are recorded. An old result snapshot is not used automatically.

## Run

After GEM data is fetched with Git, the standard library can unpack it. The project dependencies are not required:

```bash
python3 scripts/prepare/prepare_gem.py
```

By default this reads `data/gem.jsonl.gz`, expands the Git-ignored `data/gem.jsonl`, and writes a prepare manifest. `--source-root` selects the directory that contains the compressed file. `--output-root` selects the project root that receives `data/`. The output file is replaced only after gzip and JSONL both validate. A failed unpack keeps the existing file.

From the repository root, in the environment with the scientific extra:

```bash
.venv/bin/python scripts/prepare/prepare_all.py
.venv/bin/python scripts/prepare/prepare_all.py --offline
.venv/bin/python scripts/prepare/prepare_all.py --offline --output-root outputs/prepare-check
.venv/bin/python scripts/prepare/prepare_ocean.py --offline
```

The aggregate entry runs `prepare_gem.py`, `prepare_deep_earth.py`, `prepare_geotoolkg.py`, `prepare_geoquestions.py`, `prepare_mineral.py`, `prepare_climate.py`, `prepare_energy.py`, `prepare_ocean.py`, `prepare_biodeeptime.py`, `prepare_ccs.py`, `prepare_remote_sensing.py`, and `prepare_hazard.py`, in that order.

- `--output-root`: project root that receives prepared `data/`. Default: this repository.
- `--cache-root`: raw cache. Default: `data/raw/` in this repository.
- `--source-root`: extra retained prepared inputs. Default: `data/` in this repository. Use this for original attachments, hand-built assets, deep-earth inputs, and KG inputs.
- `--offline`: use only the existing raw cache.
- `--archive-threshold-mb`: also store a ZIP when a multi-file dataset exceeds this size. Default: 64 MiB.
- `--archive`: a domain entry checks and unpacks the given ZIP. Products still land in `data/<dataset>/`.

ZIP packages are `data/<dataset>.zip`. The matching prepare script expands them to `data/<dataset>/`. The runtime always reads the prepared directory. The expanded directory is a processing result and is not placed in the raw area. There is no `raw/prepared` mode.

For deep earth, KG, BioDeepTime, and CCS, if the source directory contains only a ZIP, prepare checks and unpacks it. Other datasets are rebuilt from raw sources by default. `--archive` can explicitly unpack an already prepared package. ZIP members use a stable file order and timestamp and include a per-file SHA-256 manifest. Unpack checks paths and content. The multi-file directory remains the local result that readers use. Generated packages and expanded directories are not committed. Inputs that must be retained are the exact exceptions in `.gitignore`.

## Prepared results

| Dataset | Result | Default preparation |
| --- | --- | --- |
| GeoToolEval | `data/geotooleval.jsonl` | 603 questions. Eleven workbook fields use English names, plus `question_sha256` and `answer_format`. Read directly. No prepare step. |
| GEM observations | `data/gem.jsonl.gz`, expanded to `data/gem.jsonl` | Every workbook sheet, row order, cell value, and cached formula. Git stores the compressed JSONL. |
| Minerals | `data/mineral.csv` | Selected from MRDS by the original 8,000 record IDs. |
| Hazards | `data/hazard.jsonl` | Full retained hazard attachment: 10,764 rows and 47 columns. |
| GeoQuestions | `data/geoquestions/` | 40 questions and query-result JSONL generated from the upstream files. |
| Deep earth | `data/deep_earth/` | Existing model samples, the USGS catalog, and the source note. |
| KG | `data/geotoolkg/` | Original index JSON and the import note. |
| Climate | `data/climate/` | NOAA observations, the NASA annual series, and the IPCC SSP table: 4 CSVs. |
| Energy | `data/energy/` | 11 CSVs from GEM, MRDS, OPSD, OWID, and the World Bank. |
| Ocean | `data/ocean/` | 9 CSVs from SOCAT, GLODAP, and SNAPO. |
| BioDeepTime | `data/biodeeptime/` | Observation subset, series summary, and hand-drawn zones. |
| CCS | `data/ccs/` | Retained spatial assets. Public tables are prepared from raw sources. License and schema are in `metadata.json`. |
| Remote sensing | `data/earthbench.zip` and the expanded `data/earthbench/` | Fixed 49 directories, 1,415 original files, and the asset index. Images are kept losslessly. |

Tables are UTF-8 CSV with LF newlines. Spatial data is GeoJSON. Complex records are JSONL. Indexes and source notes are JSON. Reserialization keeps rows, columns, values, and zone boundaries. It does not force the old attachment's BOM or newline bytes.

Public sources download to `data/raw/<source>/`. The model cache is `data/raw/models/fastembed/`. Energy preparation reads `data/gem.jsonl` or `data/gem.jsonl.gz` and does not depend on XLSX. Import a new attachment once with `prepare_gem.py --workbook data/raw/gem/<file>.xlsx`. Later, `prepare_gem.py` checks and expands the Git-retained compressed JSONL. If the old World Bank cache is missing, the current API ZIP is read and the indicator CSV is found by content. The NASA annual series is built from the J-D column of the current monthly table, and only complete years are kept.

## Git and checks

Git allows GeoToolEval, the compressed GEM JSONL, the original hazard records, the BioDeepTime subset and zones, the original CCS spatial assets, the deep-earth inputs, the KG JSON, and the source note in `data/metadata/geotooleval.json`. ZIP forms of those multi-file inputs may also be kept. Generated minerals, climate, energy, ocean, GeoQuestions, the remote-sensing package, and prepare manifests are ignored. `data/raw/` is always ignored.

The other GeoToolEval worksheets and the source manifest are in `data/metadata/geotooleval.json`. The main dataset is still one JSONL. Resource aliases are `data/metadata/resources.json`. They map original resource paths onto the current layout.

Answer formats are stored on each GeoToolEval record in `answer_format`. The runtime reads only the question, question ID, question hash, and format settings. The English system prompt is in `src/agentgs/datasets/geotooleval.py`. `data/answer_formats/` is not generated or maintained.

Each domain manifest is `data/metadata/prepare_<domain>.json`. The aggregate manifest is `prepare_all.json`. A manifest records the source hash actually used, the processing-code and parameter hash, and the result hash. Success returns 0. Missing input returns 2 and is marked partial. A processing error returns non-zero. The aggregate entry continues with the other domains and keeps each failure status.

Fixed reference data moved from `src` to `data`. `data/reference/energy.json` stores capital coordinates, regions, country and energy-type and critical-mineral classes, and capacity-series selections. `ocean.json` stores ocean regions. `geography.json` stores the Earth radius. `data/geoquestions/profile.json` stores the fixed query list and its source fingerprint. Git must keep these inputs. Prepare copies them and records their hashes. Formulas, quality-control algorithms, protocol constants, and path maps stay in code.
