# Microbiome Multi-Omics Networks

This repository contains a collection of data-preparation and analysis scripts for building
and exploring multi-layer networks that combine patient clinical information, microbiome
profiles, and host gene expression measurements. The workflow focuses on creating
correlation-based networks that can be used to study patient similarity, microbes and
genes jointly, or either modality independently.

## Repository Layout

```
.
├── Scripts/            # Data wrangling, network construction, and analysis utilities
├── microbiomeviz/      # Web-based visualisation client (see `index.html`)
├── index.html          # Entry point for the visualisation client
└── README.md           # Project documentation (this file)
```

Most end-to-end processing happens through the Python scripts in the `Scripts/`
folder. Each script assumes that input data live in a `Data/` directory and that
results (networks, intermediate tables, and figures) are written to `Networks/`,
`Data/`, and `Figures/` respectively. Create these folders before running the code:

```bash
mkdir -p Data Networks Figures
```

## Getting Started

1. **Create a Python environment** (Python 3.9 or newer is recommended).
2. **Install dependencies** used across the scripts:
   ```bash
   pip install pandas numpy scipy mpmath scikit-learn umap-learn igraph leidenalg \
               xnetwork tqdm matplotlib colorcet
   ```
   Some analyses additionally require the optional `RModularity` Python package
   (available from <https://github.com/filipinascimento/RModularity>) when computing
   alternative modularity scores.
3. **Prepare the input data** in `Data/`:
   - `metadata_full_18Nov2014.xlsx`: clinical and summary metadata. It must contain
     patient identifiers in a `Patient` column followed by clinical scores, microbiome
     summary statistics, and gene expression columns in the same order expected by
     `Scripts/preprocessData.py`.
   - `species_otu_cts_clean.txt`: microbiome OTU relative-abundance counts with the
     columns `OTU_Name` and `prevalence` plus one column per patient sample.

   Replace these file names or adapt the column selections in `preprocessData.py` if
   your datasets differ.

## Core Workflow (`Scripts/preprocessData.py`)

`preprocessData.py` orchestrates the end-to-end processing:

1. Loads the clinical metadata, microbiome OTU table, and gene expression matrix.
2. Normalises the microbiome table (relative abundance) and applies either a centred
   log-ratio (`clr`) or logit transform. Toggle the transform via the
   `transformation` variable near the top of the file.
3. Harmonises sample identifiers across modalities and concatenates clinical,
   microbiome, and expression features into `Data/mergedData.csv`.
4. Constructs correlation matrices for patients (rows) and features (columns) using
   Pearson or Spearman correlation. Switch using the `useSpearman` flag when calling
   `generate_network`.
5. Applies optional disparity filtering or simple percentile-based thresholding to
   prune weak edges. Control behaviour with the `useDisparityFilter` and `percentile`
   parameters.
6. Detects communities with the Leiden algorithm, embeds nodes with PCA or UMAP, and
   stores the resulting networks in `Networks/*.xnet`. Both patient-level and
   feature-level networks are produced for:
   - Gene-only correlations
   - Microbiome-only correlations
   - Combined gene–microbiome correlations
   - Subset networks stratified by clinical CRS status (CRS0/CRS1)

### Running the script

```bash
python Scripts/preprocessData.py
```

The script iterates over several parameter combinations (Spearman/Pearson,
UMAP/PCA layouts, disparity vs linear thresholds, percentile cut-offs) and writes one
`.xnet` file per configuration. These files capture node metadata such as community
labels, clinical scores, and microbiome summaries for downstream analysis or
visualisation.

## Downstream Analysis Utilities

The remaining scripts leverage the `.xnet` networks produced by `preprocessData.py`.

### `Scripts/NetworkProperties.py`
Computes network-level statistics (degree/strength distributions, clustering,
assortativity, betweenness, modularity, participation coefficient, etc.) for a pair of
networks defined in the `networks` dictionary. Extend or repurpose this script to
compare other network variants by editing the mapping.

### `Scripts/NetworkFeatures.py`
Derives node-level attributes such as module degree z-score, betweenness inside
communities, rich club coefficients, and more. These features can feed into further
statistical analyses or integration with the `microbiomeviz` front end.

### `Scripts/NetworkTables.py`
Generates ranked comparison tables/figures for selected vertex metrics (degree,
centrality, coreness, etc.) across two condition-specific networks (default CRS0 vs.
CRS1). Outputs PDF summaries to `Figures/` and compiles helper tables inside the same
folder.

### `Scripts/calculateModularities.py`
Scans all feature-level networks in `Networks/`, recomputes modularity scores using
`igraph` and optional alternative definitions from `RModularity`, and saves the summary
statistics to `Data/modularities_genemicrobiome.csv`.

## Customisation Tips

- **Change thresholds or embeddings** by editing the parameter grid at the bottom of
  `preprocessData.py`.
- **Focus on patients or features** by filtering the networks that the script generates
  (`isPatientNetwork` flag).
- **Add new clinical stratifications** by extending the `filterCRSData` helper and
  duplicating the `generate_network` calls with new cohort definitions.
- **Visualise results** by loading the `.xnet` files into the `microbiomeviz` web app
  (open `index.html` in a browser) or any compatible network-visualisation tool.

## Citation


## License (MIT)
Copyright 2025 Filipi N. Silva

Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the “Software”), to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED “AS IS”, WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
