# Precision Medicine Assignment 2 - GRN Extraction

This repository contains code and final gene regulatory networks (GRNs) generated from single-cell RNA sequencing (scRNA-seq) data.

## Dataset

The scRNA-seq dataset used in this project is available at [CZ CELLxGENE Discover](https://cellxgene.cziscience.com/collections/8c782494-01ed-491b-97b9-6f0d3b76c676). This dataset is part of the CZ CELLxGENE Discover platform, which provides curated and interoperable single-cell data for the research community.

## GRN Extraction

### PySCENIC

The script `extract_grn_pyscenic.py` is used to generate a heart failure-specific GRN using PySCENIC. The resulting file is located in the `heart_failure_grns` folder and is named `heart_failure_grn_pyscenic.h5ad`. This file follows the [GRnnData](https://github.com/cantinilab/GRnnData) format.

### scPRINT

The script `extract_grn_scprint.py` is used to generate a heart failure-specific GRN using scPRINT. The output file is located in the `heart_failure_grns` folder and is named `heart_failure_grn_scprint.h5ad`, following the [GRnnData](https://github.com/cantinilab/GRnnData) format.

