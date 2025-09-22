# castro-marim-scripts

Custom scripts used in the MSc thesis  
**“Linking metagenomics and salinity-driven microbial succession in artisanal marine saltworks”**  
(University of Algarve – CCMAR, 2025).

> **Scope**  
> Analyses are restricted to **Saltwork 1 and Saltwork 2 (water only)**.  
> Sediment analyses and the site WRED are **excluded** from this repository.

---

## Repository structure

- **analise_salinas.py**  
  Script for physicochemical analysis of **water samples**.  
  - Loads and cleans the Excel sheet (`Physicochemical Parameters table for water.xlsx`).  
  - Extracts Saltwork (1/2) and Stage (A–D) from sample IDs.  
  - Generates:
    - **Time-series plots** (Saltwork 1 and 2)  
    - **Detailed time-series by Stage & Saltwork** (water)  
    - **Mean-by-saltwork** bar plots  
    - **A–D gradient** line plots (water)  
    - **Correlation matrices** (by Saltwork and Stage)  
  - **Outputs:** PNGs under `Plots_Analysis_Final/`:
    - `Water_Plots/`, `Parameter_by_Local_Water_Plots/`,  
      `Detailed_Time_Series_Plots/`, `Correlation_Matrices/`

- **CFU_counts.py**  
  Script for analyzing **Colony-Forming Units (CFU/mL)** from culture-dependent assays.  
  - Input file: `cfu_table.xlsx` with columns:  
    `Date, Samples, Medium, Colonies, Dilution factor, CFU/mL (estimate), Status, LOQ low, LOQ high`  
  - Functions:
    - Normalize CFU values (supports scientific notation and commas).  
    - Generate **bar plots** per sampling (all media; and MA+MSA only).  
    - Generate **A–D line plots** per sampling & medium (Saltwork 1 vs 2).  
  - **Outputs:** PNGs under `figures_cfu/`

- **sequencias_sanger.py**  
  Script for processing raw **Sanger sequencing chromatograms** (`.ab1`).  
  - Trims sequences by quality (Phred cutoff).  
  - Saves **raw** and **clean** FASTA files.  
  - Generates CSV quality report.  
  - Exports good sequences into `sequencias_boas/`.  
  - Creates per-read quality plots under `graficos_qualidade/`.

---

## Requirements

Install dependencies with pip:

```bash
pip install -r requirements.txt

```
Minimal requirements.txt:
pandas>=1.5
numpy>=1.23
matplotlib>=3.6
seaborn>=0.12
openpyxl>=3.0
biopython>=1.80
pytz>=2022.0


```
```
How to run
1) Physicochemical analysis (water)
Place in the same folder:

analise_salinas.py

Physicochemical Parameters table for water.xlsx

Run:

bash
Copy code
python analise_salinas.py
Results: plots saved in Plots_Analysis_Final/.

2) CFU analysis
Place in the same folder:

CFU_counts.py

cfu_table.xlsx

Run:

bash
Copy code
python CFU_counts.py
Results: plots saved in figures_cfu/.

3) Sanger sequencing
Place in the same folder:

sequencias_sanger.py

all .ab1 chromatogram files

Run:

bash
Copy code
python sequencias_sanger.py
Results:

sequencias_brutas.fasta

sequencias_limpas.fasta

relatorio_qualidade.csv

sequencias_boas/, graficos_qualidade/


Citation
If you use these scripts, please cite:

Rocha, I. (2025). Linking metagenomics and salinity-driven microbial succession in artisanal marine saltworks (MSc Thesis, UAlg–CCMAR).
