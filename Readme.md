# BioP-VAE

## 1. Data Sources and Software Requirements

Raw sequencing data were obtained from the following public repositories:

- PRJEB49516 (Ashray_2024)  \
  https://www.ebi.ac.uk/ena/browser/view/PRJEB49516\
- PRJEB43119 (Lee_2022)  \
  https://www.ebi.ac.uk/ena/browser/view/PRJEB43119\
- PRJNA399742 (Matson_2018)  \
  https://www.ebi.ac.uk/ena/browser/view/PRJNA399742\
- PRJNA397906 (Frankel_2017)  \
  https://www.ebi.ac.uk/ena/browser/view/PRJNA397906\

Software requirements:

| Software | Version | Purpose |
|---|---|---|
| Sickle | v1.33 | Quality trimming of raw reads |
| Bowtie2 | v2.5.4 | Alignment to reference genomes |
| Samtools | v1.21 | SAM/BAM processing |
| MetaPhlAn | v3.0.13 | Metagenomic profiling |
| Prodigal | V2.6.3 | Predict protein-coding genes from contigs to generate protein sequences |
| CoverM | V0.7.0 | Predict protein-coding genes |
| Pytorch | v2.5.1 | Deep learning framework for model training and inference |

## 2. Bioinformatics Pipeline

### (1) Quality Control and Host Read Removal
#### Quality Control

Raw paired-end reads were quality-filtered using Sickle in paired-end mode.

Filtering criteria:

- Bases with quality score < Q20 were trimmed
- Reads shorter than 20 bp after trimming were discarded

- Only properly paired reads were retained for subsequent processing

#### Host Read Removal

To remove human host contamination, quality-filtered reads were aligned to the GRCh38 human reference genome using Bowtie2. Reads that did not align to the human genome were retained for downstream analyses.

### (2) Reference Genome Construction

Custom microbial reference genomes were constructed only using samples from the PRJEB49516 (Ashray_2024) cohort after host read removal.

#### Metagenomic Assembly

De novo assembly was performed individually for each sample using SPAdes in metagenomic mode.

After assembly: 
- Contigs from all samples were merged into a unified contig dataset
- Short contigs (<500 bp) were removed to improve reference quality

#### Genome Binning

MetaBAT2 clusters contigs based on sequence composition and coverage depth to generate bins.

These bins were then used to generate two non-redundant reference genome sets:

- Protein reference genome: the longest contig from each bin, used to construct protein-level biological priors.  
   - Genes were predicted from the contigs using Prodigal in metagenomic mode.  
   - Each predicted protein sequence was input into a pre-trained ESM model.  Each protein is encoded as a 320-dimensional vector, providing a fixed-length, continuous representation suitable for integration into downstream machine learning and deep learning models.

- RPKM reference genome: the top 200 contigs from each bin, used as the reference database for RPKM feature calculation.

### (3) RPKM Feature Calculation

RPKM (Reads Per Kilobase of transcript per Million mapped reads) was calculated for each sample by mapping host-filtered paired-end reads to the RPKM reference genome.  
```bash
coverm contig \
  -1 sample_filtered_R1.fastq \
  -2 sample_filtered_R2.fastq \
  -r rpkm_reference_genome.fasta \
  -t 16 \
  -m rpkm \
  -o sample_rpkm.tsv
  ```

## 3. Deep Learning Pipeline
### (1) Mutual Information Feature Selection

The script (`mutual_info.py`) calculates the mutual information between RPKM features and sample labels and saves the selected feature indices.

```python
label_file1 = "../dataset/PRJEB43119/PRJEB43119_label.csv"
RPKM_features1 = "../dataset/PRJEB43119/CA209-538-200contig_ReferenceGenome_RPKM_Features_PRJEB43119.csv"
check_mutual_info(label_file1, RPKM_features1, task)
```
After running the script, the results are saved as an .npz file

### (2) Model Training
The following example shows training on the **Ashray_2024** cohort.
```
python Train_VAE_prior.py \
--RPKM_features ../dataset/PRJEB49516/CA209-538-200contig_ReferenceGenome_RPKM_Features_CA209-538.csv \
--label_file ../dataset/PRJEB49516/CA209-538_label.csv \
--prior_file_path ../dataset/CA209-538_embedding.pth \
--task BOR \
--cuda_device 3 \
--epochs 500 \
--is_use_mutual_info 1 \
--model_save_path ../results/CA209-538/BOR
```

| Parameter            | Description                                               |
|----------------------|-----------------------------------------------------------|
| --RPKM_features       | Path to the RPKM feature matrix of the training cohort  |
| --label_file          | Path to the sample label file                             |
| --prior_file_path     | Path to the protein embedding prior file                 |
| --task                | Prediction task (BOR, not_progression_free)                         |
| --cuda_device         | GPU device ID                                  |
| --epochs              | Number of training epochs                                 |
| --is_use_mutual_info  | Whether to use features selected by mutual information (1 = yes, 0 = no) |
| --model_save_path     | Directory to save trained models and outputs             |


###(3) Cross-Cohort Testing
The following example evaluates a model trained on Ashray_2024 using the Frankel_2017 cohort.
```
python Test_VAE_prior.py \
--RPKM_features ../dataset/PRJNA397906/CA209-538-200contig_ReferenceGenome_RPKM_Features_PRJNA397906.csv \
--label_file ../dataset/PRJNA397906/PRJNA397906_label.csv \
--prior_file_path ../dataset/CA209-538_embedding.pth \
--cuda_device 8 \
--mutual_info_dir ../dataset/PRJEB49516 \
--task BOR \
--is_use_mutual_info 1 \
--mode_dirc ../results/CA209-538/BOR
```

| Parameter            | Description                                               |
|----------------------|-----------------------------------------------------------|
| --RPKM_features       | RPKM feature matrix of the testing cohort                |
| --label_file          | Label file for the testing cohort                         |
| --prior_file_path     | Protein embedding prior used during training             |
| --cuda_device         | GPU device ID                          |
| --mutual_info_dir     | Directory containing mutual information results from the training cohort |
| --task                | Prediction task (BOR, not_progression_free)                  |
| --is_use_mutual_info  | Whether to apply mutual information feature selection    |
| --mode_dirc           | Directory containing the trained model for testing       |