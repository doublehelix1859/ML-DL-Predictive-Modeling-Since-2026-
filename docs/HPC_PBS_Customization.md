# HPC/PBS Customization Guide

Every PBS file contains placeholders that users must edit after installation or before submission.

Replace:

```text
<YOUR_EMAIL_ADDRESS>
<YOUR_CONDA_ENV_NAME>
<ABSOLUTE_PATH_TO_PROJECT_WORKING_DIRECTORY>
```

Example:

```bash
#PBS -M username@university.edu
conda activate rice_env
ROOT="/home/username/project/phenotype_prediction"
```

For large genotype or transcriptome matrices, increase memory:

```bash
#PBS -lselect=1:ncpus=16:mpiprocs=16:mem=64gb
```

For memory-safe fast mode, set in `config.env`:

```bash
export FAST_MODE=1
export SELECT_K=12000
```
