# Scripts3 - Multi-Dataset Support

This folder contains updated versions of all experiment scripts with support for multiple FAERS datasets: **FAERS_ALL**, **FAERS_TB**, and **FAERS_TB_DRUGS**.

## What's New in Scripts3

### Key Changes

1. **Multi-Dataset Support**: All scripts now support three dataset variants:
   - `FAERS_ALL`: Full FAERS dataset (2018-2025)
   - `FAERS_TB`: Tuberculosis-related cases
   - `FAERS_TB_DRUGS`: TB cases with TB-specific drugs

2. **Dataset Selection**: Choose dataset via environment variable:
   ```bash
   export DATASET_NAME=FAERS_TB
   # or
   DATASET_NAME=FAERS_TB_DRUGS python run_pipeline.py
   ```

3. **Automatic Path Management**: All paths automatically adjust based on selected dataset:
   - Data: `data/FAERS_ALL/`, `data/FAERS_TB/`, `data/FAERS_TB_DRUGS/`
   - Preprocessed: `data/preprocessed_faers_all/`, etc.
   - Output: `output_faers_all/`, `output_faers_tb/`, `output_faers_tb_drugs/`

4. **Independent Outputs**: Each dataset variant stores results in separate folders

## Prerequisites

### Step 1: Build Datasets

Before running any experiments, build the dataset variants using the dataset builder:

```bash
# Build all three datasets
python build_datasets.py

# Or build specific dataset
python build_datasets.py --dataset FAERS_TB
python build_datasets.py --dataset FAERS_TB_DRUGS
```

This will create:
```
data/
├── FAERS_ALL/
│   ├── DEMOGRAPHICS.csv
│   ├── DRUGS_STANDARDIZED_DRUGBANK.csv
│   ├── ADVERSE_REACTIONS.csv
│   └── ...
├── FAERS_TB/
│   ├── DEMOGRAPHICS.csv
│   ├── DRUGS_STANDARDIZED_DRUGBANK.csv
│   └── ...
└── FAERS_TB_DRUGS/
    ├── DEMOGRAPHICS.csv
    ├── DRUGS_STANDARDIZED_DRUGBANK.csv
    └── ...
```

## Usage

### Running Full Pipeline

For FAERS_ALL (default):
```bash
cd scripts3
python run_pipeline.py
```

For FAERS_TB:
```bash
cd scripts3
DATASET_NAME=FAERS_TB python run_pipeline.py
```

For FAERS_TB_DRUGS:
```bash
cd scripts3
DATASET_NAME=FAERS_TB_DRUGS python run_pipeline.py
```

### Running Individual Steps

Step 1 - Preprocessing:
```bash
DATASET_NAME=FAERS_TB python step1_preprocessing.py
```

Step 2 - Graph Construction:
```bash
DATASET_NAME=FAERS_TB python step2_aer_graph_construction.py
```

Step 4 - Training:
```bash
DATASET_NAME=FAERS_TB python step4_training.py --variant xxx
```

### Running Specific Pipeline Steps

```bash
# Only preprocessing and graph construction
DATASET_NAME=FAERS_TB python run_pipeline.py --steps 1 2

# Only training
DATASET_NAME=FAERS_TB_DRUGS python run_pipeline.py --steps 4

# All ablation studies
DATASET_NAME=FAERS_TB python run_pipeline.py --steps 6a 6b 6c 6d 6e 6f
```

### Running Specific Variants

```bash
# Only XXX variant
DATASET_NAME=FAERS_TB python run_pipeline.py --variant xxx

# Only XXX-Gender variant
DATASET_NAME=FAERS_TB_DRUGS python run_pipeline.py --variant xxx_gender

# All variants (default)
DATASET_NAME=FAERS_TB python run_pipeline.py --variant all
```

## Configuration

Edit `scripts3/config.py` to change the default dataset:

```python
# Choose which dataset to use
DATASET_NAME = os.environ.get("DATASET_NAME", "FAERS_ALL")
# Change to: "FAERS_TB" or "FAERS_TB_DRUGS"
```

## Output Structure

Each dataset generates independent outputs:

```
output_faers_all/
├── graphs_FAERS_ALL/
├── models_FAERS_ALL/
└── results_FAERS_ALL/

output_faers_tb/
├── graphs_FAERS_TB/
├── models_FAERS_TB/
└── results_FAERS_TB/

output_faers_tb_drugs/
├── graphs_FAERS_TB_DRUGS/
├── models_FAERS_TB_DRUGS/
└── results_FAERS_TB_DRUGS/
```

## Running All Three Datasets

To run experiments on all three datasets sequentially:

```bash
# Create a batch script
cat > run_all_datasets.sh << 'EOF'
#!/bin/bash

# Run FAERS_ALL
echo "Processing FAERS_ALL..."
DATASET_NAME=FAERS_ALL python run_pipeline.py

# Run FAERS_TB
echo "Processing FAERS_TB..."
DATASET_NAME=FAERS_TB python run_pipeline.py

# Run FAERS_TB_DRUGS
echo "Processing FAERS_TB_DRUGS..."
DATASET_NAME=FAERS_TB_DRUGS python run_pipeline.py

echo "All datasets processed!"
EOF

chmod +x run_all_datasets.sh
./run_all_datasets.sh
```

## Comparison with Scripts2

| Feature | Scripts2 | Scripts3 |
|---------|----------|----------|
| Datasets | Single (full FAERS) | Three (ALL, TB, TB_DRUGS) |
| Dataset Selection | Manual path editing | Environment variable |
| Output Organization | Single folder | Dataset-specific folders |
| Configuration | Manual edits | Automatic path management |
| Data Source | `data/standardized_faers_2018_2025` | `data/FAERS_*/` |

## Files Updated

All scripts from scripts2 have been updated in scripts3:
- `config.py` - Added dataset selection and automatic paths
- `step1_preprocessing.py` - Dataset-aware preprocessing
- `step2_aer_graph_construction.py` - Dataset-aware graph construction
- `step3_model.py` - Model definition (unchanged)
- `step4_training.py` - Dataset-aware training
- `step4_training_lowmem.py` - Low memory training variant
- `step5_baselines.py` - Baseline evaluation
- `step6a_ablation_data_volume.py` - Data volume ablation
- `step6b_ablation_contrastive.py` - Contrastive learning ablation
- `step6c_ablation_demographic.py` - Demographic features ablation
- `step6d_ablation_graph_structure.py` - Graph structure ablation
- `step6e_ablation_node_features.py` - Node features ablation
- `step6f_ablation_summary.py` - Ablation summary compilation
- `run_pipeline.py` - Master pipeline runner
- `diagnose_training.py` - Training diagnostics
- `create_demographic_figures.py` - Visualization scripts
- `create_variant_figures.py` - Visualization scripts

## Notes

- **Scripts2 remains unchanged**: Your original scripts in `scripts2/` are untouched
- **Independent execution**: Scripts3 runs completely independently from scripts2
- **Data requirements**: Each dataset must be built using `build_datasets.py` before use
- **GPU support**: All training scripts support GPU acceleration via `--device cuda`
- **Memory optimization**: Use `step4_training_lowmem.py` for memory-constrained environments

## Quick Start

```bash
# 1. Build all datasets
python build_datasets.py

# 2. Run experiments on FAERS_TB
cd scripts3
DATASET_NAME=FAERS_TB python run_pipeline.py

# 3. Check results
ls ../output_faers_tb/results_FAERS_TB/
```

## Troubleshooting

### Dataset not found error
- Make sure you ran `python build_datasets.py` first
- Check that `data/FAERS_TB/` or `data/FAERS_TB_DRUGS/` exists

### Wrong dataset being used
- Verify `DATASET_NAME` environment variable is set correctly
- Check `config.py` for default dataset setting
- Use `python config.py` to print current configuration

### Out of memory errors
- Use `step4_training_lowmem.py` instead of `step4_training.py`
- Reduce `batch_size` in `config.py`
- Increase `accumulation_steps` in `config.py`

## Contact

For questions or issues, please refer to the main project documentation.
