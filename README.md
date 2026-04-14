# RUL-Mamba: Mamba-Based Remaining Useful Life Prediction for Lithium-Ion Batteries

> **Authors:**
Jiahui Huang, Lei Liu, Hongwei Zhao, Tianqi Li, Bin Li.

This repo contains the code and data from our paper published in Journal of Energy Storage.

Website: https://www.sciencedirect.com/science/article/pii/S2352152X25010898?dgcid=author.

## 1. Abstract

Lithium-ion batteries play a crucial role in the fields of renewable energy and electric vehicles. Accurately predicting their Remaining Useful Life (RUL) is essential for ensuring safe and reliable operation. However, achieving precise RUL predictions poses significant challenges due to the complexities of degradation mechanisms and the impact of operational noise, particularly the capacity regeneration phenomenon. To address these issues, we propose a lithium-ion battery RUL prediction model named RUL-Mamba, which is based on the Mamba-Feature Attention Network (FAN)-Gated Residual Network (GRN). This model employs an encoder-decoder architecture that effectively integrates the Mamba module, FAN network, and GRN network. Mamba demonstrates superior temporal representation capabilities alongside efficient inference properties. The constructed FAN network leverages a feature attention mechanism to efficiently extract key features at each time step, enabling the Mamba block in the encoder to effectively capture information related to capacity regeneration from historical capacity sequences. The designed GRN network adaptively processes the decoded features output by the Mamba blocks in the decoder through a gating mechanism, accurately modeling the nonlinear mapping relationship between the decoded feature vector and the prediction target. Compared to state-of-the-art (SOTA) time series forecasting models on three battery degradation datasets from NASA, Oxford and Tongji University, the proposed model not only achieves SOTA predictive performance across various prediction starting points, with a maximum accuracy improvement of 42.5% over existing models, but also offers advantages such as efficient training, fast inference and being less influenced by the prediction starting point.

## 2. Environment setup

- first method (recommended)

```bash
conda env create -f rulmamba_env.yaml
conda activate rulmamba
```

- second method

```bash
conda create -n rulmamba python=3.10.13
conda activate rulmamba
pip install torch==1.13.1
pip install -r requirements.txt
```

## 3. Datasets

The TJU, NASA, and Oxford datasets are already placed in the `Data/` folder. The URLs of three datasets are as follows:

- **TJU dataset**: https://zenodo.org/records/6405084
- **NASA dataset**: https://www.nasa.gov/content/prognostics-center-of-excellence-data-set-repository
- **Oxford dataset**: https://doi.org/10.5287/bodleian:KO2kdmYGg

### 3.1. TJU Dataset

The Tongji University (TJU) dataset contains battery degradation data collected from NCM-NCA batteries under constant current (1C) discharge conditions. The dataset includes multiple battery cells with complete cycle-by-cycle capacity measurements, making it suitable for RUL prediction tasks.

**Data Structure:**
- **Raw Data**: `Data/TJU_Data/Dataset_3_NCM_NCA_Battery/` - Contains CSV files for each battery cell (e.g., `CY25-05_1-#1.csv`, `CY25-05_1-#2.csv`, etc.)
- **Processed Data**: `Data/TJU_Data/Dataset_3_NCM_NCA_Battery_1C.npy` - Preprocessed numpy array ready for model training
- **Battery Cells**: Multiple cells with names like `CY25_1`, `CY25_2`, etc.

### 3.2. NASA Dataset

The NASA dataset provides data from four lithium-ion batteries (#5, 6, 7, and 18) tested under room temperature conditions. Each battery underwent charge, discharge, and impedance measurement cycles until reaching end-of-life (EOL) criteria (30% capacity fade from 2Ah to 1.4Ah).

**Data Structure:**
- **Raw Data**: `Data/NASA_Data/B0005.mat`, `B0006.mat`, `B0007.mat`, `B0018.mat` - MATLAB format files containing complete cycle data
- **Processed Data**: `Data/NASA_Data/NASA.npy` - Preprocessed numpy array
- **Batteries**: B0005, B0006, B0007, B0018

### 3.3. Oxford Dataset

The Oxford Battery Degradation Dataset 1 contains measurements from 8 small lithium-ion pouch cells (740mAh capacity) tested in a thermal chamber at 40°C. Cells underwent constant-current-constant-voltage (CC-CV) charging followed by drive cycle discharging based on the urban Artemis profile.

**Data Structure:**
- **Raw Data**: 
  - `Data/Oxford_Data/Oxford_Battery_Degradation_Dataset_1.mat` - Characterization tests every 100 cycles (~262MB)
  - `Data/Oxford_Data/ExampleDC_C1.mat` - First charge-discharge cycle example
- **Processed Data**: `Data/Oxford_Data/Oxford.npy` - Preprocessed numpy array
- **CSV Export**: `Data/Oxford_Data/Oxford.csv` - Converted CSV format
- **Battery Cells**: Cell1, Cell2, Cell3, Cell4, Cell5, Cell6, Cell7, Cell8

## 4. Usage

### 4.1. Overview

This project provides comprehensive scripts for data processing, model training, hyperparameter optimization, and result visualization. The scripts are organized into four main categories:

1. **Data Processing Scripts** - Preprocess raw battery data
2. **Training Scripts** - Train RUL prediction models
3. **Optimization Scripts** - Perform hyperparameter tuning
4. **Chart Creation Scripts** - Generate visualization and plots

### 4.2. Data Processing Scripts

#### 4.2.1. TJU Data Processing

```bash
python Scripts/Data_Process/TJU_Data_Process.py
```

#### 4.2.2. NASA Data Processing

```bash
python Scripts/Data_Process/NASA_Data_Process.py
```

#### 4.2.3. Oxford Data Processing

```bash
python Scripts/Data_Process/Oxford_Data_Process.py
```

### 4.3. Training Scripts

#### 4.3.1. TJU Univariate RUL Prediction

**Script**: `Scripts/TJU_Univariable_RUL_Prediction/Train_TJU_Univariable.py`

**Wrapper**: `Scripts/TJU_Univariable_RUL_Prediction/Train_TJU_Univariable.sh`

**Supported Models**: Autoformer, FEDformer, MambaSimple, PatchTST, PathFormer, RULMamba, TimeMixer, TimesNet

**Usage (Shell Script - Recommended):**
```bash
# Train all models
bash Scripts/TJU_Univariable_RUL_Prediction/Train_TJU_Univariable.sh

# Train specific models
bash Scripts/TJU_Univariable_RUL_Prediction/Train_TJU_Univariable.sh Autoformer RULMamba

# Train with additional parameters
bash Scripts/TJU_Univariable_RUL_Prediction/Train_TJU_Univariable.sh all --test-name CY25_1 --count 10 --gpu-id 0
```

**Usage (Python Script - Direct):**
```bash
python Scripts/TJU_Univariable_RUL_Prediction/Train_TJU_Univariable.py \
  --config Configs/TJU/Univariable/Base.yaml \
  --model RULMamba \
  --model-config Configs/TJU/Univariable/RULMamba.yaml \
  --test-name CY25_1 \
  --max-epochs 200 \
  --gpu-id 0 \
  --batch-size 128
```

#### 4.3.2. NASA Univariate RUL Prediction

**Script**: `Scripts/NASA_Univariable_RUL_Prediction/Train_NASA_Univariable.py`

**Wrapper**: `Scripts/NASA_Univariable_RUL_Prediction/Train_NASA_Univariable.sh`

**Supported Models**: Autoformer, FEDformer, MambaSimple, PatchTST, PathFormer, RULMamba, TimeMixer, TimesNet

**Usage (Shell Script):**
```bash
# Train all models
bash Scripts/NASA_Univariable_RUL_Prediction/Train_NASA_Univariable.sh

# Train specific models
bash Scripts/NASA_Univariable_RUL_Prediction/Train_NASA_Univariable.sh Autoformer RULMamba

# Train with additional parameters
bash Scripts/NASA_Univariable_RUL_Prediction/Train_NASA_Univariable.sh all --test-name B0005 --count 10 --gpu-id 0
```

**Usage (Python Script):**
```bash
python Scripts/NASA_Univariable_RUL_Prediction/Train_NASA_Univariable.py \
  --config Configs/NASA/Univariable/Base.yaml \
  --model RULMamba \
  --model-config Configs/NASA/Univariable/RULMamba.yaml \
  --test-name B0005 \
  --max-epochs 200 \
  --gpu-id 0 \
  --batch-size 16
```

#### 4.3.3. Oxford Univariate RUL Prediction

**Script**: `Scripts/Oxford_Univariable_RUL_Prediction/Train_Oxford_Univariable.py`

**Wrapper**: `Scripts/Oxford_Univariable_RUL_Prediction/Train_Oxford_Univariable.sh`

**Supported Models**: Autoformer, FEDformer, MambaSimple, PatchTST, PathFormer, RULMamba, TimeMixer, TimesNet

**Usage (Shell Script):**
```bash
# Train all models
bash Scripts/Oxford_Univariable_RUL_Prediction/Train_Oxford_Univariable.sh

# Train specific models
bash Scripts/Oxford_Univariable_RUL_Prediction/Train_Oxford_Univariable.sh Autoformer RULMamba

# Train with additional parameters
bash Scripts/Oxford_Univariable_RUL_Prediction/Train_Oxford_Univariable.sh all --test-name Cell8 --count 10 --gpu-id 0
```

**Usage (Python Script):**
```bash
python Scripts/Oxford_Univariable_RUL_Prediction/Train_Oxford_Univariable.py \
  --config Configs/Oxford/Univariable/Base.yaml \
  --model RULMamba \
  --model-config Configs/Oxford/Univariable/RULMamba.yaml \
  --test-name Cell8 \
  --max-epochs 200 \
  --gpu-id 0 \
  --batch-size 8
```

#### 4.3.4. TJU Multivariate RUL Prediction

**Script**: `Scripts/TJU_Multivariable_RUL_Prediction/Train_TJU_Multivariable.py`

**Wrapper**: `Scripts/TJU_Multivariable_RUL_Prediction/Train_TJU_Multivariable.sh`

**Supported Models**: Autoformer, FEDformer, MambaSimple, PatchTST, PathFormer, RULMambaVAN, TimeMixer, TimesNet

**Usage (Shell Script):**
```bash
# Train all models
bash Scripts/TJU_Multivariable_RUL_Prediction/Train_TJU_Multivariable.sh

# Train specific models
bash Scripts/TJU_Multivariable_RUL_Prediction/Train_TJU_Multivariable.sh Autoformer RULMambaVAN

# Train with additional parameters
bash Scripts/TJU_Multivariable_RUL_Prediction/Train_TJU_Multivariable.sh all --test-name CY25_1 --count 10 --gpu-id 0
```

**Usage (Python Script):**
```bash
python Scripts/TJU_Multivariable_RUL_Prediction/Train_TJU_Multivariable.py \
  --config Configs/TJU/Multivariable/Base.yaml \
  --model RULMambaVAN \
  --model-config Configs/TJU/Multivariable/RULMambaVAN.yaml \
  --test-name CY25_1 \
  --max-epochs 200 \
  --gpu-id 0 \
  --batch-size 8
```

### 4.4. Optimization Scripts

#### 4.4.1. TJU Hyperparameter Optimization

```bash
python Scripts/TJU_Univariable_RUL_Prediction/Optimize_TJU_Univariable_RULMamba.py \
  --model RULMamba \
  --seq-len 64 \
  --test-name CY25_1 \
  --start-points 200 300 400 \
  --count 10 \
  --seed 2025
```

#### 4.4.2. NASA Hyperparameter Optimization

```bash
python Scripts/NASA_Univariable_RUL_Prediction/Optimize_NASA_Univariable_RULMamba.py \
  --model RULMamba \
  --seq-len 16 \
  --test-name B0005 \
  --start-points 50 70 90 \
  --count 10 \
  --seed 2025
```

#### 4.4.3. Oxford Hyperparameter Optimization

```bash
python Scripts/Oxford_Univariable_RUL_Prediction/Optimize_Oxford_Univariable_RULMamba.py \
  --model RULMamba \
  --seq-len 10 \
  --test-name Cell8 \
  --start-points 20 30 40 \
  --count 10 \
  --seed 2025
```

#### 4.4.4. TJU Multivariate Optimization

```bash
python Scripts/TJU_Multivariable_RUL_Prediction/Optimize_TJU_Multivariable_RULMambaVAN.py \
  --model RULMambaVAN \
  --seq-len 64 \
  --test-name CY25_1 \
  --start-points 200 300 400 \
  --count 10 \
  --seed 2025
```

### 4.5. Chart Creation Scripts

#### 4.5.1. Capacity Prediction Curves Plotting

**Script**: `Scripts/Charts_Creation/Plot_Capacity_Prediction_Curves.py`

**Wrapper**: `Scripts/Charts_Creation/Plot_Capacity_Prediction_Curves.sh`

**Purpose**: Generate capacity prediction curve visualizations for trained models across all datasets.

**Usage (Shell Script):**
```bash
# Plot for TJU dataset - all models
bash Scripts/Charts_Creation/Plot_Capacity_Prediction_Curves.sh \
  --config Configs/TJU/Univariable/Base.yaml \
  --model RULMamba \
  --test-name CY25_1 \
  --plot-mode all

# Plot for NASA dataset
bash Scripts/Charts_Creation/Plot_Capacity_Prediction_Curves.sh \
  --config Configs/NASA/Univariable/Base.yaml \
  --model PatchTST \
  --test-name B0005 \
  --plot-mode mean

# Plot for Oxford dataset
bash Scripts/Charts_Creation/Plot_Capacity_Prediction_Curves.sh \
  --config Configs/Oxford/Univariable/Base.yaml \
  --model TimesNet \
  --test-name Cell8 \
  --plot-mode repeat
```

**Usage (Python Script):**
```bash
python Scripts/Charts_Creation/Plot_Capacity_Prediction_Curves.py \
  --config Configs/TJU/Univariable/Base.yaml \
  --model RULMamba \
  --test-name CY25_1 \
  --plot-mode all \
  --result-path Results/TJU_Univariable_RULMamba_CY25_1.pth \
  --real-data-path Results/Capacity_CY25_1_Real_Data.pth \
  --save-dir Plots/TJU/Univariable/CY25_1/RULMamba
```

**Key Parameters:**
- `--config`: Base configuration file
- `--model`: Model name
- `--test-name`: Test battery name
- `--plot-mode`: Plot mode - `repeat` (individual runs), `mean` (average across runs), or `all` (both)
- `--result-path`: Path to prediction results file
- `--real-data-path`: Path to real capacity data file
- `--save-dir`: Directory to save plots
- `--start-points`: Custom start points for prediction

**Example - Batch Plotting All Models for All Datasets:**

```bash
# TJU Dataset - Univariable Input
for model in Autoformer FEDformer MambaSimple PatchTST PathFormer RULMamba TimeMixer TimesNet; do
  bash Scripts/Charts_Creation/Plot_Capacity_Prediction_Curves.sh \
    --config "Configs/TJU/Univariable/Base.yaml" \
    --model "$model" \
    --test-name CY25_1 \
    --plot-mode all
done

# TJU Dataset - Multivariable Input
for model in Autoformer FEDformer MambaSimple PatchTST PathFormer RULMambaVAN TimeMixer TimesNet; do
  bash Scripts/Charts_Creation/Plot_Capacity_Prediction_Curves.sh \
    --config "Configs/TJU/Multivariable/Base.yaml" \
    --model "$model" \
    --test-name CY25_1 \
    --plot-mode all
done

# NASA Dataset - Univariable Input
for model in Autoformer FEDformer MambaSimple PatchTST PathFormer RULMamba TimeMixer TimesNet; do
  bash Scripts/Charts_Creation/Plot_Capacity_Prediction_Curves.sh \
    --config "Configs/NASA/Univariable/Base.yaml" \
    --model "$model" \
    --test-name B0005 \
    --plot-mode all
done

# Oxford Dataset - Univariable Input
for model in Autoformer FEDformer MambaSimple PatchTST PathFormer RULMamba TimeMixer TimesNet; do
  bash Scripts/Charts_Creation/Plot_Capacity_Prediction_Curves.sh \
    --config "Configs/Oxford/Univariable/Base.yaml" \
    --model "$model" \
    --test-name Cell8 \
    --plot-mode all
done
```

### 4.6. Configuration Files

Configuration files are organized in the `Configs/` directory:
- `Configs/TJU/Univariable/` - TJU univariate configurations
- `Configs/TJU/Multivariable/` - TJU multivariate configurations
- `Configs/NASA/Univariable/` - NASA univariate configurations
- `Configs/Oxford/Univariable/` - Oxford univariate configurations

Each dataset folder contains:
- `Base.yaml` - Base configuration with dataset parameters
- `<Model>.yaml` - Model-specific configurations (e.g., RULMamba.yaml, TimesNet.yaml)

Modify these files to customize training parameters, model architectures, and experimental settings.

### 4.7. Times New Roman Font Setup

The plotting scripts use the 'Times New Roman' font family. If you want `Times New Roman` to be available globally in matplotlib (e.g., via `rcParams`), you need to install the font at the system level. Reference: [CSDN Blog](https://blog.csdn.net/qq_49323609/article/details/139026798).

After this setup, scripts that use `plt.rcParams['font.family'] = 'Times New Roman'` will work correctly.

## 5. Citation

If you find our work useful in your research, please consider citing:

```latex
@article{HUANG2025116376,
    title = {RUL-Mamba: Mamba-based remaining useful life prediction for lithium-ion batteries},
    journal = {Journal of Energy Storage},
    volume = {120},
    pages = {116376},
    year = {2025},
    issn = {2352-152X},
    doi = {https://doi.org/10.1016/j.est.2025.116376},
    url = {https://www.sciencedirect.com/science/article/pii/S2352152X25010898},
    author = {Jiahui Huang and Lei Liu and Hongwei Zhao and Tianqi Li and Bin Li},
}
```

If you have any problems, contact me via liulei13@ustc.edu.cn.