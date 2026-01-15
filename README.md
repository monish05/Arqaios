# Human Identification Using mmWave Point Clouds - POC

## Overview

This is a **Proof-of-Concept (POC)** implementation for human identification using point cloud data, based on the MMIDNet architecture from the research paper "MMIDNet: Secure Human Identification Using Millimeter-wave Radar and Deep Learning" (Shen et al., 2024).

**What it does:**
- Uses the MPI-FAUST 3D mesh dataset to simulate mmWave radar point cloud sequences
- Implements the full MMIDNet architecture (T-Net, Residual CNN, Bi-LSTM)
- Achieves 70% test accuracy on 6 combined classes

---

## Quick Start (Google Colab)

### Step 1: Upload Project to Google Drive

1. Upload the entire project folder to your Google Drive (e.g., `MyDrive/Arqaios`)
2. Ensure your folder structure looks like this:
   ```
   Arqaios/
   ├── src/
   ├── notebooks/
   ├── MPI-FAUST/
   └── requirements.txt
   ```

### Step 2: Download MPI-FAUST Dataset

1. Download the MPI-FAUST dataset from [http://faust.is.tue.mpg.de/](http://faust.is.tue.mpg.de/) (registration required)
2. Extract the dataset and place the `MPI-FAUST` folder in the project root:
   ```
   Arqaios/
   └── MPI-FAUST/
       └── training/
           └── scans/
               ├── tr_scan_000.ply
               ├── tr_scan_001.ply
               └── ... (100 PLY files total)
   ```

### Step 3: Run the Notebook

1. Open `notebooks/mmidnet_pipeline.ipynb` in Google Colab
2. The first cell will automatically:
   - Mount Google Drive
   - Install dependencies from `requirements.txt`
   - Set up the environment
3. Run all cells sequentially to execute the complete pipeline:
   - Data preparation and visualization
   - Model training
   - Evaluation and results

That's it! The notebook handles everything automatically.

---

## Reference

This project implements the MMIDNet architecture from:

**Shen, Z., Nunez-Yanez, J., & Dahnoun, N. (2024). MMIDNet: Secure Human Identification Using Millimeter-wave Radar and Deep Learning. In *2024 13th Mediterranean Conference on Embedded Computing (MECO)* (pp. 1-7). Budva, Montenegro: IEEE. doi: 10.1109/MECO62516.2024.10577920**

```bibtex
@inproceedings{Shen:MECO:2024,
  title = {{MMIDNet}: Secure Human Identification Using Millimeter-wave Radar and Deep Learning},
  author = {Shen, Z. and Nunez-Yanez, J. and Dahnoun, N.},
  booktitle = {2024 13th Mediterranean Conference on Embedded Computing (MECO)},
  address = {Budva, Montenegro},
  publisher = {IEEE},
  pages = {1--7},
  year = {2024},
  doi = {10.1109/MECO62516.2024.10577920}
}
```

---

## Dataset Citation

This project uses the MPI-FAUST dataset:

**Bogo, F., Romero, J., Loper, M., & Black, M. J. (2014). FAUST: Dataset and evaluation for 3D mesh registration. In *Proceedings IEEE Conf. on Computer Vision and Pattern Recognition (CVPR)*. IEEE.**

```bibtex
@inproceedings{Bogo:CVPR:2014,
  title = {{FAUST}: Dataset and evaluation for {3D} mesh registration},
  author = {Bogo, Federica and Romero, Javier and Loper, Matthew and Black, Michael J.},
  booktitle = {Proceedings IEEE Conf. on Computer Vision and Pattern Recognition (CVPR)},
  address = {Piscataway, NJ, USA},
  publisher = {IEEE},
  month = jun,
  year = {2014}
}
```

---

## Project Structure

```
Arqaios/
├── src/                    # Source code modules
│   ├── data_utils.py      # Data loading and preprocessing
│   ├── model.py           # MMIDNet architecture
│   ├── train.py           # Training utilities
│   └── evaluate.py        # Evaluation metrics
├── notebooks/             # Jupyter notebooks
│   ├── mmidnet_pipeline.ipynb    # Main pipeline (run this!)
│   └── explore_faust_data.ipynb  # Data exploration
├── MPI-FAUST/            # Dataset (download separately)
└── requirements.txt      # Python dependencies
```

---

## Results

- **Test Accuracy**: 70.00%
- **Classes**: 6 combined classes (from 10 original subjects)
- **Architecture**: Full MMIDNet (T-Net + Residual CNN + Bi-LSTM)
- **Preprocessing**: FPS sampling, subject-specific augmentation, radar-like characteristics
