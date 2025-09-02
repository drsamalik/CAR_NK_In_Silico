# A framework integrating multiscale in-silico modeling and experimental data predicts CAR-NK cytotoxicity across target cell types

This project presents a model for CAR-NK cytotoxicity across various cell lines. It employs a multi-objective optimization strategy to define conditions for on-target, off-tumor lysis. The following sections provide detailed donor-specific training data and predictions for different cell lines.

---

## Directory Structure

### Main Folders
- <b style="color:#0074d9;">Donor_A_Kasumi1_HL60 and Donor_B_Kasumi1_HL60</b>  
  Contains model training for Kasumi1 and model predictions for HL60, utilizing receptor-ligand distribution data from Donors A and B.  
  The file `Donor_A_Kasumi1_HL60_Main.ipynb` serves as the main notebook in this folder, importing input data via `Model_CAR_NK.py` and `Model_Wt_NK.py` through `imp_exp_data.py`. Model training and prediction are performed using `fitting.py`, which can load optimizers from `optimizer_param.py`.

- <b style="color:#0074d9;">Donor_C_DIPG36_SKOV3 and Donor_D_DIPG36_SKOV3</b>  
  Contains model training for DIPG36 and subsequent predictions for SKOV3, based on distribution data from Donors C and D.

- <b style="color:#0074d9;">Donor_E_Mv411, Donor_F_Mv411 and Donor_G_Mv411</b>  
  Contains model training for Donors E, F, and G at 48 hours, along with predictions at 72 hours for the Mv411 cell line.

- <b style="color:#0074d9;">Donor_H_Kasumi1_Monocyte</b>  
  Contains model training for Kasumi1 and predictions for Monocytes. It includes two subfolders, Donor_H_Kasumi1_Monocyte and Donor_H_Kasumi1_Monocyte_max_likelihood, representing training and prediction performed at different levels of %CAR positivity. Parameter estimation via maximum likelihood is conducted using the notebook Plot_Mixed_Max_likelyhood_lognml.ipynb.

- <b style="color:#0074d9;">In_vivo_Prediction</b>  
  Contains model training based on wild-type cytotoxicity data and predictions for CAR-NK cytotoxicity.

- <b style="color:#0074d9;">Pareto_optimization/</b>  
  Contains analyses related to multi-objective optimization approaches.

- <b style="color:#0074d9;">Confidence_Interval_w_Donor_H</b>  
  Contains files for finding confidence interval for donor H

- <b style="color:#0074d9;">Traing_and_Prediction_w_mean</b>  
  Contains model fitting and predictions utilizing average molecule counts per cell aggregated across all donors.

---

## Setup

### Prerequisites

Ensure that Python 3.11 or higher is installed along with the following packages:
- numpy
- pandas
- matplotlib
- scipy
- platypus