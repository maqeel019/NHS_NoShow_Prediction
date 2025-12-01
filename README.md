# 🏥 NHS No-Show Prediction
This repository contains scripts, notebooks, and documentation for predicting patient no-shows in NHS patient appointments.  
It covers data collection, preprocessing, feature engineering, and model evaluation.
---

## 📁 Repository Structure

### **data/**
Contains all datasets used in the project.

- **hosp-epis-stat-outp-all-atte-2024-25-tab.xlsx** – main dataset with all outpatient attendance records  
- **hosp-epis-stat-outp-ethn-cat-2024-25-tab.xlsx** – ethnicity-based attendance data  
- **hosp-epis-stat-outp-imd-dec-2024-25-tab.xlsx** – deprivation index–based attendance data  
- **processed/** – cleaned and transformed data ready for modeling:
  - `attendance_all_long.csv`, `attendance_female_long.csv`, `attendance_male_long.csv` – reshaped attendance data  
  - `ethnicity_no_show_probs.csv`, `imd_no_show_probs.csv` – derived probability tables  
  - `nhs_no_show_enhanced_synthetic.csv` – final generated dataset used for model training

---

### **docs/**
Contains all project documentation and reports.

- `Data_Understanding_Report.pdf` – summary of data sources and initial findings   
- `methodology.md` – explanation of modeling approach and data logic  
- `project_background.md` – background and motivation of the project  

---

### **src/**
Core scripts and notebooks used for data preparation and modeling.

- `data_understanding.ipynb` – exploratory analysis of each raw NHS dataset separately  
- `data_prep.ipynb` – performs ETL for each dataset and creates cleaned datasets ready for generating enhanced synthetic dataset
- `generate_synthetic_nhs.py` – generates an enhanced synthetic dataset by combining the three datasets into **250,000 records**, preserving realistic distributions  
- `modeling.ipynb` – model training, evaluation, threshold tuning, and performance metrics  
- `__init__.py` – marks this folder as a Python package

---

### **requirements.txt**
Lists all Python dependencies required to run scripts and notebooks.

### **environment.yml**
Conda environment setup file containing equivalent dependencies.

### **README.md**
This file — describes the repository structure and each file’s purpose.

---

## ⚙️ Usage

To reproduce the workflow, execute the notebooks in `/src` in the following order:

1. `data_understanding.ipynb`  
2. `data_prep.ipynb`  
3. `generate_synthetic_nhs.py`  
4. `modeling.ipynb`

---

## 🧠 Key Concepts

- Predicting patient appointment **no-shows** using demographic and socio-economic data  
- Applying **data cleaning, feature engineering, and model optimization**  
- Evaluating models using metrics such as **Accuracy, F1 Score, and ROC-AUC**

---

## 🧩 Author
**Muhammad Aqeel**  
LinkedIn: [Muhammad Aqeel](https://www.linkedin.com/in/aqeelkhan09/)


---

