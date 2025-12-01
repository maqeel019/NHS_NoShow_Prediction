
---

# **Methodology**

This project focuses on generating and analyzing a **synthetic NHS appointment dataset** to study factors influencing patient attendance and missed appointments. The workflow was designed to ensure realism, consistency, and analytical value.

---

## **1. Data Collection**

Data was created by combining and transforming three different NHS-related datasets:

* **Ethnicity dataset** – contained attendance counts by ethnic group.
* **Deprivation dataset (IMD Decile)** – represented socioeconomic status by region.
* **Gender and Age dataset** – contained appointment records grouped by demographic features.

These datasets were merged and extended to produce a **synthetic dataset** of **200,000 records** simulating real NHS appointment patterns. The generated data includes variables such as:

* IMD Decile (deprivation level)
* Ethnicity (mapped using NHS codes)
* Age group
* Gender
* Previous appointments and no-shows
* Reminder status
* Booking and appointment dates
* No-show probability and outcome

---

## **2. Data Cleaning**

The merged dataset underwent several cleaning operations to ensure data quality:

* Removed invalid or incomplete records.
* Standardized gender values to **Male** and **Female**.
* Corrected age group formatting for consistency (e.g., “10–14”, “70–74”).
* Mapped ethnicity codes to descriptive NHS-standard labels (e.g., `A → British (White)`, `Z → Not stated`, `99 → Not known`).
* Ensured categorical values were properly encoded and free from duplicates.

---

## **3. Feature Preparation**

Additional calculated and adjusted fields were introduced to make the data analysis-ready:

* **NoShow_Prob**: Computed from attendance columns to estimate the probability of missed appointments.
* **NoShow_Prob_Adjusted**: Adjusted probability accounting for demographics and reminder influence.
* Derived binary variable **NoShow** to indicate whether the appointment was missed (`1`) or attended (`0`).

All numeric and date fields were validated and standardized for modeling and visualization.

---

## **4. Data Validation**

The synthetic dataset was checked to ensure logical and statistical consistency:

* Verified that all probabilities were within the range **0–1**.
* Randomly sampled rows were inspected for valid date relationships (booking < appointment).
* Category distributions (e.g., ethnicity, gender) were compared to real NHS statistics for proportional similarity.

---

## **5. Data Export**

The finalized dataset was exported as:
📁 **`synthetic_nhs_no_show_clean.csv`**

This dataset will serve as the foundation for exploratory data analysis, visualization, and predictive modeling tasks.

---

## **6. Ethical Considerations**

* All generated data is **synthetic** and does **not contain any real patient information**.
* Data generation followed NHS coding standards to ensure representational accuracy.
* The project maintains compliance with **GDPR** principles for data protection and privacy.

---


---

