import pandas as pd
import numpy as np



import pandas as pd

# 1️⃣ First file
df1 = pd.read_excel("./data/hosp-epis-stat-outp-ethn-cat-2024-25-tab.xlsx")
print("Header for hosp-epis-stat-outp-ethn-cat-2024-25-tab.xlsx:")
print(df1.columns.tolist())

# 2️⃣ Second file
df2 = pd.read_excel("./data/hosp-epis-stat-outp-imd-dec-2024-25-tab.xlsx")
print("\nHeader for hosp-epis-stat-outp-imd-dec-2024-25-tab.xlsx:")
print(df2.columns.tolist())

# # 3️⃣ Third file
df3 = pd.read_excel(
    "data/hosp-epis-stat-outp-all-atte-2024-25-tab.xlsx",
    sheet_name=1   # first sheet
)
print("\nHeader for hosp-epis-stat-outp-all-atte-2024-25-tab.xlsx:")
print(df3.columns.tolist())



# # 1. Load summary files
# def load_nhs_files():
#     all_att = pd.read_excel('data/hosp-epis-stat-outp-rep-tabs-2024-25-tab.xlsx', sheet_name='All Attendances')
#     male_att = pd.read_excel('data/hosp-epis-stat-outp-rep-tabs-2024-25-tab.xlsx', sheet_name='Male Attendances')
#     female_att = pd.read_excel('data/hosp-epis-stat-outp-rep-tabs-2024-25-tab.xlsx', sheet_name='Female Attendances')
#     imd = pd.read_csv('data/imd.csv')
#     ethnicity = pd.read_csv('data/ethnic_category.csv')
#     return all_att, male_att, female_att, imd, ethnicity

# # 2. Compute attendance probabilities
# def compute_probabilities(all_att, male_att, female_att):
#     # Example: calculate overall no-show rate
#     pass

# # 3. Generate synthetic patients
# def generate_synthetic_patients(n_patients, probabilities, imd, ethnicity):
#     # Create patient-level dataframe
#     pass

# # 4. Save dataset
# def save_dataset(df, filename='data/raw/nhs_no_show_synthetic.csv'):
#     df.to_csv(filename, index=False)
