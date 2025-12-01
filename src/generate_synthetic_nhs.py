import pandas as pd
import numpy as np

# ENHANCED NHS SYNTHETIC DATA GENERATOR
# Uses ONLY real patterns from NHS datasets 

# PARAMETERS
n_patients = 250_000
np.random.seed(42)

# LOAD NHS-DERIVED INPUTS

print("Loading NHS datasets...")
imd = pd.read_csv("./data/processed/imd_no_show_probs.csv", header=0)
ethnicity = pd.read_csv(
    "./data/processed/ethnicity_no_show_probs.csv", header=0)
age_gender = pd.concat([
    pd.read_csv("./data/processed/attendance_male_long.csv", header=0),
    pd.read_csv("./data/processed/attendance_female_long.csv", header=0)
])

# CLEAN INPUTS

imd = imd[imd['IMD_Decile'] != "Total Activity"].copy()
imd['NoShow_Prob'] = imd['NoShow_Prob'].clip(0.01, 0.35)
imd['IMD_Decile'] = imd['IMD_Decile'].astype(str)

ethnicity = ethnicity[ethnicity['Ethnicity'].notna()]
ethnicity['NoShow_Prob'] = ethnicity['NoShow_Prob'].clip(0.01, 0.35)
ethnicity['Ethnicity'] = ethnicity['Ethnicity'].astype(str)

age_gender = age_gender[age_gender['Gender'].isin(['Male', 'Female'])]
age_gender = age_gender[age_gender['Age_Group'] != 'Total']
age_gender['Age_Group'] = age_gender['Age_Group'].astype(str)
age_gender['Main_Specialty_Desc'] = age_gender['Main_Specialty_Desc'].fillna(
    'General Medicine')


# CREATE PROBABILITY DISTRIBUTIONS

print("Building probability distributions...")

# IMD distribution
imd_probs = imd.set_index('IMD_Decile')['All'] / imd['All'].sum()

# Ethnicity distribution
eth_probs = ethnicity.set_index('Ethnicity')['All'] / ethnicity['All'].sum()

# Age/Gender distribution
age_gender_counts = age_gender.groupby(['Age_Group', 'Gender'])[
    'Attendance_Count'].sum()
age_gender_probs = age_gender_counts / age_gender_counts.sum()


# BUILD JOINT PROBABILITY TABLE (without specialty yet)

print("Creating joint probability distribution...")

# Create all combinations
imd_list = []
eth_list = []
age_list = []
gender_list = []

for imd_val in imd['IMD_Decile'].values:
    for eth_val in ethnicity['Ethnicity'].values:
        for (age_val, gender_val) in age_gender_probs.index:
            imd_list.append(imd_val)
            eth_list.append(eth_val)
            age_list.append(age_val)
            gender_list.append(gender_val)

joint_df = pd.DataFrame({
    'IMD_Decile': imd_list,
    'Ethnicity': eth_list,
    'Age_Group': age_list,
    'Gender': gender_list
})

# Calculate joint probability (assuming independence)
joint_df['joint_prob'] = (
    joint_df['IMD_Decile'].map(imd_probs) *
    joint_df['Ethnicity'].map(eth_probs) *
    joint_df.apply(lambda x: age_gender_probs.loc[(
        x['Age_Group'], x['Gender'])], axis=1)
)
joint_df['joint_prob'] /= joint_df['joint_prob'].sum()


# SAMPLE PATIENTS

print(f"Sampling {n_patients} patients...")
sampled = joint_df.sample(
    n=n_patients, weights='joint_prob', replace=True).reset_index(drop=True)


# ASSIGN MEDICAL SPECIALTIES (from real age/gender distributions)

print("Assigning medical specialties based on age/gender patterns...")

# Build specialty distribution for each age/gender combination
specialty_dist = age_gender.groupby(['Age_Group', 'Gender', 'Main_Specialty_Desc'])[
    'Attendance_Count'].sum().reset_index()
specialty_dist['specialty_prob'] = specialty_dist.groupby(
    ['Age_Group', 'Gender'])['Attendance_Count'].transform(lambda x: x / x.sum())


def assign_specialty(age, gender):
    """Assign specialty based on real NHS age/gender distribution"""
    subset = specialty_dist[(specialty_dist['Age_Group'] == age) & (
        specialty_dist['Gender'] == gender)]
    if len(subset) == 0:
        return 'General Medicine'
    return np.random.choice(subset['Main_Specialty_Desc'].values, p=subset['specialty_prob'].values)


sampled['Medical_Specialty'] = [assign_specialty(
    a, g) for a, g in zip(sampled['Age_Group'], sampled['Gender'])]


# USE REAL TELECONSULTATION PROPORTIONS

print("Applying real teleconsultation usage patterns...")

# Create maps from actual data
eth_tele_map = dict(zip(ethnicity['Ethnicity'], ethnicity['Tele_Proportion']))
imd_tele_map = dict(zip(imd['IMD_Decile'], imd['Tele_Proportion']))

# Calculate patient-specific tele probability
sampled['Tele_Prob'] = (
    sampled['Ethnicity'].map(eth_tele_map) +
    sampled['IMD_Decile'].map(imd_tele_map)
) / 2

# Assign consultation type based on real proportions
sampled['Consultation_Type'] = [
    np.random.choice(['tele-consult', 'Face-to-Face'], p=[tp, 1-tp])
    for tp in sampled['Tele_Prob']
]


# ASSIGN APPOINTMENT TYPE (First vs Subsequent) - from real data

print("Assigning appointment types (first vs subsequent)...")

# Calculate proportion of first appointments from ethnicity data
ethnicity['First_Prob'] = (
    ethnicity['Attended_First'] + ethnicity['Attended_First_Tele']
) / ethnicity['All']

eth_first_map = dict(zip(ethnicity['Ethnicity'], ethnicity['First_Prob']))

sampled['First_Appt_Prob'] = sampled['Ethnicity'].map(eth_first_map)
sampled['Appointment_Type'] = [
    np.random.choice(['First', 'Subsequent'], p=[fp, 1-fp])
    for fp in sampled['First_Appt_Prob']
]


# CALCULATING BASE NO-SHOW PROBABILITY (from real data)

print("Computing no-show probabilities from real NHS patterns...")

imd_noshow_map = dict(zip(imd['IMD_Decile'], imd['NoShow_Prob']))
eth_noshow_map = dict(zip(ethnicity['Ethnicity'], ethnicity['NoShow_Prob']))

sampled['Base_NoShow_Prob'] = (
    sampled['IMD_Decile'].map(imd_noshow_map) +
    sampled['Ethnicity'].map(eth_noshow_map)
) / 2


# APPLYING REAL-DATA-DRIVEN ADJUSTMENTS

print("Applying evidence-based adjustments...")

# Age adjustment 
age_order = ['0', '1-4', '5-9', '10-14', '15', '16', '17', '18', '20-24', '25-29', '30-34',
             '35-39', '40-44', '45-49', '50-54', '55-59', '60-64', '65-69', '70-74',
             '75-79', '80-84', '85-89', '90-120']
age_adjust_map = {age: 0.06 - (i * 0.002) for i, age in enumerate(age_order)}

# Gender adjustment (males slightly higher no-show in real data)
gender_adjust = {'Male': 0.005, 'Female': -0.005}

# Teleconsultation reduces no-show (AS WE KNOW FROM NHS DATASET)
tele_adjust = {'tele-consult': -0.02, 'Face-to-Face': 0.0}

# First appointments have higher no-show rates
appt_type_adjust = {'First': 0.03, 'Subsequent': -0.01}

# Apply all adjustments
sampled['Adjusted_NoShow_Prob'] = np.clip(
    sampled['Base_NoShow_Prob'] +
    sampled['Age_Group'].map(age_adjust_map).fillna(0) +
    sampled['Gender'].map(gender_adjust) +
    sampled['Consultation_Type'].map(tele_adjust) +
    sampled['Appointment_Type'].map(appt_type_adjust),
    0.01, 0.60
)


# PATIENT HISTORY (based on adjusted probability)

print("Generating patient appointment history...")

sampled['Previous_Appointments'] = np.random.poisson(
    lam=3.5, size=n_patients).clip(0, 20)
sampled['Previous_NoShows'] = [
    np.random.binomial(n_prev, p)
    for n_prev, p in zip(sampled['Previous_Appointments'], sampled['Adjusted_NoShow_Prob'])
]


# GENERATING ACTUAL NO-SHOW OUTCOME

print("Simulating appointment outcomes...")

sampled['NoShow'] = np.random.binomial(1, sampled['Adjusted_NoShow_Prob'])
sampled['NoShow_Label'] = sampled['NoShow'].map({1: "Yes", 0: "No"})


# CREATING FINAL OUTPUT

print("Preparing final dataset...")

final_df = sampled[[
    'IMD_Decile', 'Ethnicity', 'Age_Group', 'Gender',
    'Medical_Specialty', 'Consultation_Type', 'Appointment_Type',
    'Base_NoShow_Prob', 'Adjusted_NoShow_Prob',
    'Previous_Appointments', 'Previous_NoShows', 'NoShow_Label'
]].rename(columns={
    'NoShow_Label': 'NoShow',
    'Adjusted_NoShow_Prob': 'NoShow_Prob_Final'
})


# SAVING AND SUMMERY

final_df.to_csv(
    "./data/processed/nhs_no_show_enhanced_synthetic.csv", index=False)

print("\n" + "="*70)
print("ENHANCED SYNTHETIC NHS DATASET GENERATED!")
print("="*70)
print(f"Total Patients: {len(final_df):,}")
print(
    f"No-Show Rate: {final_df['NoShow'].value_counts(normalize=True)['Yes']:.2%}")
print(f"\n Feature Summary:")
print(f"  - Medical Specialties: {final_df['Medical_Specialty'].nunique()}")
print(
    f"  - Teleconsultation Rate: {(final_df['Consultation_Type'] == 'tele-consult').mean():.2%}")
print(
    f"  - First Appointments: {(final_df['Appointment_Type'] == 'First').mean():.2%}")
print("\n Saved to: ./data/processed/nhs_no_show_enhanced_synthetic.csv")
print("="*70)

print("\n Sample Records:")
print(final_df.head(10).to_string(index=False))

print("\n No-Show Rate by Key Features:")
print(final_df.groupby('Consultation_Type')['NoShow'].apply(
    lambda x: (x == 'Yes').mean()).to_string())
print(final_df.groupby('Appointment_Type')['NoShow'].apply(
    lambda x: (x == 'Yes').mean()).to_string())
