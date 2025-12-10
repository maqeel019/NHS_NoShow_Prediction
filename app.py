import streamlit as st
import joblib
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
import streamlit as st
import shap
import streamlit.components.v1 as components
# -----------------------------
# Page config
# -----------------------------


st.set_page_config(page_title="NHS No-Show Predictor", layout="wide")
st.title("🩺 NHS Appointment No-Show Predictor")
st.markdown(
    "Enter patient details below. Powered by your trained XGBoost model."
)

# -----------------------------
# Load model and categories
# -----------------------------
@st.cache_resource
def load_model_and_categories():
    model = joblib.load('xgb_model.joblib')
    df = pd.read_csv('data/processed/nhs_no_show_enhanced_synthetic.csv')
    df = df.drop_duplicates()
    df = df.drop(['Base_NoShow_Prob', 'NoShow_Prob_Final'], axis=1, errors='ignore')

    df['NoShow'] = (df['NoShow'] == 'Yes').astype(int)

    le_imd = LabelEncoder()
    df['IMD_encoded'] = le_imd.fit_transform(df['IMD_Decile'])

    specialty_freq = df['Medical_Specialty'].value_counts().to_dict()
    df['Specialty_Freq'] = df['Medical_Specialty'].map(specialty_freq)

    df['NoShowRate'] = df['Previous_NoShows'] / df['Previous_Appointments'].replace(0, 1)
    df['NoShowRate'] = df['NoShowRate'].round(3)

    one_hot_cols = ['Ethnicity', 'Age_Group', 'Gender', 'Consultation_Type', 'Appointment_Type']
    df_dummies = pd.get_dummies(df[one_hot_cols], drop_first=True)

    feature_cols = ['IMD_encoded', 'Specialty_Freq', 'Previous_Appointments', 'Previous_NoShows', 'NoShowRate'] + list(df_dummies.columns)

    categories = {
        'imd_decile': sorted(df['IMD_Decile'].unique().tolist()),
        'age_group': sorted(
            df['Age_Group'].unique().tolist(),
            key=lambda x: (int(str(x).split('-')[0].strip()) if '-' in str(x) else float('inf'), str(x))
        ),
        'gender': sorted(df['Gender'].unique().tolist()),
        'ethnicity': sorted(df['Ethnicity'].unique().tolist()),
        'med_specialty': sorted(df['Medical_Specialty'].unique().tolist()),
        'consult_type': sorted(df['Consultation_Type'].unique().tolist()),
        'appt_type': sorted(df['Appointment_Type'].unique().tolist())
    }

    globals_dict = {
        'le_imd': le_imd,
        'specialty_freq': specialty_freq,
        'dummy_cols': list(df_dummies.columns),
        'feature_cols': feature_cols
    }

    return model, categories, globals_dict

# Load resources
model, categories, globals_dict = load_model_and_categories()

# -----------------------------
# Sidebar: Patient Inputs
# -----------------------------
with st.sidebar:
    st.header("📝 Patient Inputs")

    col1, col2 = st.columns(2)
    with col1:
        prev_appts = st.number_input("Previous Appointments", min_value=0, value=1, step=1)
    with col2:
        prev_noshows = st.number_input("Previous No-Shows", min_value=0, value=0, step=1)

    st.subheader("👤 Demographics")
    imd_decile = st.selectbox("IMD Decile (1=most deprived, 10=least)", options=categories['imd_decile'], index=4)
    age_group = st.selectbox("Age Group", options=categories['age_group'], index=2)
    ethnicity = st.selectbox("Ethnicity", options=categories['ethnicity'], index=0)
    gender = st.selectbox("Gender", options=categories['gender'], index=0)

    st.subheader("🏥 Appointment Details")
    med_specialty = st.selectbox("Medical Specialty", options=categories['med_specialty'], index=0)
    consult_type = st.selectbox("Consultation Type", options=categories['consult_type'], index=0)
    appt_type = st.selectbox("Appointment Type", options=categories['appt_type'], index=0)

# -----------------------------
# Predict & Store in Session State
# -----------------------------
if st.button("🔮 Predict No-Show Risk"):
    # Build input DataFrame
    input_data = pd.DataFrame({
        'IMD_Decile': [imd_decile],
        'Ethnicity': [ethnicity],
        'Age_Group': [age_group],
        'Gender': [gender],
        'Medical_Specialty': [med_specialty],
        'Consultation_Type': [consult_type],
        'Appointment_Type': [appt_type],
        'Previous_Appointments': [prev_appts],
        'Previous_NoShows': [prev_noshows]
    })

    # IMD encoding
    le_imd = globals_dict['le_imd']
    input_data['IMD_encoded'] = le_imd.transform(input_data['IMD_Decile'])

    # Specialty frequency, fallback 1 if unseen
    input_data['Specialty_Freq'] = input_data['Medical_Specialty'].map(globals_dict['specialty_freq']).fillna(1)

    # NoShowRate
    input_data['NoShowRate'] = input_data['Previous_NoShows'] / input_data['Previous_Appointments'].replace(0, 1)
    input_data['NoShowRate'] = input_data['NoShowRate'].round(3)

    # One-hot encoding with alignment
    one_hot_cols = ['Ethnicity', 'Age_Group', 'Gender', 'Consultation_Type', 'Appointment_Type']
    input_dummies = pd.get_dummies(input_data[one_hot_cols], drop_first=True)
    for col in globals_dict['dummy_cols']:
        if col not in input_dummies.columns:
            input_dummies[col] = 0
    input_dummies = input_dummies[globals_dict['dummy_cols']]  # reorder

    # Full X
    X_input = pd.concat([
        input_data[['IMD_encoded', 'Specialty_Freq', 'Previous_Appointments', 'Previous_NoShows', 'NoShowRate']],
        input_dummies
    ], axis=1)
    X_input = X_input[globals_dict['feature_cols']]

    # Predict & store in session state
    st.session_state.X_input = X_input
    st.session_state.input_data = input_data
    prob_no_show = model.predict_proba(X_input)[0][1]
    st.session_state.prob_no_show = prob_no_show
    st.session_state.prediction = "High Risk" if prob_no_show > 0.60 else "Low Risk"

# -----------------------------
# Display Prediction & SHAP
# -----------------------------
if 'prediction' in st.session_state:
    st.success(f"**Prediction:** {st.session_state.prediction}")
    st.metric(label="No-Show Probability", value=f"{st.session_state.prob_no_show:.2%}")
    st.write("**Input Summary:**")
    st.write(st.session_state.input_data.T)

    # SHAP toggle button
# SHAP toggle button
    # if st.button("Why? (SHAP Explanation)"):
    #     st.subheader("🧪 SHAP Explanation: Feature Contributions")

    #     # Initialize TreeExplainer for your XGBoost model
    #     explainer = shap.TreeExplainer(model)
        
    #     # Compute SHAP values for the single row
    #     shap_values = explainer.shap_values(st.session_state.X_input)
        
    #     # Force plot (interactive, HTML)
    #     force_plot = shap.force_plot(
    #         explainer.expected_value,        # Base value
    #         shap_values,                     # SHAP values
    #         st.session_state.X_input         # Feature values
    #     )

    #     # Render in Streamlit
    #     components.html(force_plot.html(), height=400, scrolling=True)

    #     # Optional: summary table of top contributors
    #     shap_df = pd.DataFrame({
    #         'Feature': st.session_state.X_input.columns,
    #         'SHAP Value': shap_values[0],  # For first (and only) row
    #         'Value': st.session_state.X_input.iloc[0]
    #     }).sort_values('SHAP Value', key=abs, ascending=False).head(10)

    #     st.dataframe(shap_df, use_container_width=True)
    #     # Show SHAP if toggled
        
    
    if st.session_state.get('show_shap', False):
        explainer = shap.TreeExplainer(model)
        shap_values = explainer.shap_values(st.session_state.X_input.iloc[0:1])

        # Handle scalar vs array expected_value
        expected_value = explainer.expected_value
        if isinstance(expected_value, (list, np.ndarray)):
            expected_value = expected_value[1] if len(expected_value) > 1 else expected_value[0]

        # Handle shap_values being a list (multi-class) vs array (binary)
        if isinstance(shap_values, list):
            shap_values_to_plot = shap_values[1][0]  # class 1, first sample
        else:
            shap_values_to_plot = shap_values[0]  # first (and only) row

        # Force plot
        fig = shap.force_plot(
            expected_value,
            shap_values_to_plot,
            st.session_state.X_input.iloc[0],
            matplotlib=False,
            show=False
        )

        st.subheader("🧪 SHAP Force Plot: Feature Contributions")
        st.components.v1.html(fig.data, height=400)

        shap_df = pd.DataFrame({
            'Feature': st.session_state.X_input.columns,
            'SHAP Value': shap_values_to_plot,
            'Value': st.session_state.X_input.iloc[0]
        }).sort_values('SHAP Value', key=abs, ascending=False).head(10)
        st.dataframe(shap_df, use_container_width=True)
