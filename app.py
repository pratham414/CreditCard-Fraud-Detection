import streamlit as st
import pandas as pd
import numpy as np
import joblib
from tensorflow.keras.models import load_model
import time

# --- Page Configuration ---
st.set_page_config(
    page_title="SecurePay Transaction Simulator",
    page_icon="🛡️",
    layout="wide"
)

# --- Load Models, Data, and Scalers ---
@st.cache_resource
def load_resources():
    """
    Loads all models, the original dataset, and the scalers for Amount and Time.
    This function is cached to run only once.
    """
    try:
        original_df = pd.read_csv('creditcard.csv')
        models = {
            "LightGBM (Recommended)": joblib.load('models/lgbm_reg_fraud_detector.joblib'),
            "Neural Network": load_model('models/nn_fraud_detector_v2.keras'),
            "Logistic Regression": joblib.load('models/lr_fraud_detector.joblib')
        }
        
        # Create and fit separate scalers for Amount and Time
        from sklearn.preprocessing import RobustScaler
        amount_scaler = RobustScaler().fit(original_df[['Amount']])
        time_scaler = RobustScaler().fit(original_df[['Time']])
        
        # We need the column order from the training data for consistent predictions
        # V1-V28, scaled_amount, scaled_time
        final_feature_order = [f'V{i}' for i in range(1, 29)] + ['scaled_amount', 'scaled_time']
        
        return models, original_df, amount_scaler, time_scaler, final_feature_order
    except FileNotFoundError as e:
        st.error(f"Error loading files: {e}. Please ensure your `models` folder and `creditcard.csv` are in the project directory.")
        return None, None, None, None, None

# Load all necessary resources
models, original_df, amount_scaler, time_scaler, final_feature_order = load_resources()

# --- User Interface ---
if models:
    st.title("🛡️ SecurePay Transaction Simulator")
    st.markdown("Simulate a real-time credit card transaction and see how different machine learning models predict its legitimacy.")

    # Use columns for a cleaner layout
    col1, col2 = st.columns([2, 1])

    with col1:
        st.subheader("Your Order Details")
        
        # Create a mock shopping cart for realism
        item = st.selectbox("Select an item to purchase:", 
                            ["Laptop", "Coffee", "Groceries", "Flight Ticket", "Designer Watch"])
        
        amount_map = {
            "Laptop": 1250.00, "Coffee": 4.50, "Groceries": 85.20,
            "Flight Ticket": 1800.00, "Designer Watch": 3500.00
        }
        merchant_map = {
            "Laptop": "Online Retail", "Coffee": "Restaurant / Dining", "Groceries": "Miscellaneous",
            "Flight Ticket": "Travel", "Designer Watch": "Luxury Goods"
        }
        
        amount = st.number_input("Transaction Amount ($)", value=amount_map[item], format="%.2f")
        merchant_category = merchant_map[item]
        st.write(f"**Merchant Category:** `{merchant_category}`")

    with col2:
        st.subheader("Payment Information")
        with st.form("payment_form"):
            card_number = st.text_input("Card Number", "4242 4242 4242 4242")
            card_name = st.text_input("Cardholder Name", "Jane Doe")
            expiry = st.text_input("Expiry Date (MM/YY)", "12/28")
            cvv = st.text_input("CVV", "123", type="password")

            model_choice = st.selectbox("Choose Fraud Detection Model:", models.keys())
            force_fraud = st.checkbox("Force a Fraudulent Scenario", help="Check this to test the model against a known fraud pattern.")

            submitted = st.form_submit_button("Pay Now")

    # --- Backend and Prediction Logic ---
    if submitted:
        st.markdown("---")
        progress_bar = st.progress(0, text="Sending transaction...")
        
        # --- Step 1: Handle User Inputs ---
        time_feature = time.time() % 172800
        scaled_amount = amount_scaler.transform([[amount]])[0][0]
        scaled_time = time_scaler.transform([[time_feature]])[0][0]
        
        # --- Step 2 & 3: Simulate Scenario and Select Features ---
        scenario = 'FRAUD' if force_fraud else 'NON_FRAUD'
        
        if scenario == 'FRAUD':
            sample_row = original_df[original_df['Class'] == 1].sample(1)
            simulation_type = "a known fraudulent transaction pattern"
        else:
            sample_row = original_df[original_df['Class'] == 0].sample(1)
            simulation_type = "a known legitimate transaction pattern"
        
        v_features = sample_row.loc[:, 'V1':'V28']
        
        progress_bar.progress(33, text="Analyzing transaction patterns...")
        time.sleep(1)

        # --- Step 4: Assemble Final DataFrame for Prediction ---
        prediction_data = {col: v_features[col].values[0] for col in v_features.columns}
        prediction_data['scaled_amount'] = scaled_amount
        prediction_data['scaled_time'] = scaled_time
        prediction_df = pd.DataFrame(prediction_data, index=[0])[final_feature_order]

        # --- Step 5: Make Prediction ---
        selected_model = models[model_choice]
        prediction_prob = 0.0 # Default for models that don't output probability
        
        if model_choice == "Neural Network":
            prediction_prob = selected_model.predict(prediction_df.values)[0][0]
            # Use the tuned threshold for the neural network
            prediction = 1 if prediction_prob > 0.93 else 0 
        else:
            prediction = selected_model.predict(prediction_df)[0]
            # Get probability for display if available
            if hasattr(selected_model, "predict_proba"):
                prediction_prob = selected_model.predict_proba(prediction_df)[0][1]

        progress_bar.progress(66, text="Applying fraud detection model...")
        time.sleep(1)

        # --- Step 6: Display Results ---
        progress_bar.progress(100, text="Finalizing transaction status.")
        time.sleep(0.5)
        progress_bar.empty()

        if prediction == 1:
            st.error(f"""
                ### 🛑 Transaction Flagged
                **Status:** Potential Fraud Detected
                
                Our **{model_choice}** model has flagged this transaction for a security review.
            """)
        else:
            st.success(f"""
                ### ✅ Transaction Approved
                **Status:** Payment Successful
                
                Our **{model_choice}** model has classified this transaction as legitimate.
            """)
        
        with st.expander("🔬 View Analysis Details"):
            st.write(f"- **Simulation Context:** The system simulated this transaction using `{simulation_type}`.")
            st.write(f"- **Model Used:** `{model_choice}`")
            if prediction_prob > 0:
                st.write(f"- **Fraud Probability Score:** `{prediction_prob:.2%}`")
                if model_choice == "Neural Network":
                    st.write(f"  - (A score > 93% is flagged as potential fraud by this model)")

else:
    st.error("Application failed to load. Please check that all model and data files are in the correct directories.")