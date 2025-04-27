# import streamlit as st
# import pandas as pd
# import datetime
# from sklearn.ensemble import RandomForestClassifier
# from sklearn.model_selection import train_test_split

# st.set_page_config(page_title="UPI Fraud Detection", page_icon="🔍", layout="centered")

# st.title("🔍 UPI Fraud Detection System")
# st.write("Enter Transaction Details Below:")

# # ---- User Input Form ----
# sender_identity = st.text_input("Sender's Identity")
# sender_upi = st.text_input("Sender's UPI ID")
# sender_phone = st.text_input("Sender's Phone Number")
# transaction_amount = st.number_input("Transaction Amount", min_value=1)
# receiver_upi = st.text_input("Receiver's UPI ID")
# transaction_time = st.date_input("Time of Transaction", value=datetime.date.today())
# location = st.text_input("Location")
# state = st.text_input("State")

# st.write("---")
# st.write("📂 Upload your dataset for smarter prediction (optional)")
# uploaded_file = st.file_uploader("Upload your datasets.csv", type=["csv"])

# model = None

# # ---- Model Training if CSV Uploaded ----
# if uploaded_file is not None:
#     df = pd.read_csv(uploaded_file)

#     # Feature Engineering (Simple)
#     df['Time of Transaction'] = pd.to_datetime(df['Time of Transaction'], errors='coerce')
#     df['Day'] = df['Time of Transaction'].dt.day
#     df['Month'] = df['Time of Transaction'].dt.month
#     df['Year'] = df['Time of Transaction'].dt.year

#     X = df[['Transaction Amount', 'Day', 'Month', 'Year']]
#     y = df['Fraudulent']

#     # Split and Train
#     X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
#     model = RandomForestClassifier()
#     model.fit(X_train, y_train)

# # ---- Prediction Section ----
# if st.button("Predict Fraud"):
#     default_sample = {
#         "Sender's Identity": "Pranjal Bhinge",
#         "Sender's UPI ID": "pranjalbhinge@oksbi",
#         "Sender's Phone Number": "8431212363",
#         "Transaction Amount": 548737,
#         "Receiver's UPI ID": "brownpamela@example.com",
#         "Time of Transaction": "2025-04-10",
#         "Location": "Kolkata",
#         "State": "Telangana",
#         "Fraudulent": 0
#     }

#     # Check for Default Safe Transaction
#     safe = True

#     if (sender_identity.strip().lower() != default_sample["Sender's Identity"].lower() or
#         sender_upi.strip().lower() != default_sample["Sender's UPI ID"].lower() or
#         sender_phone.strip() != default_sample["Sender's Phone Number"] or
#         abs(transaction_amount - default_sample["Transaction Amount"]) > 50000 or
#         location.strip().lower() != default_sample["Location"].lower() or
#         state.strip().lower() != default_sample["State"].lower()):
#         safe = False

#     # If safe by default
#     if safe:
#         st.success("✅ This Transaction looks Safe! (Default Check) (Fraudulent: 0)")
#     else:
#         # If model available, use ML prediction
#         if model is not None:
#             # Prepare user input features
#             input_features = pd.DataFrame({
#                 'Transaction Amount': [transaction_amount],
#                 'Day': [transaction_time.day],
#                 'Month': [transaction_time.month],
#                 'Year': [transaction_time.year]
#             })

#             prediction = model.predict(input_features)[0]
#             if prediction == 0:
#                 st.success("✅ This Transaction looks Safe! (Model Prediction) (Fraudulent: 0)")
#             else:
#                 st.error("🚨 This Transaction is Suspicious! (Model Prediction) (Fraudulent: 1)")
#         else:
#             st.warning("⚠ No Model Trained! Please upload a datasets.csv to enable smart prediction.")

import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import time
import re

# Load the dataset
@st.cache_data
def load_data():
    return pd.read_csv('upi_transaction_data.csv')

# UPI domain extraction function (defined at module level)
def extract_upi_domain(upi_id):
    if pd.isna(upi_id) or upi_id == '':
        return 'unknown'
    
    # Check for legitimate patterns first
    if re.match(r'.*@ok(sbi|hdfc|icici|axis|paytm)', upi_id.lower()):
        return 'legitimate_bank'
    elif re.match(r'.*@(oksbi|okhdfc|okicici|okaxis|okpaytm)', upi_id.lower()):
        return 'legitimate_bank'
    elif re.match(r'^\d+@upi$', upi_id.lower()):
        return 'legitimate_upi'
    elif '@' in upi_id:
        return 'suspicious_domain'
    return 'unknown'

# Phone validation function (defined at module level)
def validate_phone(phone):
    phone_str = str(phone)
    if len(phone_str) != 10:
        return 0  # invalid
    if phone_str.startswith(('6', '7', '8', '9')):
        return 1  # valid Indian number
    return 0  # invalid

# Enhanced preprocessing with better feature engineering
def preprocess_data(df):
    # Apply the UPI domain extraction
    df['Sender_Domain_Type'] = df["Sender's UPI ID"].apply(extract_upi_domain)
    df['Receiver_Domain_Type'] = df["Receiver's UPI ID"].apply(extract_upi_domain)
    
    # Convert categorical features to numerical
    domain_mapping = {
        'legitimate_bank': 0,
        'legitimate_upi': 1,
        'suspicious_domain': 2,
        'unknown': 3
    }
    df['Sender_Domain_Encoded'] = df['Sender_Domain_Type'].map(domain_mapping)
    df['Receiver_Domain_Encoded'] = df['Receiver_Domain_Type'].map(domain_mapping)
    
    # Phone number validation
    df['Phone_Valid'] = df["Sender's Phone Number"].apply(validate_phone)
    
    # Transaction time features (if available)
    if 'Time of Transaction' in df.columns:
        df['Transaction_Date'] = pd.to_datetime(df['Time of Transaction'])
        df['Transaction_Day'] = df['Transaction_Date'].dt.day
        df['Transaction_Hour'] = df['Transaction_Date'].dt.hour
        # Night transactions might be more suspicious
        df['Is_Night'] = ((df['Transaction_Hour'] >= 22) | (df['Transaction_Hour'] <= 6)).astype(int)
    
    # Location encoding
    location_mapping = {loc: idx for idx, loc in enumerate(df['Location'].unique())}
    df['Location_Encoded'] = df['Location'].map(location_mapping)
    
    # State encoding
    state_mapping = {state: idx for idx, state in enumerate(df['State'].unique())}
    df['State_Encoded'] = df['State'].map(state_mapping)
    
    # Amount bins
    df['Amount_Bin'] = pd.cut(df['Transaction Amount'], 
                             bins=[0, 1000, 10000, 50000, 100000, 1000000],
                             labels=[0, 1, 2, 3, 4])
    
    return df

# Train the model with balanced classes
def train_model(df):
    # Features and target
    features = ['Transaction Amount', 'Sender_Domain_Encoded', 'Receiver_Domain_Encoded',
                'Location_Encoded', 'State_Encoded', 'Phone_Valid', 'Amount_Bin']
    
    # Add time features if available
    if 'Is_Night' in df.columns:
        features.append('Is_Night')
    
    X = df[features]
    y = df['Fraudulent']
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    
    # Train Random Forest with class weights
    model = RandomForestClassifier(n_estimators=200, 
                                  class_weight='balanced',
                                  max_depth=10,
                                  random_state=42)
    model.fit(X_train, y_train)
    
    # Predictions
    y_pred = model.predict(X_test)
    
    # Metrics
    accuracy = accuracy_score(y_test, y_pred)
    conf_matrix = confusion_matrix(y_test, y_pred)
    class_report = classification_report(y_test, y_pred, output_dict=True)
    
    return model, accuracy, conf_matrix, class_report

# Main function with improved fraud detection
def main():
    st.title("Enhanced UPI Transaction Fraud Detection System")
    st.write("""
    This application uses machine learning to detect potentially fraudulent UPI transactions 
    with improved accuracy for legitimate transactions.
    """)
    
    # Define locations and states at module level
    locations = ['Mumbai', 'Delhi', 'Kolkata', 'Bangalore', 'Chennai', 'Hyderabad', 'Pune', 'Jaipur', 'Lucknow', 'Ahmedabad']
    states = ['Maharashtra', 'Delhi', 'West Bengal', 'Karnataka', 'Tamil Nadu', 'Telangana', 'Uttar Pradesh', 'Gujarat', 'Rajasthan', 'Punjab']
    
    # Load data
    df = load_data()
    
    # Preprocess data
    df_processed = preprocess_data(df)
    
    # Sidebar for navigation
    st.sidebar.title("Navigation")
    app_mode = st.sidebar.selectbox("Choose the page", ["Home", "Data Exploration", "Model Training", "Fraud Detection"])
    
    if app_mode == "Home":
        st.header("Dataset Overview")
        st.write(f"Total transactions: {len(df)}")
        st.write(f"Fraudulent transactions: {df['Fraudulent'].sum()} ({df['Fraudulent'].mean()*100:.2f}%)")
        
        st.subheader("Sample Data")
        st.dataframe(df.head())
        
        # Show distribution
        fig, ax = plt.subplots()
        sns.countplot(x='Fraudulent', data=df, ax=ax)
        ax.set_title('Fraudulent vs Legitimate Transactions')
        st.pyplot(fig)
        
    elif app_mode == "Data Exploration":
        st.header("Enhanced Data Exploration")
        
        # UPI Domain analysis
        st.subheader("UPI Domain Analysis")
        fig, ax = plt.subplots(1, 2, figsize=(15, 5))
        sns.countplot(x='Sender_Domain_Type', hue='Fraudulent', data=df_processed, ax=ax[0])
        ax[0].set_title('Sender UPI Domain Types')
        ax[0].tick_params(axis='x', rotation=45)
        
        sns.countplot(x='Receiver_Domain_Type', hue='Fraudulent', data=df_processed, ax=ax[1])
        ax[1].set_title('Receiver UPI Domain Types')
        ax[1].tick_params(axis='x', rotation=45)
        st.pyplot(fig)
        
        # Phone number validity
        st.subheader("Phone Number Validity")
        fig, ax = plt.subplots()
        sns.countplot(x='Phone_Valid', hue='Fraudulent', data=df_processed, ax=ax)
        ax.set_title('Phone Number Validity vs Fraud Status')
        ax.set_xticklabels(['Invalid', 'Valid'])
        st.pyplot(fig)
        
    elif app_mode == "Model Training":
        st.header("Enhanced Model Training")
        
        if st.button("Train Improved Model"):
            with st.spinner('Training enhanced model...'):
                model, accuracy, conf_matrix, class_report = train_model(df_processed)
                joblib.dump(model, 'enhanced_upi_fraud_model.pkl')
                
                st.success("Enhanced model trained successfully!")
                
                # Show metrics
                st.subheader("Model Performance")
                st.write(f"Accuracy: {accuracy:.4f}")
                
                # Confusion matrix
                st.write("Confusion Matrix:")
                fig, ax = plt.subplots()
                sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', 
                            xticklabels=['Legitimate', 'Fraudulent'],
                            yticklabels=['Legitimate', 'Fraudulent'], ax=ax)
                ax.set_xlabel('Predicted')
                ax.set_ylabel('Actual')
                st.pyplot(fig)
                
                # Classification report
                st.write("Classification Report:")
                report_df = pd.DataFrame(class_report).transpose()
                st.dataframe(report_df)
    
    elif app_mode == "Fraud Detection":
        st.header("Enhanced Fraud Detection")
        
        try:
            model = joblib.load('enhanced_upi_fraud_model.pkl')
            st.success("Enhanced model loaded successfully!")
        except:
            st.warning("Model not found. Please train the model first.")
            return
        
        # Create form for new transaction
        with st.form("transaction_form"):
            st.subheader("Enter Transaction Details")
            
            col1, col2 = st.columns(2)
            
            with col1:
                sender_name = st.text_input("Sender's Name", "Pranjal Bhinge")
                sender_upi = st.text_input("Sender's UPI ID", "pranjalbhinge@oksbi")
                phone_number = st.text_input("Sender's Phone Number", "8431212363")
                amount = st.number_input("Transaction Amount (₹)", min_value=1, max_value=1000000, value=548737)
                
            with col2:
                receiver_upi = st.text_input("Receiver's UPI ID", "brownpamela@example.com")
                location = st.selectbox("Location", locations, index=locations.index("Kolkata"))
                state = st.selectbox("State", states, index=states.index("Telangana"))
                transaction_date = st.date_input("Transaction Date")
                transaction_time = st.time_input("Transaction Time")
            
            submitted = st.form_submit_button("Check for Fraud")
            
            if submitted:
                # Create a dataframe for the new transaction
                new_trans = pd.DataFrame({
                    "Sender's Identity": [sender_name],
                    "Sender's UPI ID": [sender_upi],
                    "Sender's Phone Number": [phone_number],
                    "Transaction Amount": [amount],
                    "Receiver's UPI ID": [receiver_upi],
                    "Location": [location],
                    "State": [state],
                    "Time of Transaction": [f"{transaction_date} {transaction_time}"],
                    "Fraudulent": [0]  # Dummy value
                })
                
                # Preprocess the new transaction
                new_trans_processed = preprocess_data(new_trans)
                
                # Features for prediction
                features = ['Transaction Amount', 'Sender_Domain_Encoded', 'Receiver_Domain_Encoded',
                            'Location_Encoded', 'State_Encoded', 'Phone_Valid', 'Amount_Bin']
                if 'Is_Night' in new_trans_processed.columns:
                    features.append('Is_Night')
                
                X_new = new_trans_processed[features]
                
                # Make prediction
                prediction = model.predict(X_new)
                proba = model.predict_proba(X_new)
                
                # Show result with more detailed analysis
                st.subheader("Fraud Detection Result")
                
                if prediction[0] == 1:
                    st.error("⚠️ Warning: This transaction is predicted to be FRAUDULENT!")
                    st.write(f"Probability of fraud: {proba[0][1]*100:.2f}%")
                    
                    # Detailed reasons
                    st.write("**Potential risk factors:**")
                    
                    # Check sender UPI
                    sender_domain = extract_upi_domain(sender_upi)
                    if sender_domain != 'legitimate_bank' and sender_domain != 'legitimate_upi':
                        st.write("- ❗ Sender UPI domain appears suspicious")
                    else:
                        st.write("- ✅ Sender UPI domain appears legitimate")
                    
                    # Check receiver UPI
                    receiver_domain = extract_upi_domain(receiver_upi)
                    if receiver_domain == 'suspicious_domain':
                        st.write("- ❗ Receiver UPI domain appears suspicious")
                    else:
                        st.write("- ✅ Receiver UPI domain appears normal")
                    
                    # Check phone
                    phone_valid = validate_phone(phone_number)
                    if not phone_valid:
                        st.write("- ❗ Sender phone number appears invalid")
                    else:
                        st.write("- ✅ Sender phone number appears valid")
                    
                    # Check amount
                    if amount > 50000:
                        st.write(f"- ⚠️ High transaction amount (₹{amount:,})")
                    else:
                        st.write(f"- ✅ Transaction amount (₹{amount:,}) within normal range")
                    
                    # Check time if available
                    if 'Is_Night' in new_trans_processed.columns and new_trans_processed['Is_Night'].iloc[0]:
                        st.write("- ⚠️ Transaction occurred during night hours")
                else:
                    st.success("✅ This transaction appears to be LEGITIMATE")
                    st.write(f"Probability of being legitimate: {proba[0][0]*100:.2f}%")
                    
                    # Show verification points
                    st.write("**Verification points:**")
                    
                    sender_domain = extract_upi_domain(sender_upi)
                    if sender_domain in ['legitimate_bank', 'legitimate_upi']:
                        st.write("- ✅ Sender UPI domain verified")
                    
                    phone_valid = validate_phone(phone_number)
                    if phone_valid:
                        st.write("- ✅ Valid Indian phone number")
                    
                    if amount <= 50000:
                        st.write(f"- ✅ Normal transaction amount (₹{amount:,})")
                
                # Show loading animation
                with st.spinner('Analyzing transaction details...'):
                    time.sleep(2)
                
                # Show feature importance
                st.subheader("Key Factors in Decision")
                feature_importance = pd.DataFrame({
                    'Feature': features,
                    'Importance': model.feature_importances_
                }).sort_values('Importance', ascending=False)
                
                fig, ax = plt.subplots(figsize=(10, 6))
                sns.barplot(x='Importance', y='Feature', data=feature_importance, ax=ax)
                ax.set_title('Most Important Features in Prediction')
                st.pyplot(fig)

if __name__ == "__main__":
    main()