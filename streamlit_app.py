# UPI Fraud Detection System
# 19-04-2025


# import streamlit as st
# import requests
# import uuid
# import os

# BASE_URL = "http://127.0.0.1:5000"

# st.set_page_config(page_title="UPI Fraud Detection", layout="wide")
# st.sidebar.title("🔐 UPI Fraud Detection")
# page = st.sidebar.radio("Navigate", ["Home", "Register User", "Add UPI", "Process Transaction", "Verify UTR", "File Malware Scan","QR Scanner","fraud"])

# st.title("💳 UPI Fraud Detection System")


# # --- Style Functions ---
# def set_custom_style():
#     st.markdown("""
#         <style>
#             .card {
#                 background-color: #f5f7fa;
#                 border-radius: 15px;
#                 padding: 20px;
#                 margin-bottom: 20px;
#                 box-shadow: 2px 2px 12px rgba(0,0,0,0.1);
#             }
#             .card img {
#                 width: 50%;
#                 border-radius: 10px;
#                 margin-bottom: 10px;
#             }
#             .card-title {
#                 font-size: 22px;
#                 font-weight: 600;
#                 margin-bottom: 10px;
#                 color: #333;
#             }
#             .card-desc {
#                 font-size: 16px;
#                 color: #555;
#             }
#         </style>
#     """, unsafe_allow_html=True)

# def custom_css():
#     st.markdown("""
#         <style>
#             .popup {
#                 background-color: #f9f9f9;
#                 border-left: 6px solid #4CAF50;
#                 padding: 20px;
#                 margin-top: 20px;
#                 border-radius: 10px;
#                 box-shadow: 2px 2px 12px rgba(0,0,0,0.05);
#             }
#             .popup.fraud {
#                 border-left-color: #e53935;
#                 background-color: #ffe5e5;
#             }
#             .utr-title {
#                 font-size: 24px;
#                 font-weight: bold;
#                 color: #333;
#                 margin-bottom: 10px;
#             }
#         </style>
#     """, unsafe_allow_html=True)

# def scan_result_card(status, score=None, details=None):
#     style_map = {
#         "Malicious": ("#e53935", "#fff0f0"),
#         "Suspicious": ("#fbc02d", "#fff9e6"),
#         "Clean": ("#43a047", "#e8f5e9")
#     }
#     border_color, bg_color = style_map.get(status, ("#9e9e9e", "#f5f5f5"))

#     st.markdown(f"""
#         <div style='
#             border-left: 6px solid {border_color};
#             background-color: {bg_color};
#             padding: 20px;
#             border-radius: 10px;
#             margin-top: 20px;
#         '>
#             <h4>🔍 Scan Result: {status}</h4>
#             <p><b>Malware Probability Score:</b> {score if score is not None else "N/A"}</p>
#             <p><b>Details:</b> {details if details else "No further explanation provided."}</p>
#         </div>
#     """, unsafe_allow_html=True)


# # --- Home Page ---
# if page == "Home":
#     set_custom_style()
#     st.header("🚀 Welcome to the UPI Fraud Detection Platform")
#     st.markdown("### Your secure gateway to protect digital transactions in real-time.")

#     col1, col2 = st.columns(2)

#     with col1:
#         st.markdown("""
#             <div class='card'>
#                 <img src='https://cdn-icons-png.flaticon.com/512/1006/1006771.png' />
#                 <div class='card-title'>🔐 Register & Manage UPI IDs</div>
#                 <div class='card-desc'>Easily register users, link UPI IDs, and manage account information securely.</div>
#             </div>
#         """, unsafe_allow_html=True)

#         st.markdown("""
#             <div class='card'>
#                 <img src='https://cdn-icons-png.flaticon.com/512/2641/2641300.png' />
#                 <div class='card-title'>📁 File Threat Scanner</div>
#                 <div class='card-desc'>Upload and analyze PDF/DOCX files to detect ransomware and malicious content.</div>
#             </div>
#         """, unsafe_allow_html=True)

#     with col2:
#         st.markdown("""
#             <div class='card'>
#                 <img src='https://cdn-icons-png.flaticon.com/512/2920/2920222.png' />
#                 <div class='card-title'>🔍 UTR & Transaction Checker</div>
#                 <div class='card-desc'>Verify UTR IDs and track suspicious transactions using machine learning.</div>
#             </div>
#         """, unsafe_allow_html=True)

#         st.markdown("""
#             <div class='card'>
#                 <img src='https://cdn-icons-png.flaticon.com/512/5109/5109306.png' />
#                 <div class='card-title'>📷 QR Code Scanner</div>
#                 <div class='card-desc'>Scan payment QR codes securely and check for potential phishing or tampering.</div>
#             </div>
#         """, unsafe_allow_html=True)

#     st.markdown("""
#         <hr>
#         <h4>Why UPI Fraud Detection Platform?</h4>
#         <p>
#             With the rise in digital payments, UPI frauds have also increased. Our intelligent system empowers users to 
#             detect fraud before it happens. From UPI ID validation to machine learning-based UTR analysis, and file scanning 
#             to QR code verification — we've got your security covered. Stay informed. Stay protected. 
#         </p>
#     """, unsafe_allow_html=True)


# # --- Register User ---
# elif page == "Register User":
#     st.header("📥 Register User")
#     name = st.text_input("Name")
#     phone = st.text_input("Phone Number")
#     upi_id = st.text_input("UPI ID")
#     if st.button("Register"):
#         response = requests.post(f"{BASE_URL}/register", json={"name": name, "phone": phone, "upi_id": upi_id})
#         st.write(response.json())



# elif page == "Add UPI":
#     st.header("➕ Add UPI ID")
#     phone_upi = st.text_input("Phone (To Add UPI)", key="add_upi_phone")
#     new_upi = st.text_input("New UPI ID", key="add_upi_id")

#     if st.button("Add UPI"):
#         response = requests.post(f"{BASE_URL}/add_upi", json={
#             "phone": phone_upi,
#             "upi_id": new_upi
#         })

#         res = response.json()
#         if response.status_code == 200:
#             st.success(res.get("message"))
#         elif response.status_code == 201:
#             st.info(res.get("message"))
#         elif response.status_code == 403:
#             st.error(res.get("error"))
#         else:
#             st.warning(res.get("error"))

# # --- Process Transaction ---
# elif page == "Process Transaction":
#     st.header("💸 Process Transaction")
#     sender_upi = st.text_input("Sender UPI")
#     receiver_upi = st.text_input("Receiver UPI")
#     amount = st.number_input("Amount", min_value=1)
#     bank = st.text_input("Bank Name")
#     if st.button("Send Money"):
#         response = requests.post(f"{BASE_URL}/transaction", json={
#             "sender_upi": sender_upi,
#             "receiver_upi": receiver_upi,
#             "amount": amount,
#             "bank": bank
#         })
#         st.write(response.json())


# # --- Verify UTR ---
# elif page == "Verify UTR":
#     st.header("🔍 Check UTR & Get Transaction Details")
#     custom_css()
#     utr_id = st.text_input("Enter UTR ID")

#     if st.button("Verify UTR") and utr_id:
#         with st.spinner("🔄 Verifying UTR..."):
#             try:
#                 response = requests.get(f"{BASE_URL}/check_utr", params={"utr_id": utr_id})
#                 if response.status_code == 200:
#                     result = response.json()
#                     transaction = result.get("transaction", {})
#                     is_fraud = transaction.get("is_fraud", False)
#                     utr_value = transaction.get("utr_id", utr_id)
#                     amount = transaction.get("amount", "N/A")

#                     if is_fraud:
#                         st.markdown(f"""
#                             <div class='popup fraud'>
#                                 <div class='utr-title'>🚨 Fraudulent Transaction Detected!</div>
#                                 <p><b>UTR ID:</b> {utr_value}<br><b>Amount:</b> ₹{amount}<br>
#                                 <b>Status:</b> <span style='color: red;'>Suspicious</span></p>
#                             </div>
#                         """, unsafe_allow_html=True)
#                         st.snow()
#                     else:
#                         st.markdown(f"""
#                             <div class='popup'>
#                                 <div class='utr-title'>✅ Legitimate Transaction</div>
#                                 <p><b>UTR ID:</b> {utr_value}<br><b>Amount:</b> ₹{amount}<br>
#                                 <b>Status:</b> <span style='color: green;'>Safe</span></p>
#                             </div>
#                         """, unsafe_allow_html=True)
#                         st.balloons()

#                     with st.expander("🔎 View Transaction Details"):
#                         st.json(transaction)

#                     # Fetch user info
#                     for role in ["sender", "receiver"]:
#                         upi = transaction.get(f"{role}_upi")
#                         if upi:
#                             user_response = requests.get(f"{BASE_URL}/user_by_upi", params={"upi_id": upi})
#                             if user_response.status_code == 200:
#                                 with st.expander(f"📥 {role.capitalize()} Info"):
#                                     st.json(user_response.json().get("user", {}))
#                 else:
#                     st.error("❌ Failed to verify UTR.")
#             except Exception as e:
#                 st.error(f"❌ Error: {str(e)}")


# # --- File Malware Scan ---
# elif page == "File Malware Scan":
#     st.title("🛡️ File Malware Scanner")
#     st.markdown("Upload a `.pdf` or `.docx` file to scan for ransomware or malware.")
#     uploaded_file = st.file_uploader("Choose a file", type=["pdf", "docx"])

#     if uploaded_file:
#         os.makedirs("uploads", exist_ok=True)
#         file_id = str(uuid.uuid4())
#         file_path = f"uploads/{file_id}_{uploaded_file.name}"

#         with open(file_path, "wb") as f:
#             f.write(uploaded_file.getbuffer())
#         st.success(f"File '{uploaded_file.name}' uploaded successfully.")

#         with open(file_path, "rb") as f:
#             files = {"file": (uploaded_file.name, f)}
#             response = requests.post(f"{BASE_URL}/scan", files=files)

#         if response.status_code == 200:
#             result = response.json()
#             status = result.get("status", "Unknown")
#             score = result.get("score", "N/A")
#             mime_type = result.get("mime_type", "Unknown")
#             entropy = result.get("entropy", "N/A")
#             filename = result.get("filename", "Unknown")

#             status_color = "🟥" if status == "Malicious" else "🟩"
#             st.markdown(f"### {status_color} Status: **{status}**")
#             st.markdown("#### 📄 Scan Details")
#             st.write(f"**File:** {filename}")
#             st.write(f"**MIME Type:** {mime_type}")
#             st.write(f"**Entropy:** {entropy}")
#             st.write(f"**Prediction Score:** {score}")
#             scan_result_card(status=status, score=score)
#         else:
#             st.error("❌ Failed to scan file. Please try again.")

# # fake payment


# elif page == "QR Scanner":
#     st.header("📷 QR Code Scanner")

#     import cv2
#     import numpy as np
#     from pyzbar.pyzbar import decode
#     from PIL import Image
#     from streamlit_webrtc import webrtc_streamer, VideoTransformerBase
#     import av

#     mode = st.radio("Choose Scan Mode:", ["📤 Upload Image", "🎥 Webcam Scanner"])

#     # -------------------
#     # Upload Image Scanner
#     # -------------------
#     def scan_qr_from_image(img_np):
#         decoded_objects = decode(img_np)
#         if decoded_objects:
#             for obj in decoded_objects:
#                 points = obj.polygon
#                 if len(points) > 4:
#                     hull = cv2.convexHull(np.array([point for point in points], dtype=np.float32))
#                     hull = list(map(tuple, np.squeeze(hull)))
#                 else:
#                     hull = points

#                 for j in range(len(hull)):
#                     cv2.line(img_np, hull[j], hull[(j + 1) % len(hull)], (0, 255, 0), 2)

#                 qr_data = obj.data.decode("utf-8")
#                 cv2.putText(img_np, "QR Detected", (obj.rect.left, obj.rect.top - 10),
#                             cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

#                 return img_np, qr_data
#         return img_np, None

#     if mode == "📤 Upload Image":
#         uploaded_file = st.file_uploader("Upload an image with a QR Code", type=["png", "jpg", "jpeg"])
#         if uploaded_file is not None:
#             image = Image.open(uploaded_file)
#             img_np = np.array(image.convert('RGB'))
#             processed_image, qr_data = scan_qr_from_image(img_np)

#             st.image(processed_image, caption="Scanned Image", use_column_width=True)

#             if qr_data:
#                 st.success(f"✅ QR Code Detected: {qr_data}")
#             else:
#                 st.warning("❌ No QR code detected in the uploaded image.")

#     # -------------------
#     # Webcam Scanner
#     # -------------------
#     elif mode == "🎥 Webcam Scanner":
        
#         st.write("Show your QR code to the webcam. It will auto-detect and display instantly.")

#         class QRLiveScanner(VideoTransformerBase):
#             def transform(self, frame):
#                 img = frame.to_ndarray(format="bgr24")
#                 gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
#                 decoded = decode(gray)

#                 if decoded and "qr_data" not in st.session_state:
#                     obj = decoded[0]
#                     st.session_state.qr_data = obj.data.decode("utf-8")

#                     points = obj.polygon
#                     if len(points) > 4:
#                         hull = cv2.convexHull(np.array([point for point in points], dtype=np.float32))
#                         hull = list(map(tuple, np.squeeze(hull)))
#                     else:
#                         hull = points

#                     for j in range(len(hull)):
#                         cv2.line(img, hull[j], hull[(j + 1) % len(hull)], (0, 255, 0), 2)

#                     cv2.putText(img, "QR Detected", (obj.rect.left, obj.rect.top - 10),
#                                 cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

#                 # Add scanning indicator
#                 cv2.rectangle(img, (0, 0), (200, 30), (0, 0, 0), -1)
#                 cv2.putText(img, "Scanning...", (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

#                 return img

#         webrtc_streamer(key="qr_live", video_transformer_factory=QRLiveScanner)

#         if "qr_data" in st.session_state:
#             st.markdown(f"""
#             <div style="padding: 1rem; background-color: #e8f5e9; border-radius: 10px; border: 1px solid #81c784;">
#                 <h4>✅ QR Code Scanned</h4>
#                 <p><strong>Data:</strong> {st.session_state.qr_data}</p>
#             </div>
#             """, unsafe_allow_html=True)

#             if st.button("🔄 Scan Again"):
#                 del st.session_state.qr_data


# elif page == "fraud":
#     import joblib
#     from streamlit.components.v1 import html
#     # ===== Load Model =====
#     try:
#         model = joblib.load("models/fraud_detection_model.pkl")
#     except Exception as e:
#         st.error(f"Error loading model: {str(e)}")
#         st.stop()

#     # ===== Custom CSS Styling =====
#     st.markdown("""
#         <style>
#         /* Main styling */
#         body {
#             background-color: #eef2f7;
#             overflow-y: auto;
#         }
#         .main {
#             background-color: #f9fafb;
#             padding: 2rem;
#             border-radius: 20px;
#             max-width: 800px;
#             margin: auto;
#             box-shadow: 0 4px 12px rgba(0,0,0,0.1);
#             overflow-y: auto;
#         }
#         h1, h2, h3 {
#             text-align: center;
#             color: #003366;
#             font-family: 'Poppins', sans-serif;
#         }
#         .stButton>button {
#             color: white;
#             background-color: #004488;
#             padding: 0.75rem 1.5rem;
#             border-radius: 10px;
#             font-size: 1rem;
#             width: 100%;
#             transition: all 0.3s;
#         }
#         .stButton>button:hover {
#             background-color: #003366;
#             transform: translateY(-2px);
#             box-shadow: 0 4px 8px rgba(0,0,0,0.2);
#         }
        
#         /* Input styling */
#         .stTextInput>div>input, .stSelectbox>div>div {
#             padding: 0.75rem;
#             border-radius: 10px;
#             transition: all 0.3s;
#         }
#         .stTextInput>div>input:focus, .stSelectbox>div>div:focus {
#             box-shadow: 0 0 0 2px #00448888;
#         }
        
#         /* Transaction details */
#         .transaction-details {
#             background-color: #f0f4f8;
#             padding: 1.5rem;
#             border-radius: 10px;
#             margin-top: 1rem;
#             box-shadow: 0 2px 8px rgba(0,0,0,0.05);
#             transition: all 0.3s;
#         }
#         .transaction-details:hover {
#             transform: translateY(-3px);
#             box-shadow: 0 6px 12px rgba(0,0,0,0.1);
#         }
        
#         /* Animations */
#         @keyframes pulse {
#             0% { transform: scale(1); }
#             50% { transform: scale(1.05); }
#             100% { transform: scale(1); }
#         }
#         @keyframes shake {
#             0%, 100% { transform: translateX(0); }
#             10%, 30%, 50%, 70%, 90% { transform: translateX(-5px); }
#             20%, 40%, 60%, 80% { transform: translateX(5px); }
#         }
#         .fraud-animation {
#             animation: shake 0.5s cubic-bezier(.36,.07,.19,.97) both;
#             color: #ff0000;
#         }
#         .safe-animation {
#             animation: pulse 1s ease-in-out;
#             color: #00aa00;
#         }
        
#         /* Scroll behavior */
#         html {
#             scroll-behavior: smooth;
#         }
#         </style>
#     """, unsafe_allow_html=True)

#     # ===== Streamlit App =====
#     st.title("🚨 UPI Transaction Fraud Detection")
#     st.markdown("### 📝 Enter Transaction Details")

#     with st.form("fraud_form"):
#         col1, col2 = st.columns(2)
#         with col1:
#             sender_upi = st.text_input("Sender's UPI ID", placeholder="name@upi")
#             sender_phone = st.text_input("Sender's Phone Number", placeholder="9876543210")
#         with col2:
#             receiver_upi = st.text_input("Receiver's UPI ID", placeholder="merchant@upi")
#             transaction_amount = st.number_input("Amount (₹)", min_value=1.0, value=1000.0, step=100.0)
        
#         transaction_time = st.time_input("Time of Transaction")
        
#         # Location inputs (hidden from user but used in prediction)
#         location = 'Delhi'  # Default value
#         state = 'Delhi'     # Default value
        
#         submit_button = st.form_submit_button("🚀 Predict Fraud Status")

#     # ===== Prediction Section =====
#     if submit_button:
#         # JavaScript for smooth scrolling to results
#         html("""
#         <script>
#         window.scrollTo({
#             top: document.body.scrollHeight,
#             behavior: 'smooth'
#         });
#         </script>
#         """)
        
#         st.markdown("---")
#         result_container = st.container()
        
#         # Encode default location values
#         location_mapping = {'Delhi': 0, 'Mumbai': 1, 'Bangalore': 2, 
#                            'Hyderabad': 3, 'Chennai': 4, 'Kolkata': 5, 'Other': 6}
#         state_mapping = {'Delhi': 0, 'Maharashtra': 1, 'Karnataka': 2, 
#                         'Telangana': 3, 'Tamil Nadu': 4, 'West Bengal': 5, 'Other': 6}
        
#         location_encoded = location_mapping.get(location, 6)
#         state_encoded = state_mapping.get(state, 6)

#         # Prepare input array [Amount, Location, State]
#         input_data = np.array([[transaction_amount, location_encoded, state_encoded]])

#         try:
#             prediction = model.predict(input_data)[0]
            
#             with result_container:
#                 st.subheader("🔎 Prediction Result")
                
#                 if prediction == 1:
#                     # Fraud animation and styling
#                     st.markdown("""
#                     <div class="fraud-animation">
#                         <h2 style="color: #ff0000; text-align: center;">🚨 FRAUD ALERT!</h2>
#                         <p style="text-align: center; font-size: 1.2rem;">This transaction is predicted as <strong>high risk</strong> for fraud.</p>
#                     </div>
#                     """, unsafe_allow_html=True)
                    
#                     # Danger animation (skull icon that pulses)
#                     st.markdown("""
#                     <div style="text-align: center; font-size: 3rem; animation: pulse 2s infinite;">
#                         ☠️
#                     </div>
#                     """, unsafe_allow_html=True)
                    
#                     st.warning("""
#                     **Recommendation:**  
#                     - Verify transaction details carefully  
#                     - Contact your bank immediately  
#                     - Do not proceed if suspicious
#                     """)
                    
#                 else:
#                     # Safe transaction animation
#                     st.markdown("""
#                     <div class="safe-animation">
#                         <h2 style="color: #00aa00; text-align: center;">✅ TRANSACTION SAFE</h2>
#                         <p style="text-align: center; font-size: 1.2rem;">This transaction appears to be <strong>low risk</strong>.</p>
#                     </div>
#                     """, unsafe_allow_html=True)
                    
#                     # Celebration animation (confetti)
#                     st.markdown("""
#                     <div style="text-align: center; font-size: 3rem;">
#                         🎉
#                     </div>
#                     <script>
#                     // Simple confetti effect
#                     const duration = 2000;
#                     const end = Date.now() + duration;
#                     (function frame() {
#                         if (Date.now() > end) return;
#                         confetti({
#                             particleCount: 5,
#                             angle: 60,
#                             spread: 55,
#                             origin: { x: 0 }
#                         });
#                         confetti({
#                             particleCount: 5,
#                             angle: 120,
#                             spread: 55,
#                             origin: { x: 1 }
#                         });
#                         requestAnimationFrame(frame);
#                     }());
#                     </script>
#                     """, unsafe_allow_html=True)
                
#                 # Enhanced transaction details display
#                 with st.expander("📊 Transaction Summary", expanded=True):
#                     st.markdown(f"""
#                     <div class="transaction-details">
#                         <p><strong>🔹 From:</strong> {sender_upi}</p>
#                         <p><strong>🔹 To:</strong> {receiver_upi}</p>
#                         <p><strong>🔹 Amount:</strong> ₹{transaction_amount:,.2f}</p>
#                         <p><strong>🔹 Time:</strong> {transaction_time.strftime("%I:%M %p")}</p>
#                         <p><strong>🔹 Risk Level:</strong> <span style="color: {'#ff0000' if prediction == 1 else '#00aa00'}">{'High' if prediction == 1 else 'Low'}</span></p>
#                     </div>
#                     """, unsafe_allow_html=True)
                    
#         except Exception as e:
#             st.error(f"Prediction failed: {str(e)}")

#     # Footer
#     st.markdown("---")
#     st.caption("""
#     Built with ❤️ | Real-Time UPI Fraud Detection System  
#     Note: This system uses default location values for predictions
#     """)


import streamlit as st
import requests
import uuid
import os
import numpy as np
import joblib
from streamlit.components.v1 import html
from PIL import Image
import cv2
import numpy as np
from pyzbar.pyzbar import decode
import av
from streamlit_webrtc import webrtc_streamer, VideoTransformerBase

# Base configuration
BASE_URL = "http://127.0.0.1:5000"  # Update with your actual backend URL
MODEL_PATH = "models/fraud_detection_model.pkl"  # Ensure this path is correct

# Initialize session state
if 'qr_data' not in st.session_state:
    st.session_state.qr_data = None

# --- Page Configuration ---
st.set_page_config(
    page_title="UPI Fraud Detection",
    page_icon="💳",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- Navigation ---
st.sidebar.title("🔐 UPI Fraud Detection")
page = st.sidebar.radio("Navigate", [
    "Home", 
    "Register User", 
    "Add UPI", 
    "Process Transaction", 
    "Verify UTR", 
    "File Malware Scan",
    "QR Scanner",
    "Fraud Detection"
])

# --- Style Functions ---
def set_custom_style():
    st.markdown("""
        <style>
            /* Main styles */
            body {
                background-color: #f5f7fa;
                font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            }
            .main {
                background-color: #ffffff;
                padding: 2rem;
                border-radius: 15px;
                box-shadow: 0 4px 6px rgba(0,0,0,0.1);
                margin-bottom: 2rem;
            }
            
            /* Card styles */
            .card {
                background-color: #ffffff;
                border-radius: 12px;
                padding: 20px;
                margin-bottom: 20px;
                box-shadow: 0 2px 10px rgba(0,0,0,0.08);
                transition: transform 0.3s, box-shadow 0.3s;
            }
            .card:hover {
                transform: translateY(-5px);
                box-shadow: 0 6px 15px rgba(0,0,0,0.1);
            }
            .card-title {
                font-size: 1.1rem;
                font-weight: 600;
                margin-bottom: 10px;
                color: #2c3e50;
            }
            .card-desc {
                font-size: 0.9rem;
                color: #7f8c8d;
            }
            
            /* Input styles */
            .stTextInput>div>div>input, 
            .stNumberInput>div>div>input,
            .stSelectbox>div>div>select {
                border: 1px solid #dfe6e9;
                border-radius: 8px;
                padding: 10px;
            }
            
            /* Button styles */
            .stButton>button {
                background-color: #3498db;
                color: white;
                border: none;
                border-radius: 8px;
                padding: 0.75rem 1.5rem;
                font-size: 1rem;
                transition: all 0.3s;
            }
            .stButton>button:hover {
                background-color: #2980b9;
                transform: translateY(-2px);
                box-shadow: 0 4px 8px rgba(0,0,0,0.1);
            }
            
            /* Custom scrollbar */
            ::-webkit-scrollbar {
                width: 8px;
            }
            ::-webkit-scrollbar-track {
                background: #f1f1f1;
            }
            ::-webkit-scrollbar-thumb {
                background: #bdc3c7;
                border-radius: 4px;
            }
            ::-webkit-scrollbar-thumb:hover {
                background: #95a5a6;
            }
            
            /* Animation classes */
            @keyframes shake {
                0%, 100% { transform: translateX(0); }
                10%, 30%, 50%, 70%, 90% { transform: translateX(-5px); }
                20%, 40%, 60%, 80% { transform: translateX(5px); }
            }
            @keyframes pulse {
                0% { transform: scale(1); }
                50% { transform: scale(1.05); }
                100% { transform: scale(1); }
            }
            .fraud-animation {
                animation: shake 0.5s cubic-bezier(.36,.07,.19,.97) both;
            }
            .safe-animation {
                animation: pulse 1s ease-in-out;
            }
            
            /* Smooth scrolling */
            html {
                scroll-behavior: smooth;
            }
        </style>
    """, unsafe_allow_html=True)

def scan_result_card(status, score=None, details=None):
    style_map = {
        "Malicious": ("#e74c3c", "#ffebee"),
        "Suspicious": ("#f39c12", "#fff3e0"),
        "Clean": ("#2ecc71", "#e8f5e9")
    }
    border_color, bg_color = style_map.get(status, ("#95a5a6", "#f5f5f5"))

    st.markdown(f"""
        <div style='
            border-left: 5px solid {border_color};
            background-color: {bg_color};
            padding: 1.5rem;
            border-radius: 10px;
            margin: 1rem 0;
            transition: all 0.3s;
        '>
            <h4 style="margin-top: 0; color: {border_color};">🔍 Scan Result: {status}</h4>
            <p><b>Malware Probability Score:</b> {score if score is not None else "N/A"}</p>
            <p><b>Details:</b> {details if details else "No further explanation provided."}</p>
        </div>
    """, unsafe_allow_html=True)

# --- Home Page ---
if page == "Home":
    set_custom_style()
    st.title("🚀 Welcome to UPI Fraud Detection System")
    st.markdown("""
        <div style='text-align: center; margin-bottom: 2rem;'>
            <p style='font-size: 1.1rem; color: #7f8c8d;'>
                Your comprehensive solution for secure digital transactions and fraud prevention
            </p>
        </div>
    """, unsafe_allow_html=True)

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("""
            <div class='card'>
                <div style='display: flex; justify-content: center; margin-bottom: 1rem;'>
                    <img src='https://cdn-icons-png.flaticon.com/512/1006/1006771.png' width='80'/>
                </div>
                <div class='card-title'>🔐 Register & Manage UPI IDs</div>
                <div class='card-desc'>Securely register users and manage UPI IDs with our encrypted system</div>
            </div>
        """, unsafe_allow_html=True)

        st.markdown("""
            <div class='card'>
                <div style='display: flex; justify-content: center; margin-bottom: 1rem;'>
                    <img src='https://cdn-icons-png.flaticon.com/512/2641/2641300.png' width='80'/>
                </div>
                <div class='card-title'>📁 File Threat Scanner</div>
                <div class='card-desc'>Advanced scanning for PDF/DOCX files to detect malware and ransomware</div>
            </div>
        """, unsafe_allow_html=True)

    with col2:
        st.markdown("""
            <div class='card'>
                <div style='display: flex; justify-content: center; margin-bottom: 1rem;'>
                    <img src='https://cdn-icons-png.flaticon.com/512/2920/2920222.png' width='80'/>
                </div>
                <div class='card-title'>🔍 UTR Verification</div>
                <div class='card-desc'>Real-time verification of UTR IDs using machine learning algorithms</div>
            </div>
        """, unsafe_allow_html=True)

        st.markdown("""
            <div class='card'>
                <div style='display: flex; justify-content: center; margin-bottom: 1rem;'>
                    <img src='https://cdn-icons-png.flaticon.com/512/5109/5109306.png' width='80'/>
                </div>
                <div class='card-title'>📷 QR Code Scanner</div>
                <div class='card-desc'>Secure scanning of payment QR codes with tamper detection</div>
            </div>
        """, unsafe_allow_html=True)

    st.markdown("""
        <div style='margin-top: 2rem; padding: 1.5rem; background-color: #f8f9fa; border-radius: 12px;'>
            <h4 style='color: #2c3e50;'>Why Choose Our System?</h4>
            <p style='color: #7f8c8d;'>
                With the exponential growth in digital payments, UPI fraud has become increasingly sophisticated. 
                Our platform combines machine learning algorithms with real-time monitoring to provide comprehensive 
                protection against various types of financial fraud. From account takeovers to phishing scams, 
                we've got you covered.
            </p>
            <ul style='color: #7f8c8d;'>
                <li>Real-time transaction monitoring</li>
                <li>Advanced anomaly detection</li>
                <li>Secure QR code validation</li>
                <li>Document malware scanning</li>
                <li>User-friendly dashboard</li>
            </ul>
        </div>
    """, unsafe_allow_html=True)

# --- Register User ---
elif page == "Register User":
    set_custom_style()
    st.title("📥 User Registration")
    
    with st.form("register_form"):
        name = st.text_input("Full Name", placeholder="Enter your full name", key="reg_name")
        phone = st.text_input("Phone Number", placeholder="Enter 10-digit mobile number", key="reg_phone")
        upi_id = st.text_input("UPI ID", placeholder="username@bank", key="reg_upi")
        
        if st.form_submit_button("Register User"):
            # Client-side validation
            if not all([name.strip(), phone.strip(), upi_id.strip()]):
                st.warning("Please fill all fields")
            elif not phone.strip().isdigit() or len(phone.strip()) != 10:
                st.warning("Phone must be 10 digits")
            elif "@" not in upi_id.strip():
                st.warning("UPI ID must contain '@'")
            else:
                try:
                    response = requests.post(
                        f"{BASE_URL}/register",
                        json={
                            "name": name.strip(),
                            "phone": phone.strip(),
                            "upi_id": upi_id.strip().lower()
                        },
                        timeout=5
                    )
                    
                    # Debugging output
                    st.write(f"Status Code: {response.status_code}")
                    st.write(f"Response Text: {response.text}")
                    
                    try:
                        response_json = response.json()
                    except ValueError:
                        st.error("Server returned invalid JSON. Is Flask running?")
                        pass
                        
                    if response.status_code == 201:
                        st.success("✅ Registration successful!")
                        st.json(response_json["user"])
                    else:
                        st.error(f"Error: {response_json.get('error', 'Unknown error')}")
                        
                except requests.exceptions.RequestException as e:
                    st.error(f"Connection failed: {str(e)}")
                    st.error(f"Please ensure the Flask server is running at {BASE_URL}")
# --- Add UPI ---
elif page == "Add UPI":
    set_custom_style()
    st.title("➕ Add UPI ID")
    
    with st.form("add_upi_form"):
        phone = st.text_input("Registered Phone Number", placeholder="Enter registered phone number")
        new_upi = st.text_input("New UPI ID", placeholder="new_upi@bank")
        
        if st.form_submit_button("Add UPI ID"):
            if not all([phone, new_upi]):
                st.warning("Please fill all fields")
            else:
                try:
                    response = requests.post(
                        f"{BASE_URL}/add_upi", 
                        json={"phone": phone, "upi_id": new_upi},
                        timeout=5
                    )
                    
                    res = response.json()
                    if response.status_code == 200:
                        st.success(res.get("message", "UPI added successfully"))
                    elif response.status_code == 201:
                        st.info(res.get("message", "UPI updated successfully"))
                    elif response.status_code == 403:
                        st.error(res.get("error", "Operation not allowed"))
                    else:
                        st.warning(res.get("error", "Unknown error occurred"))
                except requests.exceptions.RequestException as e:
                    st.error(f"Connection error: {str(e)}")

# --- Process Transaction ---
elif page == "Process Transaction":
    set_custom_style()
    st.title("💸 Process Transaction")
    
    with st.form("transaction_form"):
        col1, col2 = st.columns(2)
        with col1:
            sender_upi = st.text_input("Sender UPI ID", placeholder="your_upi@bank")
        with col2:
            receiver_upi = st.text_input("Receiver UPI ID", placeholder="recipient_upi@bank")
        
        amount = st.number_input("Amount (₹)", min_value=1, value=100, step=100)
        bank = st.text_input("Bank Name", placeholder="Your bank name")
        
        if st.form_submit_button("Process Transaction"):
            if not all([sender_upi, receiver_upi, amount, bank]):
                st.warning("Please fill all fields")
            else:
                try:
                    with st.spinner("Processing transaction..."):
                        response = requests.post(
                            f"{BASE_URL}/transaction", 
                            json={
                                "sender_upi": sender_upi,
                                "receiver_upi": receiver_upi,
                                "amount": amount,
                                "bank": bank
                            },
                            timeout=10
                        )
                        
                        if response.status_code == 200:
                            result = response.json()
                            st.success("✅ Transaction processed successfully!")
                            with st.expander("View Transaction Details"):
                                st.json(result)
                        else:
                            st.error(f"Transaction failed: {response.json().get('error', 'Unknown error')}")
                except requests.exceptions.RequestException as e:
                    st.error(f"Connection error: {str(e)}")

# --- Verify UTR ---
elif page == "Verify UTR":
    set_custom_style()
    st.title("🔍 Verify UTR")
    
    with st.form("utr_form"):
        utr_id = st.text_input("Enter UTR ID", placeholder="Unique Transaction Reference ID")
        
        if st.form_submit_button("Verify UTR"):
            if not utr_id:
                st.warning("Please enter a UTR ID")
            else:
                try:
                    with st.spinner("Verifying UTR..."):
                        response = requests.get(
                            f"{BASE_URL}/check_utr", 
                            params={"utr_id": utr_id},
                            timeout=5
                        )
                        
                        if response.status_code == 200:
                            result = response.json()
                            transaction = result.get("transaction", {})
                            is_fraud = transaction.get("is_fraud", False)
                            
                            if is_fraud:
                                st.error("""
                                ## 🚨 Fraudulent Transaction Detected!
                                **Recommendation:** Do not proceed with this transaction
                                """)
                                st.snow()
                            else:
                                st.success("""
                                ## ✅ Legitimate Transaction
                                **Status:** This transaction appears to be safe
                                """)
                                st.balloons()
                            
                            with st.expander("Transaction Details"):
                                st.json(transaction)
                            
                            # Show sender/receiver info if available
                            for role in ["sender", "receiver"]:
                                upi = transaction.get(f"{role}_upi")
                                if upi:
                                    try:
                                        user_response = requests.get(
                                            f"{BASE_URL}/user_by_upi", 
                                            params={"upi_id": upi},
                                            timeout=5
                                        )
                                        if user_response.status_code == 200:
                                            with st.expander(f"{role.capitalize()} Information"):
                                                st.json(user_response.json().get("user", {}))
                                    except requests.exceptions.RequestException:
                                        pass
                        else:
                            st.error("Failed to verify UTR. Please try again.")
                except requests.exceptions.RequestException as e:
                    st.error(f"Connection error: {str(e)}")

# --- File Malware Scan ---
elif page == "File Malware Scan":
    set_custom_style()
    st.title("🛡️ File Malware Scanner")
    
    uploaded_file = st.file_uploader(
        "Upload a file for scanning", 
        type=["pdf", "docx", "doc"],
        help="Supported formats: PDF, DOCX"
    )
    
    if uploaded_file is not None:
        try:
            # Create uploads directory if not exists
            os.makedirs("uploads", exist_ok=True)
            file_id = str(uuid.uuid4())
            file_path = f"uploads/{file_id}_{uploaded_file.name}"
            
            # Save the uploaded file
            with open(file_path, "wb") as f:
                f.write(uploaded_file.getbuffer())
            
            st.success(f"File '{uploaded_file.name}' uploaded successfully")
            
            # Perform the scan
            with st.spinner("Scanning file for malware..."):
                with open(file_path, "rb") as f:
                    files = {"file": (uploaded_file.name, f)}
                    response = requests.post(
                        f"{BASE_URL}/scan", 
                        files=files,
                        timeout=15
                    )
                
                if response.status_code == 200:
                    result = response.json()
                    status = result.get("status", "Unknown")
                    score = result.get("score", "N/A")
                    details = result.get("details", "No additional details available")
                    
                    scan_result_card(status, score, details)
                    
                    # Show additional file info
                    with st.expander("File Information"):
                        st.write(f"**Filename:** {uploaded_file.name}")
                        st.write(f"**Type:** {uploaded_file.type}")
                        st.write(f"**Size:** {uploaded_file.size / 1024:.2f} KB")
                else:
                    st.error("Failed to scan file. Please try again.")
            
            # Clean up - remove the uploaded file
            try:
                os.remove(file_path)
            except:
                pass
                
        except Exception as e:
            st.error(f"An error occurred: {str(e)}")

# --- QR Scanner ---
elif page == "QR Scanner":
    set_custom_style()
    st.title("📷 QR Code Scanner")
    
    class QRScanner(VideoTransformerBase):
        def transform(self, frame):
            img = frame.to_ndarray(format="bgr24")
            
            # Convert to grayscale for QR detection
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            decoded = decode(gray)
            
            if decoded:
                # Get the first QR code found
                obj = decoded[0]
                qr_data = obj.data.decode("utf-8")
                
                # Draw bounding box around QR code
                points = obj.polygon
                if len(points) > 4:
                    hull = cv2.convexHull(np.array([point for point in points], dtype=np.float32))
                    hull = list(map(tuple, np.squeeze(hull)))
                else:
                    hull = points
                
                # Draw the polygon
                for j in range(len(hull)):
                    cv2.line(img, hull[j], hull[(j + 1) % len(hull)], (0, 255, 0), 3)
                
                # Put text
                cv2.putText(img, "QR DETECTED", (hull[0][0], hull[0][1] - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
                
                # Store QR data in session state
                if "qr_data" not in st.session_state or st.session_state.qr_data != qr_data:
                    st.session_state.qr_data = qr_data
            
            return img
    
    mode = st.radio("Select Scan Mode:", ["📤 Upload Image", "🎥 Live Camera"])
    
    if mode == "📤 Upload Image":
        uploaded_file = st.file_uploader(
            "Upload an image containing QR code", 
            type=["jpg", "jpeg", "png"]
        )
        
        if uploaded_file is not None:
            try:
                image = Image.open(uploaded_file)
                img_np = np.array(image.convert('RGB'))
                
                # Detect QR codes
                decoded = decode(img_np)
                
                if decoded:
                    qr_data = decoded[0].data.decode("utf-8")
                    st.session_state.qr_data = qr_data
                    
                    # Draw on image
                    points = decoded[0].polygon
                    if len(points) > 4:
                        hull = cv2.convexHull(np.array([point for point in points], dtype=np.float32))
                        hull = list(map(tuple, np.squeeze(hull)))
                    else:
                        hull = points
                    
                    for j in range(len(hull)):
                        cv2.line(img_np, hull[j], hull[(j + 1) % len(hull)], (0, 255, 0), 3)
                    
                    cv2.putText(img_np, "QR DETECTED", (hull[0][0], hull[0][1] - 10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
                    
                    st.image(img_np, caption="Scanned Image", use_column_width=True)
                    st.success(f"QR Code Detected: {qr_data}")
                else:
                    st.warning("No QR code found in the uploaded image")
                    st.image(img_np, caption="Uploaded Image", use_column_width=True)
            except Exception as e:
                st.error(f"Error processing image: {str(e)}")
    
    elif mode == "🎥 Live Camera":
        st.info("Point your camera at a QR code to scan it automatically")
        
        # Start the webcam
        webrtc_ctx = webrtc_streamer(
            key="qr-scanner",
            video_transformer_factory=QRScanner,
            rtc_configuration={"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]},
            media_stream_constraints={"video": True, "audio": False},
        )
        
        if st.session_state.qr_data:
            st.success(f"Scanned QR Code: {st.session_state.qr_data}")
            
            if st.button("Clear Scan"):
                st.session_state.qr_data = None
                st.experimental_rerun()

# --- Fraud Detection ---
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
if page == "Fraud Detection":
    st.header("UPI Transaction Fraud Detection")
    
    # 1. Define constants and helper functions
    locations = ['Mumbai', 'Delhi', 'Kolkata', 'Bangalore', 'Chennai', 
                'Hyderabad', 'Pune', 'Jaipur', 'Lucknow', 'Ahmedabad']
    states = ['Maharashtra', 'Delhi', 'West Bengal', 'Karnataka', 'Tamil Nadu',
              'Telangana', 'Uttar Pradesh', 'Gujarat', 'Rajasthan', 'Punjab']

    @st.cache_data
    def load_data():
        return pd.read_csv('upi_transaction_data.csv')

    def extract_upi_domain(upi_id):
        if pd.isna(upi_id) or upi_id == '':
            return 'unknown'
        patterns = [
            (r'.*@ok(sbi|hdfc|icici|axis|paytm)', 'legitimate_bank'),
            (r'.*@(oksbi|okhdfc|okicici|okaxis|okpaytm)', 'legitimate_bank'),
            (r'^\d+@upi$', 'legitimate_upi')
        ]
        for pattern, domain_type in patterns:
            if re.match(pattern, upi_id.lower()):
                return domain_type
        return 'suspicious_domain' if '@' in upi_id else 'unknown'

    def validate_phone(phone):
        phone_str = str(phone)
        return int(len(phone_str) == 10 and phone_str[0] in '6789')

    # 2. Data preprocessing function
    def preprocess_data(df):
        # Domain type extraction
        df['Sender_Domain_Type'] = df["Sender's UPI ID"].apply(extract_upi_domain)
        df['Receiver_Domain_Type'] = df["Receiver's UPI ID"].apply(extract_upi_domain)
        
        # Encoding categorical features
        domain_mapping = {'legitimate_bank':0, 'legitimate_upi':1, 'suspicious_domain':2, 'unknown':3}
        df['Sender_Domain_Encoded'] = df['Sender_Domain_Type'].map(domain_mapping)
        df['Receiver_Domain_Encoded'] = df['Receiver_Domain_Type'].map(domain_mapping)
        
        # Phone validation
        df['Phone_Valid'] = df["Sender's Phone Number"].apply(validate_phone)
        
        # Time features
        if 'Time of Transaction' in df.columns:
            df['Transaction_Date'] = pd.to_datetime(df['Time of Transaction'])
            df['Is_Night'] = ((df['Transaction_Date'].dt.hour >= 22) | 
                             (df['Transaction_Date'].dt.hour <= 6)).astype(int)
        
        # Location/state encoding
        df['Location_Encoded'] = df['Location'].apply(lambda x: locations.index(x) if x in locations else -1)
        df['State_Encoded'] = df['State'].apply(lambda x: states.index(x) if x in states else -1)
        
        # Amount binning
        bins = [0, 1000, 10000, 50000, 100000, float('inf')]
        df['Amount_Bin'] = pd.cut(df['Transaction Amount'], bins=bins, labels=False)
        
        return df

    # 3. Load data and model
    df = load_data()
    df_processed = preprocess_data(df)
    
    try:
        model = joblib.load('enhanced_upi_fraud_model.pkl')
        model_loaded = True
    except FileNotFoundError:
        st.error("Model not found. Please train the model first.")
        model_loaded = False

    # 4. Fraud detection form
    if model_loaded:
        with st.form("fraud_detection_form"):
            st.subheader("Transaction Details")
            
            col1, col2 = st.columns(2)
            
            with col1:
                sender_name = st.text_input("Sender Name", "Pranjal Bhinge")
                sender_upi = st.text_input("Sender UPI ID", "pranjalbhinge@oksbi")
                phone = st.text_input("Phone Number", "8431212363")
                amount = st.number_input("Amount (₹)", min_value=1, value=548737)
                
            with col2:
                receiver_upi = st.text_input("Receiver UPI ID", "brownpamela@example.com")
                location = st.selectbox("Location", locations, index=2)  # Default Kolkata
                state = st.selectbox("State", states, index=5)  # Default Telangana
                trans_time = st.time_input("Transaction Time")
            
            if st.form_submit_button("Analyze Transaction"):
                # Create transaction record
                new_trans = pd.DataFrame({
                    "Sender's UPI ID": [sender_upi],
                    "Sender's Phone Number": [phone],
                    "Transaction Amount": [amount],
                    "Receiver's UPI ID": [receiver_upi],
                    "Location": [location],
                    "State": [state],
                    "Time of Transaction": [f"2023-01-01 {trans_time}"],  # Dummy date
                    "Fraudulent": [0]  # Placeholder
                })
                
                # Preprocess and predict
                processed = preprocess_data(new_trans)
                features = ['Transaction Amount', 'Sender_Domain_Encoded', 
                           'Receiver_Domain_Encoded', 'Location_Encoded', 
                           'State_Encoded', 'Phone_Valid', 'Amount_Bin']
                
                if 'Is_Night' in processed.columns:
                    features.append('Is_Night')
                
                proba = model.predict_proba(processed[features])[0]
                is_fraud = model.predict(processed[features])[0]
                
                # Display results
                st.subheader("Analysis Results")
                if is_fraud:
                    st.error(f"⚠️ Potential Fraud Detected ({proba[1]*100:.1f}% confidence)")
                    
                    # Detailed risk factors
                    risk_factors = []
                    if extract_upi_domain(sender_upi) not in ['legitimate_bank', 'legitimate_upi']:
                        risk_factors.append("Unverified sender UPI domain")
                    if extract_upi_domain(receiver_upi) == 'suspicious_domain':
                        risk_factors.append("Suspicious receiver domain")
                    if not validate_phone(phone):
                        risk_factors.append("Invalid phone number")
                    if amount > 50000:
                        risk_factors.append("High transaction amount")
                    if processed.get('Is_Night', [0])[0]:
                        risk_factors.append("Night-time transaction")
                    
                    if risk_factors:
                        st.write("**Risk Factors:**")
                        for factor in risk_factors:
                            st.write(f"- {factor}")
                else:
                    st.success(f"✅ Legitimate Transaction ({proba[0]*100:.1f}% confidence)")
                    
                    # Verification points
                    st.write("**Verification Checks Passed:**")
                    if extract_upi_domain(sender_upi) in ['legitimate_bank', 'legitimate_upi']:
                        st.write("- Valid sender UPI domain")
                    if validate_phone(phone):
                        st.write("- Valid Indian phone number")
                    if amount <= 50000:
                        st.write("- Normal transaction amount")
# Load the dataset

    


# elif page == "Fraud Detection":
#     set_custom_style()
#     st.title("🚨 Fraud Detection System")
    
#     # Load the fraud detection model
#     try:
#         model = joblib.load(MODEL_PATH)
#     except Exception as e:
#         st.error(f"Failed to load fraud detection model: {str(e)}")
#         st.stop()
    
#     with st.form("fraud_form"):
#         st.markdown("### Transaction Details")
        
#         col1, col2 = st.columns(2)
#         with col1:
#             sender_upi = st.text_input("Sender UPI ID", placeholder="sender@upi")
#             sender_phone = st.text_input("Sender Phone", placeholder="9876543210")
#         with col2:
#             receiver_upi = st.text_input("Receiver UPI ID", placeholder="receiver@upi")
#             transaction_amount = st.number_input("Amount (₹)", min_value=1.0, value=1000.0, step=100.0)
        
#         transaction_time = st.time_input("Transaction Time")
        
#         # Using default location values (hidden from user)
#         location = 'Delhi'
#         state = 'Delhi'
        
#         if st.form_submit_button("Check for Fraud"):
#             if not all([sender_upi, receiver_upi, transaction_amount]):
#                 st.warning("Please fill all required fields")
#             else:
#                 try:
#                     # Encode location values
#                     location_mapping = {
#                         'Delhi': 0, 'Mumbai': 1, 'Bangalore': 2, 
#                         'Hyderabad': 3, 'Chennai': 4, 'Kolkata': 5, 'Other': 6
#                     }
#                     state_mapping = {
#                         'Delhi': 0, 'Maharashtra': 1, 'Karnataka': 2, 
#                         'Telangana': 3, 'Tamil Nadu': 4, 'West Bengal': 5, 'Other': 6
#                     }
                    
#                     location_encoded = location_mapping.get(location, 6)
#                     state_encoded = state_mapping.get(state, 6)
                    
#                     # Prepare input data
#                     input_data = np.array([[transaction_amount, location_encoded, state_encoded]])
                    
#                     # Make prediction
#                     prediction = model.predict(input_data)[0]
                    
#                     # Scroll to results
#                     html("""
#                     <script>
#                         window.scrollTo({
#                             top: document.body.scrollHeight,
#                             behavior: 'smooth'
#                         });
#                     </script>
#                     """)
                    
#                     st.markdown("---")
#                     st.subheader("Fraud Prediction Result")
                    
#                     if prediction == 1:
#                         # Fraud detected
#                         st.markdown("""
#                         <div class="fraud-animation">
#                             <h2 style="color: #e74c3c; text-align: center;">🚨 FRAUD ALERT</h2>
#                             <p style="text-align: center; font-size: 1.1rem;">
#                                 This transaction is predicted to be <strong>fraudulent</strong> with high confidence.
#                             </p>
#                         </div>
#                         """, unsafe_allow_html=True)
                        
#                         st.error("""
#                         **Recommended Actions:**
#                         - Immediately verify the transaction details
#                         - Contact your bank's fraud department
#                         - Do not proceed with the payment if suspicious
#                         """)
                        
#                         # Danger icon
#                         st.markdown("""
#                         <div style="text-align: center; font-size: 3rem; margin: 1rem 0;">
#                             ⚠️
#                         </div>
#                         """, unsafe_allow_html=True)
#                     else:
#                         # Safe transaction
#                         st.markdown("""
#                         <div class="safe-animation">
#                             <h2 style="color: #2ecc71; text-align: center;">✅ TRANSACTION SAFE</h2>
#                             <p style="text-align: center; font-size: 1.1rem;">
#                                 This transaction appears to be <strong>legitimate</strong>.
#                             </p>
#                         </div>
#                         """, unsafe_allow_html=True)
                        
#                         st.success("""
#                         **Recommendation:**  
#                         You may proceed with this transaction as it shows no signs of fraud.
#                         """)
                        
#                         # Celebration animation
#                         st.balloons()
                    
#                     # Show transaction details
#                     with st.expander("📋 Transaction Summary", expanded=True):
#                         st.markdown(f"""
#                         <div style="
#                             background-color: #f8f9fa;
#                             padding: 1.5rem;
#                             border-radius: 10px;
#                             margin-top: 1rem;
#                         ">
#                             <p><strong>🔹 From:</strong> {sender_upi}</p>
#                             <p><strong>🔹 To:</strong> {receiver_upi}</p>
#                             <p><strong>🔹 Amount:</strong> ₹{transaction_amount:,.2f}</p>
#                             <p><strong>🔹 Time:</strong> {transaction_time.strftime('%I:%M %p')}</p>
#                             <p><strong>🔹 Status:</strong> <span style="color: {'#e74c3c' if prediction == 1 else '#2ecc71'}">
#                                 {'High Risk' if prediction == 1 else 'Low Risk'}
#                             </span></p>
#                         </div>
#                         """, unsafe_allow_html=True)
                        
#                 except Exception as e:
#                     st.error(f"Error during fraud prediction: {str(e)}")

# # --- Footer ---
# st.markdown("""
#     <div style="text-align: center; margin-top: 3rem; color: #7f8c8d; font-size: 0.9rem;">
#         <hr>
#         <p>UPI Fraud Detection System © 2023 | Secure Digital Transactions</p>
#     </div>
# """, unsafe_allow_html=True)