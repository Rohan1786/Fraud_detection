#18-4-2025

# app.py

# from flask import Flask, request, jsonify
# from flask_pymongo import PyMongo
# from werkzeug.utils import secure_filename
# import uuid, datetime, os
# from utils.analyzer import analyze_file  # Your malware detection logic
# import pandas as pd
# from flask_cors import CORS
# app = Flask(__name__)

# # === MongoDB ===
# app.config["MONGO_URI"] = "mongodb://localhost:27017/upi_fraud_detection"
# mongo = PyMongo(app)

# # === Upload Config ===
# UPLOAD_FOLDER = 'uploads'
# ALLOWED_EXTENSIONS = {'pdf', 'docx'}
# app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
# os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# def allowed_file(filename):
#     return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

# # === Register User ===
# @app.route('/register', methods=['POST'])
# def register():
#     data = request.json

#     if not data.get("name") or not data.get("upi_id") or not data.get("phone"):
#         return jsonify({"error": "Name, UPI ID, and Phone are required"}), 400

#     # Check if a user with the same name already exists
#     existing_user = mongo.db.users.find_one({"name": data["name"]})
#     if existing_user:
#         return jsonify({"error": f"User with name '{data['name']}' already exists"}), 409

#     user = {
#         "user_id": str(uuid.uuid4()),
#         "name": data["name"],
#         "phone": data["phone"],
#         "upi_ids": [data["upi_id"]],
#         "created_at": datetime.datetime.utcnow()
#     }

#     mongo.db.users.insert_one(user)

#     # Avoid returning full user object directly for safety
#     return jsonify({
#         "message": "User registered successfully",
#         "user": {
#             "user_id": user["user_id"],
#             "name": user["name"],
#             "phone": user["phone"],
#             "upi_ids": user["upi_ids"]
#         }
#     }), 201

# # === Add UPI ID ===
# # @app.route('/add_upi', methods=['POST'])
# # def add_upi():
# #     data = request.json
# #     user = mongo.db.users.find_one({"phone": data.get("phone")})
# #     if not user:
# #         return jsonify({"error": "User not found"}), 404

# #     mongo.db.users.update_one(
# #         {"phone": data.get("phone")},
# #         {"$addToSet": {"upi_ids": data.get("upi_id")}}
# #     )
# #     return jsonify({"message": "UPI ID added successfully"}), 200


# @app.route('/add_upi', methods=['POST'])
# def add_upi():
#     data = request.json
#     phone = data.get("phone")
#     upi_id = data.get("upi_id")

#     # Check MongoDB for existing user
#     user = mongo.db.users.find_one({"phone": phone})
#     if user:
#         mongo.db.users.update_one(
#             {"phone": phone},
#             {"$addToSet": {"upi_ids": upi_id}}
#         )
#         return jsonify({"message": "UPI ID added to existing user"}), 200

#     # If user not found in DB, check the dataset
#     upi_dataset = pd.read_csv("Datasets/realistic_utr_dataset_with_upi.csv")

#     # Match phone and upi_id in dataset
#     match = upi_dataset[
#         (upi_dataset["mobile_number"].astype(str) == str(phone)) &
#         (upi_dataset["upi_id"] == upi_id)
#     ]

#     if not match.empty:
#         is_fake = int(match.iloc[0]["is_fake"])
#         if is_fake == 1:
#             return jsonify({"error": "UPI ID is marked as fake in dataset"}), 403
#         else:
#             # Create user and add UPI ID if it's safe
#             mongo.db.users.insert_one({
#                 "phone": phone,
#                 "upi_ids": [upi_id]
#             })
#             return jsonify({"message": "User verified via dataset. UPI ID added"}), 201

#     # If no match found at all
#     return jsonify({"error": "User not found in DB or dataset"}), 404

# # === Process Transaction ===
# from datetime import datetime, timezone  # Make sure to import timezone

# @app.route('/transaction', methods=['POST'])
# def transaction():
#     try:
#         data = request.get_json()
        
#         # Validate required fields
#         required_fields = ["sender_upi", "receiver_upi", "amount", "bank"]
#         if not all(field in data for field in required_fields):
#             return jsonify({"error": f"Missing required fields: {', '.join(required_fields)}"}), 400

#         # Check for empty values
#         if not all(data[field] for field in required_fields):
#             return jsonify({"error": "All fields must contain values"}), 400

#         # Validate amount is positive
#         if not isinstance(data["amount"], (int, float)) or data["amount"] <= 0:
#             return jsonify({"error": "Amount must be a positive number"}), 400

#         # Generate transaction record
#         utr_id = str(uuid.uuid4().int)[:16]
#         transaction = {
#             "utr_id": utr_id,
#             "sender_upi": data["sender_upi"].strip().lower(),
#             "receiver_upi": data["receiver_upi"].strip().lower(),
#             "amount": float(data["amount"]),
#             "bank": data["bank"].strip(),
#             "timestamp": datetime.now(timezone.utc),  # Fixed datetime usage
#             "status": "completed",
#             "fraud_label": None
#         }

#         # Insert into database
#         mongo.db.transactions.insert_one(transaction)
        
#         return jsonify({
#             "message": "Transaction successful",
#             "utr_id": utr_id,
#             "details": {
#                 "sender": transaction["sender_upi"],
#                 "receiver": transaction["receiver_upi"],
#                 "amount": transaction["amount"],
#                 "bank": transaction["bank"],
#                 "timestamp": transaction["timestamp"].isoformat()
#             }
#         }), 201

#     except Exception as e:
#         return jsonify({"error": f"Transaction processing failed: {str(e)}"}), 500
# # === Check UTR Validity ===
# @app.route('/check_utr', methods=['GET'])
# def check_utr():
#     utr_id = request.args.get("utr_id")
#     if not utr_id:
#         return jsonify({"error": "UTR ID required"}), 400

#     transaction = mongo.db.transactions.find_one({"utr_id": utr_id})
#     if not transaction:
#         return jsonify({"error": "UTR ID not found"}), 404

#     return jsonify({"transaction": transaction}), 200

# # === Get User by UPI ID ===
# @app.route('/user_by_upi', methods=['GET'])
# def get_user_by_upi():
#     upi_id = request.args.get("upi_id")
#     user = mongo.db.users.find_one({"upi_ids": upi_id})
#     if not user:
#         return jsonify({"error": "User not found"}), 404

#     return jsonify({"user": user}), 200

# # === 🔐 Malware Scan Endpoint ===
# @app.route('/scan', methods=['POST'])
# def scan_file():
#     if 'file' not in request.files:
#         return jsonify({'error': 'No file part in request'}), 400

#     file = request.files['file']
#     if file.filename == '':
#         return jsonify({'error': 'No selected file'}), 400

#     if file and allowed_file(file.filename):
#         filename = secure_filename(file.filename)
#         file_id = str(uuid.uuid4())
#         filepath = os.path.join(app.config['UPLOAD_FOLDER'], f"{file_id}_{filename}")
#         file.save(filepath)

#         result = analyze_file(filepath, filename)
#         return jsonify(result)
#     else:
#         return jsonify({'error': 'Invalid file type'}), 400

# # === Run Server ===
# if __name__ == '__main__':
#     app.run(debug=True)





# accurate database


from flask import Flask, request, jsonify
from flask_pymongo import PyMongo
from werkzeug.utils import secure_filename
import uuid
from datetime import datetime, timezone
import os
from utils.analyzer import analyze_file  # Your malware detection logic
import pandas as pd
from flask_cors import CORS
from bson import json_util
import json

app = Flask(__name__)
CORS(app)  # Enable CORS for all routes

# === Configuration ===
app.config["MONGO_URI"] = "mongodb://localhost:27017/upi_fraud_detection"
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB limit
ALLOWED_EXTENSIONS = {'pdf', 'docx'}

mongo = PyMongo(app)
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

# === Helper Functions ===
def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def parse_json(data):
    return json.loads(json_util.dumps(data))

# === User Management ===
@app.route('/register', methods=['POST'])
def register():
    try:
        data = request.get_json()
        
        # Validate input
        if not all(key in data for key in ["name", "upi_id", "phone"]):
            return jsonify({"error": "Name, UPI ID, and Phone are required"}), 400
        
        if not all(isinstance(data[key], str) and data[key].strip() for key in ["name", "upi_id", "phone"]):
            return jsonify({"error": "All fields must be non-empty strings"}), 400

        # Check for existing user
        if mongo.db.users.find_one({"$or": [{"name": data["name"]}, {"phone": data["phone"]}]}):
            return jsonify({"error": "User with this name or phone already exists"}), 409

        # Create new user
        user = {
            "user_id": str(uuid.uuid4()),
            "name": data["name"].strip(),
            "phone": data["phone"].strip(),
            "upi_ids": [data["upi_id"].strip().lower()],
            "created_at": datetime.now(timezone.utc),
            "last_updated": datetime.now(timezone.utc)
        }

        mongo.db.users.insert_one(user)
        
        # Return safe user data (without internal MongoDB _id)
        return jsonify({
            "message": "User registered successfully",
            "user": {
                "user_id": user["user_id"],
                "name": user["name"],
                "phone": user["phone"],
                "upi_ids": user["upi_ids"]
            }
        }), 201

    except Exception as e:
        return jsonify({"error": f"Registration failed: {str(e)}"}), 500

@app.route('/add_upi', methods=['POST'])
def add_upi():
    try:
        data = request.get_json()
        phone = data.get("phone", "").strip()
        upi_id = data.get("upi_id", "").strip().lower()

        if not phone or not upi_id:
            return jsonify({"error": "Phone and UPI ID are required"}), 400

        # Check MongoDB for existing user
        user = mongo.db.users.find_one({"phone": phone})
        if user:
            # Check if UPI already exists
            if upi_id in user.get("upi_ids", []):
                return jsonify({"error": "UPI ID already exists for this user"}), 409
                
            mongo.db.users.update_one(
                {"phone": phone},
                {
                    "$addToSet": {"upi_ids": upi_id},
                    "$set": {"last_updated": datetime.now(timezone.utc)}
                }
            )
            return jsonify({"message": "UPI ID added to existing user"}), 200

        # If user not found in DB, check the dataset
        try:
            upi_dataset = pd.read_csv("Datasets/realistic_utr_dataset_with_upi.csv")
            match = upi_dataset[
                (upi_dataset["mobile_number"].astype(str) == str(phone)) &
                (upi_dataset["upi_id"] == upi_id)
            ]

            if not match.empty:
                is_fake = int(match.iloc[0]["is_fake"])
                if is_fake == 1:
                    return jsonify({"error": "UPI ID is marked as fake in dataset"}), 403
                else:
                    # Create user and add UPI ID if it's safe
                    mongo.db.users.insert_one({
                        "user_id": str(uuid.uuid4()),
                        "phone": phone,
                        "upi_ids": [upi_id],
                        "created_at": datetime.now(timezone.utc),
                        "last_updated": datetime.now(timezone.utc)
                    })
                    return jsonify({"message": "User verified via dataset. UPI ID added"}), 201

            return jsonify({"error": "User not found in DB or dataset"}), 404

        except Exception as e:
            return jsonify({"error": f"Dataset verification failed: {str(e)}"}), 500

    except Exception as e:
        return jsonify({"error": f"UPI addition failed: {str(e)}"}), 500

# === Transaction Processing ===
@app.route('/transaction', methods=['POST'])
def transaction():
    try:
        data = request.get_json()
        
        # Validate input
        required_fields = ["sender_upi", "receiver_upi", "amount", "bank"]
        if not all(field in data for field in required_fields):
            return jsonify({"error": f"Missing required fields: {', '.join(required_fields)}"}), 400

        # Type and format validation
        try:
            amount = float(data["amount"])
            if amount <= 0:
                raise ValueError("Amount must be positive")
        except (ValueError, TypeError):
            return jsonify({"error": "Amount must be a positive number"}), 400

        sender_upi = data["sender_upi"].strip().lower()
        receiver_upi = data["receiver_upi"].strip().lower()
        bank = data["bank"].strip()

        # Check if sender exists
        sender = mongo.db.users.find_one({"upi_ids": sender_upi})
        if not sender:
            return jsonify({"error": "Sender UPI ID not registered"}), 404

        # Create transaction record
        transaction = {
            "utr_id": str(uuid.uuid4().int)[:16],
            "sender_upi": sender_upi,
            "receiver_upi": receiver_upi,
            "amount": amount,
            "bank": bank,
            "timestamp": datetime.now(timezone.utc),
            "status": "completed",
            "fraud_label": None,
            "sender_user_id": sender["user_id"]
        }

        # Insert transaction
        mongo.db.transactions.insert_one(transaction)
        
        # Update sender's last activity
        mongo.db.users.update_one(
            {"user_id": sender["user_id"]},
            {"$set": {"last_updated": datetime.now(timezone.utc)}}
        )

        return jsonify({
            "message": "Transaction successful",
            "utr_id": transaction["utr_id"],
            "details": {
                "sender": transaction["sender_upi"],
                "receiver": transaction["receiver_upi"],
                "amount": transaction["amount"],
                "bank": transaction["bank"],
                "timestamp": transaction["timestamp"].isoformat()
            }
        }), 201

    except Exception as e:
        return jsonify({"error": f"Transaction processing failed: {str(e)}"}), 500

# === Query Endpoints ===
@app.route('/check_utr', methods=['GET'])
def check_utr():
    try:
        utr_id = request.args.get("utr_id")
        if not utr_id:
            return jsonify({"error": "UTR ID required"}), 400

        transaction = mongo.db.transactions.find_one({"utr_id": utr_id})
        if not transaction:
            return jsonify({"error": "Transaction not found"}), 404

        return jsonify({"transaction": parse_json(transaction)}), 200

    except Exception as e:
        return jsonify({"error": f"UTR check failed: {str(e)}"}), 500

# === File Scanning ===
# @app.route('/scan', methods=['POST'])
# def scan_file():
#     try:
#         if 'file' not in request.files:
#             return jsonify({'error': 'No file part in request'}), 400

#         file = request.files['file']
#         if file.filename == '':
#             return jsonify({'error': 'No selected file'}), 400

#         if not allowed_file(file.filename):
#             return jsonify({'error': 'Invalid file type'}), 400

#         # Secure file handling
#         filename = secure_filename(file.filename)
#         file_id = str(uuid.uuid4())
#         filepath = os.path.join(app.config['UPLOAD_FOLDER'], f"{file_id}_{filename}")
#         file.save(filepath)

#         # Analyze file
#         result = analyze_file(filepath, filename)
        
#         # Clean up
#         try:
#             os.remove(filepath)
#         except:
#             pass

#         return jsonify(result)

#     except Exception as e:
#         return jsonify({'error': f'Scan failed: {str(e)}'}), 500

# === File Scanning ===
@app.route('/scan', methods=['POST']) 
def scan_file():
    try:
        if 'file' not in request.files:
            return jsonify({'error': 'No file part in request'}), 400

        file = request.files['file']
        if file.filename == '':
            return jsonify({'error': 'No selected file'}), 400

        if not allowed_file(file.filename):
            return jsonify({'error': 'Invalid file type'}), 400

        # Secure file handling
        filename = secure_filename(file.filename)
        file_id = str(uuid.uuid4())
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], f"{file_id}_{filename}")
        file.save(filepath)

        # Analyze file for malware
        result = analyze_file(filepath, filename)
        
        # Clean up file after analysis
        try:
            os.remove(filepath)
        except:
            pass

        # Return scan result (malicious or clean)
        if result.get('malicious', False):
            return jsonify({'error': 'File is malicious'}), 403
        
        return jsonify(result)

    except Exception as e:
        return jsonify({'error': f'Scan failed: {str(e)}'}), 500
# === Health Check ===
@app.route('/health', methods=['GET'])
def health_check():
    try:
        # Test database connection
        mongo.db.command('ping')
        return jsonify({"status": "healthy", "database": "connected"}), 200
    except Exception as e:
        return jsonify({"status": "unhealthy", "error": str(e)}), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)