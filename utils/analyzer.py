import os
import magic
import hashlib
import random
import math

def calculate_entropy(file_path):
    with open(file_path, 'rb') as f:
        data = f.read()
        if not data:
            return 0
        byte_counts = [0]*256
        for byte in data:
            byte_counts[byte] += 1
        entropy = 0
        for count in byte_counts:
            if count:
                p = count / len(data)
                entropy -= p * math.log2(p)
        return entropy

def get_file_hash(filepath):
    with open(filepath, "rb") as f:
        return hashlib.sha256(f.read()).hexdigest()

def analyze_file(filepath, filename):
    mime_type = magic.from_file(filepath, mime=True)
    file_size = os.path.getsize(filepath)
    entropy = calculate_entropy(filepath)
    file_hash = get_file_hash(filepath)

    known_hashes = ["abc123...", "fakebadfilehash123456"]
    if file_hash in known_hashes:
        return {
            'status': 'malicious',
            'threat_score': 1.0,
            'method': 'signature',
            'details': f'{filename} matched known ransomware signature.'
        }

    score = round(random.uniform(0, 1), 2)
    status = 'malicious' if score > 0.7 else 'safe'

    return {
        'status': status,
        'threat_score': score,
        'method': 'ml',
        'features': {
            'mime_type': mime_type,
            'file_size': file_size,
            'entropy': round(entropy, 4),
            'hash': file_hash
        }
    }
