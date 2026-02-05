#predict
import tensorflow as tf
import pickle
import numpy as np
from tensorflow.keras.preprocessing.sequence import pad_sequences

import warnings
warnings.filterwarnings("ignore")


# Function to load models and tokenizer
def load_models(binary_model_path, multi_model_path, tokenizer_path):
    binary_model = tf.keras.models.load_model(binary_model_path)
    multi_model = tf.keras.models.load_model(multi_model_path)
    
    with open(tokenizer_path, 'rb') as handle:
        tokenizer = pickle.load(handle)
    
    return binary_model, multi_model, tokenizer

# Function to predict SQLi and its type
def predict_query(query, binary_model, multi_model, tokenizer, max_seq_length=100):
    attack_type_mapping = {
        0: "Not SQLi",
        1: "Union-based",
        2: "Error-based",
        3: "Boolean-based",
        4: "Time-based",
        5: "Stacked queries",
        6: "Mass assignment",
        7: "Schema-based",
        8: "Authentication bypass", 
        9: "Tautology"
    }
    
    sequence = tokenizer.texts_to_sequences([query])
    padded_sequence = pad_sequences(sequence, maxlen=max_seq_length)
    
    is_sqli_prob = binary_model.predict(padded_sequence)[0][0]
    is_sqli = is_sqli_prob >= 0.5

    #print(is_sqli_prob)
    #print(is_sqli)
    
    print(f"Query: {query}")
    print(f"SQLi Probability: {is_sqli_prob:.4f}")
    print(f"Is SQLi: {'Yes' if is_sqli else 'No'}")
    
    attack_type = "Not SQLi"
    attack_type_prob = 0.0
    
    if is_sqli:
        attack_type_probs = multi_model.predict(padded_sequence)[0]
        attack_type_id = np.argmax(attack_type_probs)
        attack_type = attack_type_mapping.get(attack_type_id, "Unknown")
        attack_type_prob = attack_type_probs[attack_type_id]
        print(f"Attack Type: {attack_type}")
        print(f"Attack Type Probability: {attack_type_prob:.4f}")
    
    return {
        'query': query,
        'is_sqli': bool(is_sqli),
        'sqli_probability': float(is_sqli_prob),
        'attack_type': attack_type,
        'attack_type_probability': float(attack_type_prob)
    }

# Load models (ensure files are uploaded in Colab before running)
binary_model_path = r'C:\Users\ghora\Desktop\DMT_PRJ_2\For_google_colab\sqli_binary_model.h5'
multi_model_path = 'For_google_colab/sqli_type_model.h5'
tokenizer_path = r'C:\Users\ghora\Desktop\DMT_PRJ_2\For_google_colab\sqli_tokenizer.pickle'

binary_model, multi_model, tokenizer = load_models(binary_model_path, multi_model_path, tokenizer_path)

# User input for query
query = input("Enter an SQL query to analyze: ")
result = predict_query(query, binary_model, multi_model, tokenizer)

print("\nPrediction Result:", result)


