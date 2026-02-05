import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTE
from collections import Counter
import matplotlib.pyplot as plt
import seaborn as sns
import hashlib

# Set random seed for reproducibility
np.random.seed(42)
tf.random.set_seed(42)

# Function to convert text to numerical features for SMOTE
def text_to_features(texts):
    # Simple feature extraction: hash the text and take first 10 digits
    features = []
    for text in texts:
        # Convert text to a numeric hash and extract first 10 digits as features
        hash_object = hashlib.md5(text.encode())
        hex_dig = hash_object.hexdigest()
        # Convert to a numeric vector with 10 features
        feature_vector = [int(hex_dig[i:i+2], 16) / 255.0 for i in range(0, 20, 2)]
        features.append(feature_vector)
    return np.array(features)

# Function to generate synthetic text samples without using SMOTE
def generate_synthetic_text_samples(texts, n_samples):
    """
    Generate synthetic text samples by combining parts of existing texts.
    This is a simple alternative to SMOTE for text data.
    """
    synthetic_texts = []
    for _ in range(n_samples):
        # Select two random texts to combine
        idx1, idx2 = np.random.choice(len(texts), 2, replace=True)
        text1, text2 = texts[idx1], texts[idx2]

        # Split each text
        split_point1 = len(text1) // 2
        split_point2 = len(text2) // 2

        # Create a new text by combining parts
        if np.random.random() > 0.5:
            new_text = text1[:split_point1] + text2[split_point2:]
        else:
            new_text = text2[:split_point2] + text1[split_point1:]

        synthetic_texts.append(new_text)

    return np.array(synthetic_texts)

# Function to load and preprocess the dataset
def preprocess_dataset(file_path, max_seq_length=100):
    # Load the dataset
    print(f"Loading dataset from {file_path}...")
    df = pd.read_csv(file_path)

    # Display basic info about the dataset
    print("Original dataset shape:", df.shape)
    print("Label distribution:")
    print(df['Label'].value_counts())
    print("SQLi Type distribution:")
    print(df['SQLi_Type'].value_counts())

    # Step 1: Balance each attack type to have 500 samples
    balanced_attack_df = pd.DataFrame()
    attack_types = df[df['Label'] == 1]['SQLi_Type'].unique()

    for attack_type in attack_types:
        attack_samples = df[(df['Label'] == 1) & (df['SQLi_Type'] == attack_type)]

        if len(attack_samples) > 500:
            # Downsample if we have more than 500 samples
            attack_samples = attack_samples.sample(500, random_state=42)
        elif len(attack_samples) < 500:
            # If we have fewer than 500 samples, generate synthetic samples
            print(f"Attack type {attack_type} has {len(attack_samples)} samples. Generating {500-len(attack_samples)} more.")

            # Use our custom text generation function instead of SMOTE
            base_queries = attack_samples['Query'].values
            n_to_generate = 500 - len(attack_samples)

            # Generate synthetic queries
            synthetic_queries = generate_synthetic_text_samples(base_queries, n_to_generate)

            # Create a new dataframe with the synthetic samples
            synthetic_samples = pd.DataFrame({
                'Query': synthetic_queries,
                'Label': 1,
                'SQLi_Type': attack_type
            })

            # Combine original and synthetic samples
            attack_samples = pd.concat([attack_samples, synthetic_samples])

            # Ensure we have exactly 500 samples
            if len(attack_samples) > 500:
                attack_samples = attack_samples.sample(500, random_state=42)

        balanced_attack_df = pd.concat([balanced_attack_df, attack_samples])

    # Calculate how many legitimate queries (Label=0) we need
    total_attack_samples = len(balanced_attack_df)
    print(f"Total attack samples after balancing: {total_attack_samples}")

    # Get legitimate samples
    legitimate_samples = df[df['Label'] == 0]

    if len(legitimate_samples) > total_attack_samples:
        # Downsample legitimate samples to match attack samples
        legitimate_samples = legitimate_samples.sample(total_attack_samples, random_state=42)
    elif len(legitimate_samples) < total_attack_samples:
        # Generate synthetic legitimate queries
        print(f"Legitimate samples: {len(legitimate_samples)}. Generating {total_attack_samples-len(legitimate_samples)} more.")

        # Calculate how many more we need
        n_to_generate = total_attack_samples - len(legitimate_samples)

        # Generate synthetic queries
        base_queries = legitimate_samples['Query'].values
        synthetic_queries = generate_synthetic_text_samples(base_queries, n_to_generate)

        # Create DataFrame with synthetic samples
        synthetic_samples = pd.DataFrame({
            'Query': synthetic_queries,
            'Label': 0,
            'SQLi_Type': 0
        })

        # Combine original and synthetic samples
        legitimate_samples = pd.concat([legitimate_samples, synthetic_samples])

    # Combine attack and legitimate samples
    balanced_df = pd.concat([balanced_attack_df, legitimate_samples])

    # Display info about the balanced dataset
    print("\nBalanced dataset shape:", balanced_df.shape)
    print("Label distribution:")
    print(balanced_df['Label'].value_counts())
    print("SQLi Type distribution:")
    print(balanced_df['SQLi_Type'].value_counts())

    # Tokenize queries
    tokenizer = Tokenizer(char_level=True)
    tokenizer.fit_on_texts(balanced_df['Query'])

    # Convert queries to sequences
    X_sequences = tokenizer.texts_to_sequences(balanced_df['Query'])
    X_padded = pad_sequences(X_sequences, maxlen=max_seq_length)

    # Prepare labels
    y_binary = balanced_df['Label'].values
    y_multi = balanced_df['SQLi_Type'].values

    # Split the dataset
    X_train, X_test, y_binary_train, y_binary_test, y_multi_train, y_multi_test = train_test_split(
        X_padded, y_binary, y_multi, test_size=0.2, random_state=42, stratify=y_binary
    )

    return (X_train, X_test, y_binary_train, y_binary_test, y_multi_train, y_multi_test,
            tokenizer, max_seq_length)

# Function to build and train the LSTM model
def build_and_train_model(X_train, y_binary_train, y_multi_train,
                          X_test, y_binary_test, y_multi_test,
                          vocab_size, max_seq_length,
                          embedding_dim=32, epochs=10, batch_size=32):

    # Binary classification model (SQLi or not)
    binary_model = Sequential([
        Embedding(vocab_size, embedding_dim, input_length=max_seq_length),
        LSTM(64, return_sequences=True),
        LSTM(32),
        Dense(16, activation='relu'),
        Dropout(0.3),
        Dense(1, activation='sigmoid')
    ])

    binary_model.compile(
        optimizer='adam',
        loss='binary_crossentropy',
        metrics=['accuracy']
    )

    print("Binary model summary:")
    binary_model.summary()

    # Multi-class model (SQLi type classification)
    num_classes = len(np.unique(y_multi_train))
    multi_model = Sequential([
        Embedding(vocab_size, embedding_dim, input_length=max_seq_length),
        LSTM(64, return_sequences=True),
        LSTM(32),
        Dense(16, activation='relu'),
        Dropout(0.3),
        Dense(num_classes, activation='softmax')
    ])

    multi_model.compile(
        optimizer='adam',
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )

    print("\nMulti-class model summary:")
    multi_model.summary()

    # Define early stopping callback
    early_stopping = EarlyStopping(
        monitor='val_loss',
        patience=3,
        restore_best_weights=True
    )

    # Train binary model
    print("\nTraining binary classification model...")
    binary_history = binary_model.fit(
        X_train, y_binary_train,
        validation_data=(X_test, y_binary_test),
        epochs=epochs,
        batch_size=batch_size,
        callbacks=[early_stopping]
    )

    # Train multi-class model
    print("\nTraining multi-class classification model...")
    # Filter data to only include SQLi examples for type classification
    X_train_sqli = X_train[y_binary_train == 1]
    y_multi_train_sqli = y_multi_train[y_binary_train == 1]
    X_test_sqli = X_test[y_binary_test == 1]
    y_multi_test_sqli = y_multi_test[y_binary_test == 1]

    # Check if we have more than one class in the filtered data
    unique_classes = np.unique(y_multi_train_sqli)
    if len(unique_classes) <= 1:
        print("WARNING: Not enough classes for multi-class training. Skipping multi-class model training.")
        # Create a dummy model that always predicts the most common class
        most_common_class = int(unique_classes[0]) if len(unique_classes) > 0 else 0

        def predict_dummy(x):
            return np.array([[1.0 if i == most_common_class else 0.0 for i in range(num_classes)]
                             for _ in range(len(x))])

        multi_model.predict = predict_dummy
        multi_history = None
    else:
        multi_history = multi_model.fit(
            X_train_sqli, y_multi_train_sqli,
            validation_data=(X_test_sqli, y_multi_test_sqli),
            epochs=epochs,
            batch_size=batch_size,
            callbacks=[early_stopping]
        )

    # Evaluate binary model
    binary_loss, binary_accuracy = binary_model.evaluate(X_test, y_binary_test)
    print(f"\nBinary model - Test accuracy: {binary_accuracy:.4f}")

    # Evaluate multi-class model if we trained it
    if multi_history is not None and len(X_test_sqli) > 0:
        multi_loss, multi_accuracy = multi_model.evaluate(X_test_sqli, y_multi_test_sqli)
        print(f"Multi-class model - Test accuracy: {multi_accuracy:.4f}")

    # Plot training history
    plt.figure(figsize=(12, 5))

    plt.subplot(1, 2, 1)
    plt.plot(binary_history.history['accuracy'])
    plt.plot(binary_history.history['val_accuracy'])
    plt.title('Binary Model Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend(['Train', 'Validation'], loc='lower right')

    if multi_history is not None:
        plt.subplot(1, 2, 2)
        plt.plot(multi_history.history['accuracy'])
        plt.plot(multi_history.history['val_accuracy'])
        plt.title('Multi-class Model Accuracy')
        plt.xlabel('Epoch')
        plt.ylabel('Accuracy')
        plt.legend(['Train', 'Validation'], loc='lower right')

    plt.tight_layout()
    plt.savefig('model_training_history.png')
    plt.close()

    return binary_model, multi_model

# Function to predict SQLi for a new query
def predict_sqli(query, binary_model, multi_model, tokenizer, max_seq_length, attack_type_mapping):
    # Preprocess the query
    sequence = tokenizer.texts_to_sequences([query])
    padded_sequence = pad_sequences(sequence, maxlen=max_seq_length)

    # Predict if it's an SQLi attack
    is_sqli_prob = binary_model.predict(padded_sequence)[0][0]
    is_sqli = is_sqli_prob >= 0.5

    result = {
        'query': query,
        'is_sqli': bool(is_sqli),
        'sqli_probability': float(is_sqli_prob)
    }

    # If it's predicted as SQLi, predict the attack type
    if is_sqli:
        attack_type_probs = multi_model.predict(padded_sequence)[0]
        attack_type_id = np.argmax(attack_type_probs)
        attack_type = attack_type_mapping.get(attack_type_id, "Unknown")

        result['attack_type'] = attack_type
        result['attack_type_id'] = int(attack_type_id)
        result['attack_type_probability'] = float(attack_type_probs[attack_type_id])

    return result

# Main function
def main():
    # Define the file path
    file_path = r"C:\Users\ghora\Desktop\DMT_PRJ_2\Verified_SQL_Dataset_with_Types.csv"

    # Define attack type mapping (based on your dataset)
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

    try:
        # Load and preprocess the dataset
        preprocessed_data = preprocess_dataset(file_path)
        X_train, X_test, y_binary_train, y_binary_test, y_multi_train, y_multi_test, tokenizer, max_seq_length = preprocessed_data

        # Build and train the models
        vocab_size = len(tokenizer.word_index) + 1
        binary_model, multi_model = build_and_train_model(
            X_train, y_binary_train, y_multi_train,
            X_test, y_binary_test, y_multi_test,
            vocab_size, max_seq_length
        )

        # Save the models
        binary_model.save("sqli_binary_model.h5")
        multi_model.save("sqli_type_model.h5")

        # Save the tokenizer
        import pickle
        with open("sqli_tokenizer.pickle", "wb") as handle:
            pickle.dump(tokenizer, handle, protocol=pickle.HIGHEST_PROTOCOL)

        print("\nModels and tokenizer saved successfully.")

        # Example predictions
        test_queries = [
            "SELECT * FROM users",
            "SELECT * FROM users WHERE username='admin' OR 1=1",
            "1'; DROP TABLE users; --",
            "SELECT * FROM products WHERE category='Electronics'"
        ]

        print("\nExample predictions:")
        for query in test_queries:
            result = predict_sqli(query, binary_model, multi_model, tokenizer, max_seq_length, attack_type_mapping)
            print(f"\nQuery: {result['query']}")
            print(f"Is SQLi: {result['is_sqli']} (Probability: {result['sqli_probability']:.4f})")
            if result['is_sqli']:
                print(f"Attack Type: {result['attack_type']} (Probability: {result['attack_type_probability']:.4f})")

    except Exception as e:
        print(f"An error occurred: {str(e)}")

if __name__ == "__main__":
    main()