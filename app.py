from flask import Flask, render_template, request, jsonify, send_file
import pandas as pd
import numpy as np
from faker import Faker
import json
from scipy import stats
import os
import matplotlib.pyplot as plt
import seaborn as sns
import io
import base64
from datetime import datetime, timedelta
import uuid
import re
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder
import warnings
import matplotlib

matplotlib.use("Agg")  # Use non-interactive backend that doesn't require a display

warnings.filterwarnings("ignore")

app = Flask(__name__)
fake = Faker()

# Directory for saving datasets
SAVE_DIR = "saved_datasets"
os.makedirs(SAVE_DIR, exist_ok=True)


# Sample data for demonstration - replace with your database connection
def get_sample_data():
    # This would be replaced with your actual database query
    data = {
        "name": [
            "John Smith",
            "Jane Doe",
            "John Smith",
            "Robert Johnson",
            "Emily Davis",
        ],
        "age": [32, 28, 45, 36, 41],
        "salary": [75000, 82000, 65000, 92000, 78000],
        "department": ["Engineering", "Marketing", "Engineering", "Finance", "HR"],
        "years_of_service": [5, 3, 8, 6, 4],
        "email": [
            "john.smith@example.com",
            "jane.doe@example.com",
            "john.smith2@example.com",
            "robert.j@example.com",
            "emily.d@example.com",
        ],
        "hire_date": [
            "2018-05-12",
            "2020-03-15",
            "2015-08-22",
            "2017-11-03",
            "2019-06-30",
        ],
    }
    return pd.DataFrame(data)


# Store mapping dictionaries globally
name_mapping = {}
categorical_mapping = {}
email_mapping = {}
date_mapping = {}
dl_models = {}
scalers = {}
encoders = {}


@app.route("/")
def index():
    df = get_sample_data()
    columns = df.columns.tolist()
    return render_template("index.html", columns=columns)


@app.route("/get_data")
def get_data():
    df = get_sample_data()
    return jsonify(
        {"data": df.to_dict(orient="records"), "columns": df.columns.tolist()}
    )


@app.route("/generate_synthetic", methods=["POST"])
def generate_synthetic():
    global name_mapping, categorical_mapping, email_mapping, date_mapping

    # Get original data
    df_original = get_sample_data()

    # Get columns to synthesize
    columns_to_synthesize = request.json.get("columns", [])

    # Get generation method
    generation_method = request.json.get("method", "standard")

    # Create a copy for synthetic data
    df_synthetic = df_original.copy()

    for column in columns_to_synthesize:
        column_data = df_original[column]
        column_name = column.lower()

        # Determine column type and generate appropriate synthetic data
        if pd.api.types.is_numeric_dtype(column_data):
            # For numeric columns, preserve distribution
            synthetic_values = generate_synthetic_numeric(
                column_data, method=generation_method
            )
            df_synthetic[column] = synthetic_values

        elif pd.api.types.is_string_dtype(column_data):
            # Check for different types of string data
            if any(name in column_name for name in ["name", "first", "last"]):
                synthetic_values = generate_synthetic_names(column_data)
                df_synthetic[column] = synthetic_values
            elif "email" in column_name:
                synthetic_values = generate_synthetic_emails(column_data)
                df_synthetic[column] = synthetic_values
            elif any(
                date_term in column_name
                for date_term in ["date", "time", "day", "month", "year"]
            ):
                # Try to parse as date
                try:
                    date_data = pd.to_datetime(column_data)
                    synthetic_values = generate_synthetic_dates(date_data)
                    df_synthetic[column] = synthetic_values
                except:
                    # If not a date, treat as categorical
                    synthetic_values = generate_synthetic_categorical(column_data)
                    df_synthetic[column] = synthetic_values
            else:
                # For categorical columns
                synthetic_values = generate_synthetic_categorical(column_data)
                df_synthetic[column] = synthetic_values

    # Generate visualization if requested
    if request.json.get("visualize", False):
        visualizations = {}
        for column in columns_to_synthesize:
            if pd.api.types.is_numeric_dtype(df_original[column]):
                img_data = generate_distribution_plot(
                    df_original[column], df_synthetic[column], column
                )
                visualizations[column] = img_data
    else:
        visualizations = None

    # Save dataset if requested
    dataset_id = None
    if request.json.get("save", False):
        dataset_id = save_synthetic_dataset(
            df_original, df_synthetic, columns_to_synthesize
        )

    return jsonify(
        {
            "original_data": df_original.to_dict(orient="records"),
            "synthetic_data": df_synthetic.to_dict(orient="records"),
            "columns": df_original.columns.tolist(),
            "visualizations": visualizations,
            "dataset_id": dataset_id,
        }
    )


def generate_synthetic_numeric(column_data, method="standard"):
    if method == "advanced":
        # Try to detect patterns in the data
        if is_sequential(column_data):
            # Generate sequential data with small variations
            return generate_sequential_data(column_data)
        elif is_cyclical(column_data):
            # Generate cyclical data
            return generate_cyclical_data(column_data)

    # Standard method or fallback
    # Check if we have enough samples for normaltest
    if len(column_data) >= 8:
        # Determine if data follows a normal distribution
        _, p_value = stats.normaltest(column_data)

        if p_value > 0.05:  # Likely normal distribution
            mean = column_data.mean()
            std = column_data.std()
            synthetic = np.random.normal(mean, std, len(column_data))
            # Round to same precision as original
            decimals = get_decimal_precision(column_data)
            synthetic = np.round(synthetic, decimals)
        else:
            # Use kernel density estimation for non-normal distributions
            synthetic = generate_using_kde(column_data)
    else:
        # For small datasets, use bootstrap resampling with small random variations
        mean = column_data.mean()
        std = column_data.std() if column_data.std() > 0 else 0.1 * mean

        # Generate synthetic data based on mean and std
        synthetic = np.random.normal(mean, std, len(column_data))

        # Round to same precision as original
        decimals = get_decimal_precision(column_data)
        synthetic = np.round(synthetic, decimals)

    # Ensure data type matches original
    if column_data.dtype == int:
        synthetic = synthetic.astype(int)

    return synthetic


def is_sequential(column_data):
    """Check if data appears to be sequential (like IDs)"""
    if len(column_data) < 3:
        return False

    sorted_data = sorted(column_data)
    diffs = [sorted_data[i + 1] - sorted_data[i] for i in range(len(sorted_data) - 1)]
    avg_diff = sum(diffs) / len(diffs)

    # Check if differences are consistent (allowing some variation)
    consistent_diffs = sum(abs(d - avg_diff) <= 1 for d in diffs)
    return consistent_diffs / len(diffs) > 0.7


def is_cyclical(column_data):
    """Check if data appears to be cyclical"""
    if len(column_data) < 5:
        return False

    # Check for common cyclical patterns (like days of week, months)
    if column_data.max() == 7 and column_data.min() >= 1:  # Days of week
        return True
    if column_data.max() == 12 and column_data.min() >= 1:  # Months
        return True
    if column_data.max() == 24 and column_data.min() >= 0:  # Hours
        return True

    return False


def generate_sequential_data(column_data):
    """Generate sequential data with small variations"""
    sorted_data = sorted(column_data)
    min_val = sorted_data[0]
    max_val = sorted_data[-1]

    # Calculate average step size
    avg_step = (
        (max_val - min_val) / (len(column_data) - 1) if len(column_data) > 1 else 1
    )

    # Generate new sequence with small random variations
    synthetic = [
        min_val + i * avg_step + np.random.normal(0, avg_step * 0.1)
        for i in range(len(column_data))
    ]

    # Ensure data type matches original
    if column_data.dtype == int:
        synthetic = [int(round(x)) for x in synthetic]

    return synthetic


def generate_cyclical_data(column_data):
    """Generate cyclical data preserving the cycle pattern"""
    min_val = column_data.min()
    max_val = column_data.max()

    # Generate new values within the same range
    synthetic = np.random.randint(min_val, max_val + 1, size=len(column_data))

    return synthetic


def get_decimal_precision(column_data):
    # Determine decimal precision in original data
    sample = column_data.dropna().iloc[0] if not column_data.dropna().empty else 0
    return len(str(sample).split(".")[-1]) if "." in str(sample) else 0


def generate_using_kde(column_data):
    # Use kernel density estimation for non-normal distributions
    from sklearn.neighbors import KernelDensity

    # Reshape data for KDE
    data = column_data.values.reshape(-1, 1)

    # Fit KDE model
    kde = KernelDensity(bandwidth=0.5).fit(data)

    # Generate synthetic samples
    synthetic_samples = kde.sample(len(column_data))
    synthetic = synthetic_samples.flatten()

    return synthetic


def generate_synthetic_names(column_data):
    global name_mapping

    # Create a mapping for consistent replacement
    unique_values = column_data.unique()

    # Initialize mapping if not already done for this column
    for value in unique_values:
        if value not in name_mapping:
            # Check if it looks like a full name (contains space)
            if " " in str(value):
                name_mapping[value] = fake.name()
            else:
                # Check if it's likely a first name or last name
                if len(str(value)) < 8:  # Arbitrary threshold
                    name_mapping[value] = fake.first_name()
                else:
                    name_mapping[value] = fake.last_name()

    # Apply mapping to create synthetic values
    return column_data.map(name_mapping)


def generate_synthetic_emails(column_data):
    global email_mapping, name_mapping

    # Create a mapping for consistent replacement
    unique_values = column_data.unique()

    # Initialize mapping if not already done for this column
    for value in unique_values:
        if value not in email_mapping:
            # Extract name part from email if possible
            name_part = value.split("@")[0] if "@" in str(value) else value
            domain_part = value.split("@")[1] if "@" in str(value) else "example.com"

            # Check if we already have a synthetic name for this person
            if name_part in name_mapping:
                synthetic_name = name_mapping[name_part]
                first_name = synthetic_name.split()[0].lower()
                last_name = (
                    synthetic_name.split()[-1].lower()
                    if len(synthetic_name.split()) > 1
                    else ""
                )

                # Create email from synthetic name
                if last_name:
                    email = f"{first_name}.{last_name}@{domain_part}"
                else:
                    email = f"{first_name}@{domain_part}"
            else:
                # Generate a new email
                email = fake.email()

            email_mapping[value] = email

    # Apply mapping to create synthetic values
    return column_data.map(email_mapping)


def generate_synthetic_dates(date_data):
    global date_mapping

    # Create a mapping for consistent replacement
    unique_values = date_data.unique()

    # Calculate date range
    min_date = date_data.min()
    max_date = date_data.max()
    date_range = (max_date - min_date).days

    # Initialize mapping if not already done
    for value in unique_values:
        if value not in date_mapping:
            # Generate a random date within the same range
            random_days = np.random.randint(0, date_range + 1)
            synthetic_date = min_date + timedelta(days=random_days)
            date_mapping[value] = synthetic_date

    # Apply mapping to create synthetic values
    return date_data.map(date_mapping)


def generate_synthetic_categorical(column_data):
    global categorical_mapping

    # Create a mapping for consistent replacement
    unique_values = column_data.unique()

    # Initialize mapping if not already done for this column
    for value in unique_values:
        if value not in categorical_mapping:
            # Generate a similar but different categorical value
            if isinstance(value, str):
                # For string categories, use faker to generate similar type of data
                if any(
                    dept in column_data.name.lower()
                    for dept in ["department", "division", "team"]
                ):
                    categorical_mapping[value] = fake.job()
                elif any(
                    loc in column_data.name.lower()
                    for loc in ["country", "location", "city", "state"]
                ):
                    categorical_mapping[value] = fake.city()
                else:
                    categorical_mapping[value] = fake.word().capitalize()

    # Apply mapping to create synthetic values
    return column_data.map(categorical_mapping)


def generate_distribution_plot(original_data, synthetic_data, column_name):
    """Generate a comparison plot of original vs synthetic data distributions"""
    plt.figure(figsize=(10, 6))

    # Create KDE plot for continuous data
    sns.kdeplot(original_data, label="Original Data", color="blue")
    sns.kdeplot(synthetic_data, label="Synthetic Data", color="red")

    # Add histogram in the background
    plt.hist(original_data, alpha=0.3, bins=15, color="blue", density=True)
    plt.hist(synthetic_data, alpha=0.3, bins=15, color="red", density=True)

    plt.title(f"Distribution Comparison for {column_name}")
    plt.xlabel(column_name)
    plt.ylabel("Density")
    plt.legend()

    # Save plot to a bytes buffer
    buf = io.BytesIO()
    plt.savefig(buf, format="png")
    plt.close()
    buf.seek(0)

    # Convert to base64 for embedding in HTML
    img_str = base64.b64encode(buf.read()).decode("utf-8")
    return img_str


def save_synthetic_dataset(original_df, synthetic_df, modified_columns):
    """Save the synthetic dataset to a CSV file"""
    dataset_id = str(uuid.uuid4())[:8]
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    # Create directory if it doesn't exist
    os.makedirs(SAVE_DIR, exist_ok=True)

    # Save original and synthetic datasets
    filename = f"{SAVE_DIR}/synthetic_data_{dataset_id}_{timestamp}.csv"
    synthetic_df.to_csv(filename, index=False)

    # Save metadata
    metadata = {
        "dataset_id": dataset_id,
        "timestamp": timestamp,
        "modified_columns": modified_columns,
        "original_shape": original_df.shape,
        "synthetic_shape": synthetic_df.shape,
    }

    with open(f"{SAVE_DIR}/metadata_{dataset_id}_{timestamp}.json", "w") as f:
        json.dump(metadata, f)

    return dataset_id


@app.route("/download/<dataset_id>")
def download_dataset(dataset_id):
    """Download a saved synthetic dataset"""
    # Find the file with the matching dataset_id
    files = os.listdir(SAVE_DIR)
    dataset_file = None

    for file in files:
        if file.startswith(f"synthetic_data_{dataset_id}") and file.endswith(".csv"):
            dataset_file = file
            break

    if dataset_file:
        return send_file(
            f"{SAVE_DIR}/{dataset_file}",
            as_attachment=True,
            download_name=f"synthetic_data_{dataset_id}.csv",
            mimetype="text/csv",
        )
    else:
        return jsonify({"error": "Dataset not found"}), 404


@app.route("/saved_datasets")
def list_saved_datasets():
    """List all saved datasets"""
    files = os.listdir(SAVE_DIR)
    metadata_files = [
        f for f in files if f.startswith("metadata_") and f.endswith(".json")
    ]

    datasets = []
    for file in metadata_files:
        with open(f"{SAVE_DIR}/{file}", "r") as f:
            metadata = json.load(f)
            datasets.append(metadata)

    return jsonify({"datasets": datasets})


@app.route("/generate_dl_synthetic", methods=["POST"])
def generate_dl_synthetic():
    global name_mapping, categorical_mapping, email_mapping, date_mapping

    # Get original data
    df_original = get_sample_data()

    # Get columns to synthesize
    columns_to_synthesize = request.json.get("columns", [])

    # Get deep learning model type
    dl_model_type = request.json.get("dl_model", "gan")

    # Create a copy for synthetic data
    df_synthetic = df_original.copy()

    # Prepare data for deep learning
    numeric_columns = [
        col
        for col in columns_to_synthesize
        if pd.api.types.is_numeric_dtype(df_original[col])
    ]

    categorical_columns = [
        col
        for col in columns_to_synthesize
        if pd.api.types.is_string_dtype(df_original[col])
    ]

    # Generate synthetic data using deep learning if we have numeric columns
    if numeric_columns:
        try:
            if dl_model_type == "gan":
                synthetic_numeric = generate_gan_data(df_original[numeric_columns])
            elif dl_model_type == "vae":
                synthetic_numeric = generate_vae_data(df_original[numeric_columns])
            elif dl_model_type == "transformer":
                synthetic_numeric = generate_transformer_data(
                    df_original[numeric_columns]
                )
            else:
                # Fallback to GAN
                synthetic_numeric = generate_gan_data(df_original[numeric_columns])

            # Update synthetic dataframe with generated numeric data
            for i, col in enumerate(numeric_columns):
                df_synthetic[col] = synthetic_numeric[:, i]
        except Exception as e:
            # If deep learning fails, fall back to statistical methods
            print(f"Deep learning generation failed: {e}")
            for col in numeric_columns:
                df_synthetic[col] = generate_synthetic_numeric(df_original[col])

    # Handle categorical columns with traditional methods for now
    for column in categorical_columns:
        column_data = df_original[column]
        column_name = column.lower()

        # Check for different types of string data
        if any(name in column_name for name in ["name", "first", "last"]):
            synthetic_values = generate_synthetic_names(column_data)
            df_synthetic[column] = synthetic_values
        elif "email" in column_name:
            synthetic_values = generate_synthetic_emails(column_data)
            df_synthetic[column] = synthetic_values
        elif any(
            date_term in column_name
            for date_term in ["date", "time", "day", "month", "year"]
        ):
            try:
                date_data = pd.to_datetime(column_data)
                synthetic_values = generate_synthetic_dates(date_data)
                df_synthetic[column] = synthetic_values
            except:
                synthetic_values = generate_synthetic_categorical(column_data)
                df_synthetic[column] = synthetic_values
        else:
            synthetic_values = generate_synthetic_categorical(column_data)
            df_synthetic[column] = synthetic_values

    # Generate visualization if requested
    if request.json.get("visualize", False):
        visualizations = {}
        for column in columns_to_synthesize:
            if pd.api.types.is_numeric_dtype(df_original[column]):
                img_data = generate_distribution_plot(
                    df_original[column], df_synthetic[column], column
                )
                visualizations[column] = img_data
    else:
        visualizations = None

    # Save dataset if requested
    dataset_id = None
    if request.json.get("save", False):
        dataset_id = save_synthetic_dataset(
            df_original, df_synthetic, columns_to_synthesize
        )

    return jsonify(
        {
            "original_data": df_original.to_dict(orient="records"),
            "synthetic_data": df_synthetic.to_dict(orient="records"),
            "columns": df_original.columns.tolist(),
            "visualizations": visualizations,
            "dataset_id": dataset_id,
            "model_type": dl_model_type,
        }
    )


def generate_gan_data(df_numeric):
    """Generate synthetic data using a GAN (Generative Adversarial Network)"""
    # Get column names and data
    columns = df_numeric.columns
    data = df_numeric.values

    # Scale the data
    scaler = MinMaxScaler()
    scaled_data = scaler.fit_transform(data)

    # Define the GAN model
    class Generator(nn.Module):
        def __init__(self, input_dim, output_dim):
            super(Generator, self).__init__()
            self.model = nn.Sequential(
                nn.Linear(input_dim, 128),
                nn.LeakyReLU(0.2),
                nn.Linear(128, 256),
                nn.LeakyReLU(0.2),
                nn.Linear(256, 512),
                nn.LeakyReLU(0.2),
                nn.Linear(512, output_dim),
                nn.Sigmoid(),  # Output between 0 and 1
            )

        def forward(self, x):
            return self.model(x)

    class Discriminator(nn.Module):
        def __init__(self, input_dim):
            super(Discriminator, self).__init__()
            self.model = nn.Sequential(
                nn.Linear(input_dim, 512),
                nn.LeakyReLU(0.2),
                nn.Linear(512, 256),
                nn.LeakyReLU(0.2),
                nn.Linear(256, 128),
                nn.LeakyReLU(0.2),
                nn.Linear(128, 1),
                nn.Sigmoid(),  # Output probability
            )

        def forward(self, x):
            return self.model(x)

    # Set dimensions
    input_dim = 100  # Noise dimension
    output_dim = scaled_data.shape[1]  # Number of features

    # Initialize models
    generator = Generator(input_dim, output_dim)
    discriminator = Discriminator(output_dim)

    # Define loss and optimizers
    criterion = nn.BCELoss()
    g_optimizer = optim.Adam(generator.parameters(), lr=0.0002)
    d_optimizer = optim.Adam(discriminator.parameters(), lr=0.0002)

    # Convert data to PyTorch tensors
    real_data = torch.FloatTensor(scaled_data)

    # Training parameters
    batch_size = min(32, len(scaled_data))
    epochs = 200

    # Create DataLoader
    dataset = TensorDataset(real_data)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    # Training loop
    for epoch in range(epochs):
        for i, (real_batch,) in enumerate(dataloader):
            batch_size = real_batch.size(0)

            # Train Discriminator
            d_optimizer.zero_grad()

            # Real data
            real_labels = torch.ones(batch_size, 1)
            d_real_output = discriminator(real_batch)
            d_real_loss = criterion(d_real_output, real_labels)

            # Fake data
            noise = torch.randn(batch_size, input_dim)
            fake_data = generator(noise)
            fake_labels = torch.zeros(batch_size, 1)
            d_fake_output = discriminator(fake_data.detach())
            d_fake_loss = criterion(d_fake_output, fake_labels)

            # Total discriminator loss
            d_loss = d_real_loss + d_fake_loss
            d_loss.backward()
            d_optimizer.step()

            # Train Generator
            g_optimizer.zero_grad()

            # Generate fake data
            noise = torch.randn(batch_size, input_dim)
            fake_data = generator(noise)

            # Try to fool the discriminator
            g_output = discriminator(fake_data)
            g_loss = criterion(g_output, real_labels)

            g_loss.backward()
            g_optimizer.step()

    # Generate synthetic data
    with torch.no_grad():
        noise = torch.randn(len(data), input_dim)
        synthetic_scaled = generator(noise).numpy()

    # Inverse transform to get back to original scale
    synthetic_data = scaler.inverse_transform(synthetic_scaled)

    return synthetic_data


def generate_vae_data(df_numeric):
    """Generate synthetic data using a VAE (Variational Autoencoder)"""
    # Get column names and data
    columns = df_numeric.columns
    data = df_numeric.values

    # Scale the data
    scaler = MinMaxScaler()
    scaled_data = scaler.fit_transform(data)

    # Set TensorFlow to run eagerly for better thread safety
    tf.config.run_functions_eagerly(True)

    # Define the VAE model using TensorFlow/Keras
    input_dim = scaled_data.shape[1]
    latent_dim = max(2, input_dim // 2)  # Latent space dimension

    # Encoder
    encoder_inputs = keras.Input(shape=(input_dim,))
    x = layers.Dense(128, activation="relu")(encoder_inputs)
    x = layers.Dense(64, activation="relu")(x)

    # Mean and variance for latent space
    z_mean = layers.Dense(latent_dim)(x)
    z_log_var = layers.Dense(latent_dim)(x)

    # Sampling function
    def sampling(args):
        z_mean, z_log_var = args
        batch = tf.shape(z_mean)[0]
        dim = tf.shape(z_mean)[1]
        epsilon = tf.keras.backend.random_normal(shape=(batch, dim))
        return z_mean + tf.exp(0.5 * z_log_var) * epsilon

    z = layers.Lambda(sampling)([z_mean, z_log_var])

    # Decoder
    decoder_inputs = keras.Input(shape=(latent_dim,))
    x = layers.Dense(64, activation="relu")(decoder_inputs)
    x = layers.Dense(128, activation="relu")(x)
    decoder_outputs = layers.Dense(input_dim, activation="sigmoid")(x)

    # Define models
    encoder = keras.Model(encoder_inputs, [z_mean, z_log_var, z], name="encoder")
    decoder = keras.Model(decoder_inputs, decoder_outputs, name="decoder")

    # VAE model
    outputs = decoder(encoder(encoder_inputs)[2])
    vae = keras.Model(encoder_inputs, outputs, name="vae")

    # Add KL divergence loss
    kl_loss = -0.5 * tf.reduce_mean(
        z_log_var - tf.square(z_mean) - tf.exp(z_log_var) + 1
    )
    vae.add_loss(kl_loss)

    # Suppress TensorFlow warnings during training
    import logging

    logging.getLogger("tensorflow").setLevel(logging.ERROR)

    # Compile and train with reduced verbosity and epochs for faster processing
    vae.compile(optimizer="adam", loss="mse")

    # Use a try-except block to handle potential threading issues
    try:
        vae.fit(scaled_data, scaled_data, epochs=50, batch_size=32, verbose=0)
    except Exception as e:
        print(f"VAE training error: {e}")
        # Fallback to a simpler approach if VAE fails
        return generate_synthetic_numeric(pd.Series(data.flatten())).reshape(data.shape)

    # Generate synthetic data
    try:
        z_sample = np.random.normal(size=(len(data), latent_dim))
        synthetic_scaled = decoder.predict(z_sample, verbose=0)

        # Inverse transform to get back to original scale
        synthetic_data = scaler.inverse_transform(synthetic_scaled)

        # Reset TensorFlow eager execution setting
        tf.config.run_functions_eagerly(False)

        return synthetic_data
    except Exception as e:
        print(f"VAE generation error: {e}")
        # Fallback to a simpler approach if VAE fails
        tf.config.run_functions_eagerly(False)
        return generate_synthetic_numeric(pd.Series(data.flatten())).reshape(data.shape)


def generate_transformer_data(df_numeric):
    """Generate synthetic data using a Transformer model"""
    # Get column names and data
    columns = df_numeric.columns
    data = df_numeric.values

    # Scale the data
    scaler = MinMaxScaler()
    scaled_data = scaler.fit_transform(data)

    # Define the Transformer model
    input_dim = scaled_data.shape[1]

    # Simple transformer-based autoencoder
    class TransformerAutoencoder(nn.Module):
        def __init__(self, input_dim, d_model=64, nhead=4, num_layers=2):
            super(TransformerAutoencoder, self).__init__()

            # Input projection
            self.input_proj = nn.Linear(input_dim, d_model)

            # Transformer encoder
            encoder_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead)
            self.transformer_encoder = nn.TransformerEncoder(
                encoder_layer, num_layers=num_layers
            )

            # Output projection
            self.output_proj = nn.Linear(d_model, input_dim)

            # Positional encoding
            self.pos_encoder = nn.Parameter(torch.zeros(1, 1, d_model))

        def forward(self, x):
            # Add batch dimension if needed
            if x.dim() == 2:
                x = x.unsqueeze(1)  # [batch_size, 1, input_dim]

            # Project input to d_model dimensions
            x = self.input_proj(x)  # [batch_size, 1, d_model]

            # Add positional encoding
            x = x + self.pos_encoder

            # Apply transformer encoder
            x = self.transformer_encoder(x)  # [batch_size, 1, d_model]

            # Project back to input dimensions
            x = self.output_proj(x)  # [batch_size, 1, input_dim]

            # Remove sequence dimension
            x = x.squeeze(1)  # [batch_size, input_dim]

            return x

    # Initialize model
    model = TransformerAutoencoder(input_dim)

    # Define loss and optimizer
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    # Convert data to PyTorch tensors
    tensor_data = torch.FloatTensor(scaled_data)

    # Training parameters
    batch_size = min(32, len(scaled_data))
    epochs = 100

    # Create DataLoader
    dataset = TensorDataset(tensor_data, tensor_data)  # Input = Output for autoencoder
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    # Training loop
    for epoch in range(epochs):
        for inputs, targets in dataloader:
            # Forward pass
            outputs = model(inputs)
            loss = criterion(outputs, targets)

            # Backward pass and optimize
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

    # Generate synthetic data
    # We'll use the model to reconstruct random noise
    with torch.no_grad():
        # Generate random noise in the same shape as our data
        noise = torch.randn(len(data), input_dim)
        # Scale noise to be in the same range as our normalized data
        noise = torch.sigmoid(noise * 0.1)
        # Generate synthetic data
        synthetic_scaled = model(noise).numpy()

    # Inverse transform to get back to original scale
    synthetic_data = scaler.inverse_transform(synthetic_scaled)

    return synthetic_data


if __name__ == "__main__":
    app.run(debug=True)
