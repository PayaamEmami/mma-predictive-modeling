import torch
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
import pickle
import boto3
import os
from config import HYPERPARAMETERS


def train_model(name, model, X_train, y_train, device, verbose=True):
    """
    Train a model.

    Args:
        name: Name of the model
        model: Model instance to train (PyTorch or sklearn)
        X_train: Training features
        y_train: Training labels
        device: PyTorch device to use
        verbose: Whether to print training progress

    Returns:
        The trained model
    """
    if verbose:
        print(f"\nTraining {name}...")

    params = HYPERPARAMETERS[name]

    # Determine which type of model to train (PyTorch or sklearn)
    if isinstance(model, torch.nn.Module):
        criterion = torch.nn.CrossEntropyLoss()

        # Configure optimizer based on model parameters
        if params["optimizer"].lower() == "sgd":
            optimizer = optim.SGD(
                model.parameters(),
                lr=params["learning_rate"],
                weight_decay=params["weight_decay"],
                momentum=params["momentum"],
            )
        elif params["optimizer"].lower() == "adam":
            optimizer = optim.Adam(
                model.parameters(),
                lr=params["learning_rate"],
                weight_decay=params["weight_decay"],
                betas=params["betas"],
                eps=params["eps"],
            )
        elif params["optimizer"].lower() == "adamw":
            optimizer = optim.AdamW(
                model.parameters(),
                lr=params["learning_rate"],
                weight_decay=params["weight_decay"],
                betas=params["betas"],
                eps=params["eps"],
                amsgrad=params["amsgrad"],
            )
        elif params["optimizer"].lower() == "rmsprop":
            optimizer = optim.RMSprop(
                model.parameters(),
                lr=params["learning_rate"],
                weight_decay=params["weight_decay"],
                alpha=params["alpha"],
                eps=params["eps"],
                momentum=params["momentum"],
                centered=params["centered"],
            )
        else:
            print(
                f"Warning: Unknown optimizer '{params['optimizer']}'. Falling back to SGD."
            )
            optimizer = optim.SGD(
                model.parameters(),
                lr=params["learning_rate"],
                weight_decay=params["weight_decay"],
            )

        X_train_tensor = torch.tensor(X_train.astype(np.float32)).to(device)
        y_train_tensor = torch.tensor(y_train.astype(np.longlong)).to(device)

        dataset = TensorDataset(X_train_tensor, y_train_tensor)
        dataloader = DataLoader(dataset, batch_size=params["batch_size"], shuffle=True)

        for epoch in range(params["num_epochs"]):
            model.train()
            epoch_loss = 0.0

            for batch_X, batch_y in dataloader:
                optimizer.zero_grad()
                outputs = model(batch_X)
                loss = criterion(outputs, batch_y)
                loss.backward()
                optimizer.step()
                epoch_loss += loss.item()
    else:
        model.fit(X_train, y_train)

    return model


def save_models(models, input_size, s3_bucket, models_prefix="models/"):
    """
    Save trained models to S3 for inference.

    Args:
        models: Dictionary of trained models
        input_size: Input feature size
        s3_bucket: S3 bucket name
        models_prefix: S3 prefix for model files
    """
    s3_client = boto3.client("s3")

    for name, model in models.items():
        try:
            if isinstance(model, torch.nn.Module):
                # Save PyTorch models with metadata
                model_data = {
                    "state_dict": model.state_dict(),
                    "input_size": input_size,
                }

                if name == "FNN":
                    model_data["hidden_size"] = HYPERPARAMETERS[name].get(
                        "hidden_size", 64
                    )
                elif name == "Transformer":
                    model_data["embedding_dim"] = HYPERPARAMETERS[name].get(
                        "embedding_dim", 64
                    )
                    model_data["num_layers"] = HYPERPARAMETERS[name].get(
                        "num_layers", 2
                    )
                    model_data["nhead"] = HYPERPARAMETERS[name].get("nhead", 8)

                # Save to local file first
                model_filename = f"{name}.pth"
                torch.save(model_data, model_filename)

                # Upload to S3
                s3_key = f"{models_prefix}{model_filename}"
                s3_client.upload_file(model_filename, s3_bucket, s3_key)

                # Clean up local file
                os.remove(model_filename)

                print(f"Saved PyTorch model {name} to s3://{s3_bucket}/{s3_key}")

            else:
                # Save scikit-learn models
                model_filename = f"{name.replace(' ', '_')}.pkl"

                # Save to local file first
                with open(model_filename, "wb") as f:
                    pickle.dump(model, f)

                # Upload to S3
                s3_key = f"{models_prefix}{model_filename}"
                s3_client.upload_file(model_filename, s3_bucket, s3_key)

                # Clean up local file
                os.remove(model_filename)

                print(f"Saved sklearn model {name} to s3://{s3_bucket}/{s3_key}")

        except Exception as e:
            print(f"Failed to save model {name}: {e}")


def save_label_encoder(label_encoder, s3_bucket, models_prefix="models/"):
    """
    Save label encoder to S3.

    Args:
        label_encoder: Trained label encoder
        s3_bucket: S3 bucket name
        models_prefix: S3 prefix for model files
    """
    try:
        s3_client = boto3.client("s3")

        # Save to local file first
        encoder_filename = "label_encoder.pkl"
        with open(encoder_filename, "wb") as f:
            pickle.dump(label_encoder, f)

        # Upload to S3
        s3_key = f"{models_prefix}{encoder_filename}"
        s3_client.upload_file(encoder_filename, s3_bucket, s3_key)

        # Clean up local file
        os.remove(encoder_filename)

        print(f"Saved label encoder to s3://{s3_bucket}/{s3_key}")

    except Exception as e:
        print(f"Failed to save label encoder: {e}")
