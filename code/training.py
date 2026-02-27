import copy
import torch
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
import pickle
import boto3
import os
from config import HYPERPARAMETERS, RESULTS_PATH, SEED, is_experimental


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

        # Hold out last 15% of training data for early stopping monitoring
        patience = params.get("early_stopping_patience", 20)
        val_split = int(len(X_train) * 0.85)
        X_fit = X_train[:val_split]
        y_fit = y_train[:val_split]
        X_es_val = X_train[val_split:]
        y_es_val = y_train[val_split:]

        X_fit_tensor = torch.tensor(X_fit.astype(np.float32)).to(device)
        y_fit_tensor = torch.tensor(y_fit.astype(np.longlong)).to(device)
        X_es_val_tensor = torch.tensor(X_es_val.astype(np.float32)).to(device)
        y_es_val_tensor = torch.tensor(y_es_val.astype(np.longlong)).to(device)

        dataset = TensorDataset(X_fit_tensor, y_fit_tensor)
        generator = torch.Generator()
        generator.manual_seed(SEED)
        dataloader = DataLoader(
            dataset, batch_size=params["batch_size"], shuffle=True, generator=generator
        )

        best_val_loss = float("inf")
        epochs_without_improvement = 0
        best_state_dict = None

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

            # Early stopping: check validation loss
            model.eval()
            with torch.no_grad():
                val_outputs = model(X_es_val_tensor)
                val_loss = criterion(val_outputs, y_es_val_tensor).item()

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                epochs_without_improvement = 0
                best_state_dict = copy.deepcopy(model.state_dict())
            else:
                epochs_without_improvement += 1
                if epochs_without_improvement >= patience:
                    if verbose:
                        print(f"  Early stopping at epoch {epoch + 1}/{params['num_epochs']}")
                    model.load_state_dict(best_state_dict)
                    break

        # Restore best weights if training completed all epochs
        if best_state_dict is not None and epochs_without_improvement < patience:
            model.load_state_dict(best_state_dict)
    else:
        model.fit(X_train, y_train)

    return model


def save_models(models, input_size, s3_bucket, models_prefix="models/"):
    """
    Save trained models to S3 for inference.
    Skips S3 upload for experimental runs to avoid overwriting production models.

    Args:
        models: Dictionary of trained models
        input_size: Input feature size
        s3_bucket: S3 bucket name
        models_prefix: S3 prefix for model files
    """
    # Skip model saving entirely for experimental runs
    if is_experimental():
        print("Experimental run detected - skipping model uploads to S3")
        return

    s3_client = boto3.client("s3")

    for name, model in models.items():
        try:
            if isinstance(model, torch.nn.Module):
                # Save PyTorch models with metadata
                model_data = {
                    "state_dict": model.state_dict(),
                    "input_size": input_size,
                    "hyperparameters": HYPERPARAMETERS[name],
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

                # Clean up temporary file
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

                # Clean up temporary file
                os.remove(model_filename)

                print(f"Saved sklearn model {name} to s3://{s3_bucket}/{s3_key}")

        except Exception as e:
            print(f"Failed to save model {name}: {e}")


def save_label_encoder(label_encoder, s3_bucket, models_prefix="models/"):
    """
    Save label encoder to S3.
    Skips S3 upload for experimental runs to avoid overwriting production encoder.

    Args:
        label_encoder: Trained label encoder
        s3_bucket: S3 bucket name
        models_prefix: S3 prefix for model files
    """
    # Skip encoder saving entirely for experimental runs
    if is_experimental():
        print("Experimental run detected - skipping label encoder upload to S3")
        return

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
