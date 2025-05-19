"""
Main training script for protein structure classification model.

This script handles the training, evaluation, and visualization of the
protein structure classification model using k-fold cross-validation.
"""

import os
import pickle
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras
import matplotlib.pyplot as plt

from util import (
    prep_data,
    make_model,
    cross_validate,
    eval_models,
    BATCH_SIZE
)
from visualisation import plot_confusion_matrix, plot_loss, plot_roc

def check_gpu():
    """Check and print GPU availability."""
    print("\nGPU Information:")
    print("Num GPUs Available:", len(tf.config.list_physical_devices('GPU')))
    print("TensorFlow version:", tf.__version__)

def load_data(data_path):
    """Load and prepare data for training.
    
    Args:
        data_path: Path to the data file
        
    Returns:
        Tuple of (train_val_X, train_val_y, test_X, test_y, train_val_name, test_name)
    """
    print("\nLoading data...")
    df = pd.read_csv(data_path)
    return prep_data(df)

def train_models(train_val_X, train_val_y):
    """Train models using k-fold cross-validation.
    
    Args:
        train_val_X: Training features
        train_val_y: Training labels
        
    Returns:
        Tuple of (history_list, model_list, datasplit_list)
    """
    print("\nStarting model training...")
    return cross_validate(train_val_X, train_val_y, make_model)

def save_models(model_list, history_list, save_dir='./models/'):
    """Save trained models and their histories.
    
    Args:
        model_list: List of trained models
        history_list: List of training histories
        save_dir: Directory to save models
    """
    print("\nSaving models and histories...")
    os.makedirs(save_dir, exist_ok=True)
    os.makedirs(os.path.join(save_dir, 'histories'), exist_ok=True)
    
    for i, model in enumerate(model_list):
        model.save(os.path.join(save_dir, f'model_{i}'))
        
    for i, history in enumerate(history_list):
        with open(os.path.join(save_dir, 'histories', f'history_{i}'), 'wb') as f:
            pickle.dump(history.history, f)

def load_models(model_dir='./models/', num_models=5):
    """Load saved models and their histories.
    
    Args:
        model_dir: Directory containing saved models
        num_models: Number of models to load
        
    Returns:
        Tuple of (model_list, history_list)
    """
    print("\nLoading saved models...")
    model_list = []
    history_list = []
    
    for i in range(num_models):
        model = keras.models.load_model(os.path.join(model_dir, f'model_{i}'))
        model_list.append(model)
        
        with open(os.path.join(model_dir, 'histories', f'history_{i}'), "rb") as f:
            history = pickle.load(f)
        history_list.append(history)
    
    return model_list, history_list

def create_visualizations(model_list, history_list, test_X, test_y, img_dir='./img/'):
    """Create and save model performance visualizations.
    
    Args:
        model_list: List of trained models
        history_list: List of training histories
        test_X: Test features
        test_y: Test labels
        img_dir: Directory to save images
    """
    print("\nCreating visualizations...")
    os.makedirs(img_dir, exist_ok=True)
    
    # Plot ROC curves
    fig_roc, ax_roc = plt.subplots(1, 1, figsize=(10, 15))
    for i, model in enumerate(model_list):
        test_pred = model.predict(test_X, batch_size=BATCH_SIZE)
        plot_roc(test_y, test_pred, i,
                xlabel="False Positive Rate",
                ylabel="True Positive Rate",
                title="ROC Curves",
                ax=ax_roc)
    
    ax_roc.plot([0, 1], [0, 1], "k--",
                label="chance level (AUC = 0.5)",
                linewidth=2.5, alpha=0.7)
    plt.axis("square")
    plt.savefig(os.path.join(img_dir, "ROC.png"), dpi=300, bbox_inches='tight')
    plt.close()
    
    # Plot confusion matrices
    for i, model in enumerate(model_list):
        test_pred = model.predict(test_X, batch_size=BATCH_SIZE)
        fig_cm, ax_cm = plt.subplots(1, 1, figsize=(10, 10))
        plot_confusion_matrix(
            test_y, test_pred,
            title=f"Neural Network Fold {i}",
            ylabel="Actual label",
            xlabel="Predicted label",
            ax=ax_cm
        )
        plt.savefig(os.path.join(img_dir, f"cm_{i}.png"),
                   dpi=300, bbox_inches='tight')
        plt.close()
    
    # Plot loss curves
    fig_loss, ax_loss = plt.subplots(1, 1, figsize=(10, 9))
    for i, history in enumerate(history_list):
        plot_loss(history, i, ax_loss)
    plt.savefig(os.path.join(img_dir, "loss.png"),
                dpi=300, bbox_inches='tight')
    plt.close()

def main():
    """Main function to run the training pipeline."""
    # Setup
    check_gpu()
    
    # Load and prepare data
    data_path = "./data/cleaned_data_cropped.csv"
    train_val_X, train_val_y, test_X, test_y, train_val_name, test_name = load_data(data_path)
    
    # Train or load models
    TRAIN_NEW_MODELS = False
    if TRAIN_NEW_MODELS:
        history_list, model_list, datasplit_list = train_models(train_val_X, train_val_y)
        save_models(model_list, history_list)
    else:
        model_list, history_list = load_models()
    
    # Create visualizations
    create_visualizations(model_list, history_list, test_X, test_y)
    
    print("\nTraining pipeline completed successfully!")

if __name__ == "__main__":
    main()
    
    
    
