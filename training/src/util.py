"""
Utility functions for model training and evaluation.

This module provides functions for data preparation, model creation,
cross-validation, and evaluation of protein structure classification models.
"""

import tensorflow as tf
from tensorflow import keras
print(tf.__version__)

# import matplotlib as mpl
# import matplotlib.pyplot as plt
import numpy as np

import sklearn
from sklearn.preprocessing import RobustScaler
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.metrics import (
    accuracy_score, balanced_accuracy_score, roc_auc_score,
    average_precision_score, matthews_corrcoef, precision_score,
    recall_score, confusion_matrix
)

from imblearn.over_sampling import SMOTE

from tensorflow.keras.callbacks import LearningRateScheduler

SEED = 7
BATCH_SIZE = 1024
EPOCHS = 300
LEARNING_RATE = 1e-3
DECAY_RATE = LEARNING_RATE / EPOCHS

METRICS = [
    keras.metrics.BinaryAccuracy(name='accuracy'),
    keras.metrics.Precision(name='precision'),
    keras.metrics.Recall(name='recall'),
    keras.metrics.AUC(name='auc'),
    keras.metrics.AUC(name='prc', curve='PR'),
]




def make_model(metrics=METRICS, output_bias=None):
    """Create and compile the neural network model.
    
    Args:
        metrics: List of metrics to track
        output_bias: Initial bias for output layer
    
    Returns:
        Compiled Keras model
    """
    if output_bias is not None:
        output_bias = tf.keras.initializers.Constant(output_bias)
    
    model = keras.Sequential([
        keras.layers.Dense(32, activation='relu', input_shape=(6,)),
        keras.layers.Dropout(0.5),
        keras.layers.Dense(32, activation='relu'),
        keras.layers.Dropout(0.5),
        keras.layers.Dense(1, activation='sigmoid', bias_initializer=output_bias),
    ])
    
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=LEARNING_RATE),
        loss=keras.losses.BinaryCrossentropy(),
        metrics=metrics
    )
    
    return model

def get_model_name(k):
    return 'model_'+str(k)+'.h5'


def prep_data(dataframe):
    """Prepare data for model training.
    
    Args:
        dataframe: Input DataFrame with features and labels
    
    Returns:
        Tuple of (train_val_X, train_val_y, test_X, test_y, train_val_name, test_name)
    """
    # Split data
    train_val_df, test_df = train_test_split(
        dataframe, 
        test_size=0.2,
        stratify=dataframe['type'],
        random_state=SEED
    )
    
    # Extract labels and names
    train_val_y = np.array(train_val_df.pop('type'))
    test_y = np.array(test_df.pop('type'))
    train_val_name = np.array(train_val_df.pop('name'))
    test_name = np.array(test_df.pop('name'))
    
    # Extract features
    feature_columns = [
        '-logP_Struc', '-logP_Contact', "area_sc",
        'per_con', 'per_no_con', 'contact_order'
    ]
    train_val_X = np.array(train_val_df[feature_columns])
    test_X = np.array(test_df[feature_columns])
    
    # Scale features
    scaler = RobustScaler()
    train_val_X = scaler.fit_transform(train_val_X)
    test_X = scaler.transform(test_X)
    
    return train_val_X, train_val_y, test_X, test_y, train_val_name, test_name

def exp_decay(epoch):
    """Calculate learning rate with exponential decay.
    
    Args:
        epoch: Current epoch number
    
    Returns:
        Decayed learning rate
    """
    return LEARNING_RATE * np.exp(-DECAY_RATE * epoch)

def cross_validate(X, y, create_model):
    """Perform k-fold cross-validation with SMOTE resampling.
    
    Args:
        X: Input features
        y: Target labels
        create_model: Function to create model
    
    Returns:
        Tuple of (history_list, model_list, datasplit_list)
    """
    history_list = []
    model_list = []
    datasplit_list = []
    
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
    
    for fold_num, (train_idx, val_idx) in enumerate(skf.split(X, y), 1):
        print(f"\n{'='*30} Fold {fold_num} {'='*30}")
        
        # Split data
        train_X, val_X = X[train_idx], X[val_idx]
        train_y, val_y = y[train_idx], y[val_idx]
        
        # Apply SMOTE
        smote = SMOTE(random_state=SEED)
        train_X_resampled, train_y_resampled = smote.fit_resample(train_X, train_y)
        
        # Create datasets
        val_ds = tf.data.Dataset.from_tensor_slices((val_X, val_y)).batch(BATCH_SIZE).prefetch(1)
        train_ds = tf.data.Dataset.from_tensor_slices(
            (train_X_resampled, train_y_resampled)
        ).batch(BATCH_SIZE).prefetch(1)
        
        # Train model
        model = create_model()
        history = model.fit(
            train_ds,
            epochs=EPOCHS,
            steps_per_epoch=10,
            callbacks=[keras.callbacks.LearningRateScheduler(exp_decay)],
            batch_size=BATCH_SIZE,
            validation_data=val_ds
        )
        
        history_list.append(history)
        model_list.append(model)
        datasplit_list.append([train_idx, val_idx])
        
        tf.keras.backend.clear_session()
    
    return history_list, model_list, datasplit_list


def eval_models(df, model_list, datasplit_list):
    """Evaluate models on training and validation sets.
    
    Args:
        df: Input DataFrame
        model_list: List of trained models
        datasplit_list: List of data split indices
    
    Returns:
        Tuple of (train_scores, val_scores)
    """
    METRICS = [
        'Accuracy', 'Balanced Accuracy', 'ROC_AUC',
        'Average Precision', 'Matthews Correlation',
        'Precision', 'Recall'
    ]
    
    train_scores = []
    val_scores = []
    threshold = 0.5
    
    # Prepare data
    train_val_X, train_val_y, test_X, test_y, _, _ = prep_data(df)
    
    for i, (model, (train_idx, val_idx)) in enumerate(zip(model_list, datasplit_list)):
        # Get predictions
        train_X, val_X = train_val_X[train_idx], train_val_X[val_idx]
        train_y, val_y = train_val_y[train_idx], train_val_y[val_idx]
        
        train_pred = model.predict(train_X, batch_size=BATCH_SIZE)
        val_pred = model.predict(val_X, batch_size=BATCH_SIZE)
        test_pred = model.predict(test_X, batch_size=BATCH_SIZE)
        
        # Calculate metrics
        cm = confusion_matrix(test_y, test_pred > threshold)
        fp_rate = cm[0][1] / (cm[0][0] + cm[0][1])
        fn_rate = cm[1][0] / (cm[1][0] + cm[1][1])
        
        # Training scores
        train_metrics = [
            accuracy_score(train_y, train_pred > threshold),
            balanced_accuracy_score(train_y, train_pred > threshold),
            roc_auc_score(train_y, train_pred),
            average_precision_score(train_y, train_pred),
            matthews_corrcoef(train_y, train_pred >= threshold),
            precision_score(train_y, train_pred >= threshold),
            recall_score(train_y, train_pred >= threshold),
            fp_rate,
            fn_rate
        ]
        
        # Validation scores
        val_metrics = [
            accuracy_score(val_y, val_pred > threshold),
            balanced_accuracy_score(val_y, val_pred > threshold),
            roc_auc_score(val_y, val_pred),
            average_precision_score(val_y, val_pred),
            matthews_corrcoef(val_y, val_pred >= threshold),
            precision_score(val_y, val_pred >= threshold),
            recall_score(val_y, val_pred >= threshold),
            fp_rate,
            fn_rate
        ]
        
        train_scores.append(train_metrics)
        val_scores.append(val_metrics)
    
    # Calculate and print average scores
    train_avg = np.array(train_scores).mean(axis=0).round(2)
    val_avg = np.array(val_scores).mean(axis=0).round(2)
    train_std = np.array(train_scores).std(axis=0).round(2)
    val_std = np.array(val_scores).std(axis=0).round(2)
    
    for i, metric in enumerate(METRICS):
        print(f"\n{metric}:")
        print(f"  Training:    {train_avg[i]} ± {train_std[i]}")
        print(f"  Validation:  {val_avg[i]} ± {val_std[i]}")
    
    return train_scores, val_scores
# def plot_loss(history, label, n):
#     # Use a log scale on y-axis to show the wide range of values.
#     colors = ["#ee6a00","#72acff","#fdaf00","#907aff"]
#     plt.semilogy(history.epoch, history.history['loss'],
#                color=colors[n], label='Train ' + label)
#     plt.semilogy(history.epoch, history.history['val_loss'],
#                color=colors[n], label='Val ' + label,
#                linestyle="--")
#     plt.xlabel('Epoch')
#     plt.ylabel('Loss')
#     plt.legend()

    
    
# def plot_metrics(history,title='', 
#                   xlabel='', ylabel='',ax=None,grid=False,my_col ='0.2'):
#     colors = plt.rcParams['axes.prop_cycle'].by_key()['color']
#     metrics = ['loss', 'auc', 'precision', 'recall']
#     for n, metric in enumerate(metrics):
#         name = metric.replace("_"," ").capitalize()
#         plt.subplot(2,2,n+1)
#         plt.plot(history.epoch, history.history[metric], color=colors[0], label='Train')
#         plt.plot(history.epoch, history.history['val_'+metric],
#                  color=colors[0], linestyle="--", label='Val')


#         plt.title(title, fontsize=20,fontname='Arial', weight='bold', color=my_col)
#         plt.xlabel(xlabel, fontsize=15,fontname='Arial', weight='bold', color=my_col)
#         plt.ylabel(ylabel, fontsize=15,fontname='Arial', weight='bold', color=my_col)
#         plt.xticks(fontsize=15,fontname='Arial', weight='bold', color=my_col)
#         plt.yticks(fontsize=15,fontname='Arial', weight='bold', color=my_col)
#         if grid:    
#             plt.grid(color='black',axis='both',linestyle=':',linewidth=0.5,which='major')


#         plt.xlabel('Epoch')
#         plt.ylabel(name)
#         if metric == 'loss':
#             plt.ylim([0, plt.ylim()[1]])
#         elif metric == 'auc':
#             plt.ylim([0.8,1])
#         else:
#             plt.ylim([0,1])

#         plt.legend()

    
    
# def plot_roc(name, labels, predictions, **kwargs):
#     fp, tp, _ = roc_curve(labels, predictions)

#     plt.plot(100*fp, 100*tp, label=name, linewidth=2, **kwargs)
#     plt.xlabel('False positives [%]')
#     plt.ylabel('True positives [%]')
#     plt.xlim([-0.5,20])
#     plt.ylim([80,100.5])
#     plt.grid(True)
#     ax = plt.gca()
#     ax.set_aspect('equal')
    
    
# def make_ds(features, labels):
#     ds = tf.data.Dataset.from_tensor_slices((features, labels))#.cache()
#     ds = ds.shuffle(100000).repeat()
#     return ds


# def plot_prc(name, labels, predictions, **kwargs):
#     precision, recall, _ = precision_recall_curve(labels, predictions)

#     plt.plot(precision, recall, label=name, linewidth=2, **kwargs)
#     plt.xlabel('Precision')
#     plt.ylabel('Recall')
#     plt.grid(True)
#     ax = plt.gca()
#     ax.set_aspect('equal')
    
    
