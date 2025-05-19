"""
Visualization utilities for model evaluation and analysis.

This module provides functions for plotting confusion matrices, loss curves,
and ROC curves with consistent styling.
"""

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, RocCurveDisplay

# Constants for styling
STYLE = {
    'color': '0.2',
    'fontname': 'Arial',
    'fontsize': {
        'title': 35,
        'label': 25,
        'ticks': 20,
        'legend': 20
    },
    'linewidth': 2.5,
    'colors': ["#3d348b", "#7678ed", "#f7b801", "#f18701", "#f35b04"]
}

# Set default figure size
mpl.rcParams['figure.figsize'] = (12, 10)

def _apply_axis_styling(ax, title='', xlabel='', ylabel='', show_top_right=False):
    """Apply consistent styling to plot axes."""
    # Set labels
    if title:
        ax.set_title(title, fontsize=STYLE['fontsize']['title'], 
                    fontname=STYLE['fontname'], weight='bold', color=STYLE['color'])
    if xlabel:
        ax.set_xlabel(f"\n{xlabel}", fontsize=STYLE['fontsize']['label'],
                     fontname=STYLE['fontname'], weight='bold', color=STYLE['color'])
    if ylabel:
        ax.set_ylabel(f"{ylabel}\n", fontsize=STYLE['fontsize']['label'],
                     fontname=STYLE['fontname'], weight='bold', color=STYLE['color'])
    
    # Style ticks
    ax.tick_params(width=STYLE['linewidth'], length=10, pad=10, 
                  color=STYLE['color'])
    for label in ax.get_xticklabels() + ax.get_yticklabels():
        label.set_fontsize(STYLE['fontsize']['ticks'])
        label.set_fontname(STYLE['fontname'])
        label.set_weight('bold')
        label.set_color(STYLE['color'])
    
    # Style spines
    for spine in ax.spines.values():
        spine.set_linewidth(STYLE['linewidth'])
        spine.set_color(STYLE['color'])
    
    if not show_top_right:
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)

def plot_confusion_matrix(labels, predictions, threshold=0.5, title='', 
                         xlabel='', ylabel='', ax=None):
    """Plot confusion matrix with consistent styling.
    
    Args:
        labels: True labels
        predictions: Model predictions (probabilities)
        threshold: Classification threshold
        title: Plot title
        xlabel: X-axis label
        ylabel: Y-axis label
        ax: Matplotlib axis to plot on
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=(10, 10))
    
    # Create confusion matrix display
    cm_display = ConfusionMatrixDisplay.from_predictions(
        y_true=labels,
        y_pred=predictions > threshold,
        display_labels=['natural', 'de novo'],
        cmap=plt.cm.Blues,
        normalize=None,
        ax=ax,
        colorbar=False
    )
    
    # Style text in confusion matrix
    cm = confusion_matrix(labels, predictions > threshold)
    kwargs = {"fontsize": STYLE['fontsize']['label'],
              "fontname": STYLE['fontname'],
              "weight": "bold"}
    
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            color = 'w' if cm[i, j] > cm.max() / 2 else STYLE['color']
            ax.text(j, i, cm[i, j], ha='center', va='center',
                   color=color, **kwargs)
    
    # Add and style colorbar
    cbar = plt.colorbar(mappable=cm_display.im_, ax=ax)
    cbar.set_ticklabels(
        np.arange(0, 1751, 250),
        fontsize=STYLE['fontsize']['ticks'],
        fontname=STYLE['fontname'],
        weight='bold',
        color=STYLE['color']
    )
    
    _apply_axis_styling(ax, title, xlabel, ylabel, show_top_right=True)
    return cm

def plot_loss(history, fold_num, ax=None):
    """Plot training and validation loss curves.
    
    Args:
        history: Training history dictionary
        fold_num: Fold number for labeling
        ax: Matplotlib axis to plot on
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=(10, 9))
    
    kwargs_plot = {'linewidth': STYLE['linewidth']}
    
    # Plot training and validation loss
    ax.plot(history['loss'], color=STYLE['colors'][fold_num],
            label=f'Train Fold_{fold_num}', **kwargs_plot)
    ax.plot(history['val_loss'], color=STYLE['colors'][fold_num],
            label=f'Val Fold_{fold_num}', linestyle="--", **kwargs_plot)
    
    ax.set_ylim(0.1, 0.8)
    ax.legend(prop={'size': STYLE['fontsize']['legend'],
                   'weight': 'bold'})
    
    _apply_axis_styling(ax, "Loss Function", "Epoch", "Loss")

def plot_roc(labels, predictions, fold_num, title='', xlabel='', ylabel='', ax=None):
    """Plot ROC curve.
    
    Args:
        labels: True labels
        predictions: Model predictions (probabilities)
        fold_num: Fold number for labeling
        title: Plot title
        xlabel: X-axis label
        ylabel: Y-axis label
        ax: Matplotlib axis to plot on
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=(10, 9))
    
    RocCurveDisplay.from_predictions(
        y_true=labels,
        y_pred=predictions,
        name=f"Fold {fold_num}",
        color=STYLE['colors'][fold_num],
        ax=ax,
        linewidth=STYLE['linewidth'],
        alpha=0.7
    )
    
    _apply_axis_styling(ax, title, xlabel, ylabel)
    
    # Style legend
    legend = ax.legend(loc=4, prop={
        'size': STYLE['fontsize']['legend'],
        'weight': 'bold'
    })
    for text in legend.get_texts():
        text.set_color(STYLE['color'])