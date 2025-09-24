import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np

from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, RocCurveDisplay


STD_COLOR='0.2'
mpl.rcParams['figure.figsize'] = (12, 10)

my_colors = ["#3d348b","#7678ed","#f7b801","#f18701","#f35b04"]
# PAL=[my_colors[1],my_colors[3],my_colors[2],my_colors[0]]
PAL = my_colors

def plot_cm(labels, predictions, cmap, threshold=0.5, title='', xlabel='', ylabel='', ax=None, grid=False, my_col='0.2', normalize=None):
    cm = confusion_matrix(labels, predictions > threshold)
    class_names = ['natural', 'de novo']

    # plt.figure(figsize=(10, 10))

    fig, ax = plt.subplots(figsize=(10, 10))
    cm_display = ConfusionMatrixDisplay.from_predictions(y_true = labels,
                                      y_pred=predictions>threshold,
                                      display_labels=class_names,
                                      cmap=cmap,
                                      normalize=normalize,
                                      ax=ax,
                                      text_kw={"fontsize":30,
                                      "fontname":"Arial",
                                      "weight":"bold"},
                                      colorbar=False,)
    # cm_display.plot(colormap=my_col)
    
    kwargs = {"fontsize":30,
              "fontname":"Arial",
              "weight":"bold"}
    
    ax.tick_params(width=2,length=10,pad=10, color='0.2')
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            cm_display.ax_.text(j, i, cm_display.confusion_matrix[i, j], ha='center', va='center',
                          color='w' if cm_display.confusion_matrix[i, j] > cm_display.confusion_matrix.max() / 2 else '0.2',**kwargs)


    # Add a colorbar associated with cm_display
    # kwargs_cbar={'fontsize':40,'fontname':'Arial', 'weight':'bold', 'color':'0.2'}
    cbar = plt.colorbar(mappable=cm_display.im_, ax=ax)
    
    
    cbar.set_ticklabels(np.arange(0,1751,250), fontsize=30, fontname='Arial', weight='bold', color=my_col)
    

    
    plt.title(title, fontsize=30, fontname='Arial', weight='bold', color=my_col)
    plt.xlabel(xlabel, fontsize=25, fontname='Arial', weight='bold', color=my_col)
    plt.ylabel(ylabel, fontsize=25, fontname='Arial', weight='bold', color=my_col)
    plt.xticks(fontsize=25, fontname='Arial', weight='bold', color=my_col)
    plt.yticks(fontsize=25, fontname='Arial', weight='bold', color=my_col)

    if grid:
        plt.grid(color='black', axis='both', linestyle=':', linewidth=0.5, which='major')

    for axis in ['bottom','left','right','top']:
        ax.spines[axis].set_linewidth(2)
        ax.spines[axis].set_color('0.2')
        
    print(cm)
    
    
def plot_loss(history,n,label,ax=None):


    kwargs_title={'fontsize':35, 'weight':'bold', 'color':STD_COLOR}
    kwargs_label={'fontsize':20, 'weight':'bold', 'color':STD_COLOR}
    kwargs_ticks={'fontsize':20, 'weight':'bold', 'color':STD_COLOR}
    kwargs_plots={'linewidth':2.5}

    # fig, ax = plt.subplots(1,1,figsize=(10,9))


    ax.plot(history['loss'],
            color=PAL[n], label='Train Fold_' + str(label), **kwargs_plots)
    ax.plot(history['val_loss'],
            color=PAL[n], label='Val Fold_' + str(label),
            linestyle="--",**kwargs_plots)

    # Axes
    plt.title("Lossfunction\n",**kwargs_title)
    plt.ylabel("Loss\n",**kwargs_label)
    plt.xlabel("\nEpoch",**kwargs_label)
    plt.yticks(**kwargs_ticks)
    plt.xticks(**kwargs_ticks)
    plt.ylim(0.1,0.8)
    plt.legend()


    for axis in ['bottom','left']:
        ax.spines[axis].set_linewidth(2.5)
        ax.spines[axis].set_color(STD_COLOR)

    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.tick_params(width=2.5,length=5,pad=10, color=STD_COLOR)



def plot_ROC(labels, predictions, i, title='', xlabel='', ylabel='', ax=None, grid=False):
    
    kwargs_title={'fontsize':35, 'weight':'bold', 'color':STD_COLOR}
    kwargs_label={'fontsize':20, 'weight':'bold', 'color':STD_COLOR}
    kwargs_ticks={'fontsize':20, 'weight':'bold', 'color':STD_COLOR}
    kwargs_plots={'linewidth':2.5, 'alpha':0.7}
    
    
    # fig, ax = plt.subplots(1,1,figsize=(10,15))

    # test_pred = model.predict(test_X, batch_size = BATCH_SIZE)
    RocCurveDisplay.from_predictions(y_true = labels,
                    y_pred=predictions,
                    name=f"Fold {i}",
                    color=PAL[i],
                    ax=ax,
                    linewidth=3,
                    alpha=0.7
    )

    # ax.plot([0, 1], [0, 1], "k--", label="chance level (AUC = 0.5)",**kwargs_plots)
    # plt.axis("square")



    # Axes
    plt.title(title,**kwargs_title)
    plt.ylabel(ylabel,**kwargs_label)
    plt.xlabel(xlabel,**kwargs_label)
    plt.yticks(**kwargs_ticks)
    plt.xticks(**kwargs_ticks)
    # plt.ylim(0.1,0.8)
    # plt.legend()

    # add_cosmetics(title="One-vs-Rest ROC curves:\nde_novo vs (high & low)\n",xlabel="\nFalse Positive Rate",ylabel="True Positive Rate\n")

    ax.tick_params(width=2.5,length=10,pad=10, color=STD_COLOR)
    for axis in ['bottom','left']:
        ax.spines[axis].set_linewidth(2.5)
        ax.spines[axis].set_color(STD_COLOR)
        
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    plt.rc('font',family='Arial',weight='bold')
    plt.rc('legend' )
    l = plt.legend(loc=4, prop={'size': 20})
    for text in l.get_texts():
        text.set_color("0.2")
    # plt.show()