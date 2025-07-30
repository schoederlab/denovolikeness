import tensorflow as tf
from tensorflow import keras
print(tf.__version__)

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np

import sklearn
from sklearn.preprocessing import RobustScaler
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

from imblearn.over_sampling import SMOTE

from tensorflow.keras.callbacks import LearningRateScheduler

SEED = 7
BATCH_SIZE = 1024
EPOCHS = 350
LEARNING_RATE = 1e-3
DECAY_RATE = LEARNING_RATE / EPOCHS

METRICS = [
      keras.metrics.BinaryAccuracy(name='accuracy'),
      keras.metrics.Precision(name='precision'),
      keras.metrics.Recall(name='recall'),
      keras.metrics.AUC(name='auc'),
      keras.metrics.AUC(name='prc', curve='PR'), # precision-recall curve
]




def make_model(metrics=METRICS, output_bias=None):
    if output_bias is not None:
        output_bias = tf.keras.initializers.Constant(output_bias)
        
    model = keras.Sequential([        
        keras.layers.Dense(
            32, activation='relu',
            input_shape=(6,),
            # kernel_initializer=initializers.Ones()
        ),
        keras.layers.Dropout(0.5),        
        keras.layers.Dense(
            32, activation='relu',
            input_shape=(6,)),
        keras.layers.Dropout(0.5),
        
        keras.layers.Dense(1, activation='sigmoid',
                           bias_initializer=output_bias),
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
    train_val_df, test_df = train_test_split(dataframe, test_size=0.2,stratify=dataframe['type'],random_state=SEED)
    train_val_y = np.array(train_val_df.pop('type'))
    test_y = np.array(test_df.pop('type'))
    train_val_name = np.array(train_val_df.pop('name'))
    test_name = np.array(test_df.pop('name'))

    train_val_X = np.array(train_val_df[['-logP_Struc','-logP_Contact',"area_sc",'per_con','per_no_con','contact_order']])
    test_X = np.array(test_df[['-logP_Struc','-logP_Contact',"area_sc",'per_con','per_no_con','contact_order']])

    scaler = RobustScaler()
    train_val_X = scaler.fit_transform(train_val_X)
    test_X = scaler.transform(test_X)

    return train_val_X, train_val_y, test_X, test_y, train_val_name, test_name

def exp_decay(epoch):
    lrate = LEARNING_RATE * np.exp(-DECAY_RATE*epoch)
    return lrate

def cross_validate(X, y, create_model):
    
    history_list = []
    model_list = []
    datasplit_list = []

    fold_num = 0

    sKfold = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
    split = sKfold.split(X, y)    
    
    for train_ix, val_ix in split:

        fold_num += 1
        print(f"\n################################# Fold Number {fold_num}#################################")

        train_X, val_X = X[train_ix], X[val_ix]
        train_y, val_y = y[train_ix], y[val_ix]



        ros = SMOTE(random_state=SEED)
        X_train_resampled, y_train_resampled = ros.fit_resample(train_X, train_y)



        val_ds = tf.data.Dataset.from_tensor_slices((val_X, val_y))
        val_ds = val_ds.batch(BATCH_SIZE).prefetch(1) 

        resampled_ds = tf.data.Dataset.from_tensor_slices((X_train_resampled,y_train_resampled))
        resampled_ds = resampled_ds.batch(BATCH_SIZE).prefetch(1)
                        
        lr_rate = LearningRateScheduler(exp_decay)
        callbacks_list = [lr_rate]
        
        
        model = create_model()
        
        
        history = model.fit(
            resampled_ds,
            epochs=EPOCHS,
            steps_per_epoch=10,
            callbacks=callbacks_list,
            batch_size=BATCH_SIZE,
            validation_data=val_ds)


        # model.save_weights(save_dir+get_model_name(fold_num))
        # model.load_weights("./saved_models/model_ext_"+str(fold_num)+".h5")

        # results = model.evaluate(val_X, val_y)

        history_list.append(history)
        model_list.append(model)
        datasplit_list.append([train_ix,val_ix])    
            
        tf.keras.backend.clear_session()
        
    return history_list, model_list, datasplit_list


def eval_models(df, model_list, datasplit_list):
    
    
    EVAL_METRICS = ['Accuracy','Balanced Accuracy', 'ROC_AUC', 'Average Precision', 'Matthews Correlationcoefficient', 'Precision', 'Recall']
    train_scores = []
    val_scores = []
    
    thresh = 0.5
    
    train_val_X, train_val_y, test_X, test_y, _, _ = prep_data(df)
    
    for i,model in enumerate(model_list):
        
        train_ix,val_ix = datasplit_list[i]
        train_X, val_X = train_val_X[train_ix], train_val_X[val_ix]
        train_y, val_y = train_val_y[train_ix], train_val_y[val_ix]
        
        model.load_weights("./saved_models/model_ext_"+str(i+1)+".h5")
        test_pred = model.predict(test_X, batch_size=BATCH_SIZE)
        train_pred = model.predict(train_X, batch_size=BATCH_SIZE)
        val_pred = model.predict(val_X, batch_size=BATCH_SIZE)
        
        cm = confusion_matrix(test_y,test_pred>thresh)
        fp_rate = cm[0][1]/(cm[0][0]+cm[0][1])
        fn_rate = cm[1][0]/(cm[1][0]+cm[1][1])

    
        train_acc = sklearn.metrics.accuracy_score(train_y,train_pred>thresh)
        train_bal_acc = sklearn.metrics.balanced_accuracy_score(train_y,train_pred>thresh)
        train_roc_auc = sklearn.metrics.roc_auc_score(train_y, train_pred)
        train_avg_prec = sklearn.metrics.average_precision_score(train_y, train_pred)
        train_mcc = sklearn.metrics.matthews_corrcoef(train_y, train_pred >= thresh)
        train_prec = sklearn.metrics.precision_score(train_y, train_pred >= thresh)
        train_rec = sklearn.metrics.recall_score(train_y, train_pred >= thresh)
        
        val_acc = sklearn.metrics.accuracy_score(val_y,val_pred>thresh)
        val_bal_acc = sklearn.metrics.balanced_accuracy_score(val_y,val_pred>thresh)
        val_roc_auc = sklearn.metrics.roc_auc_score(val_y, val_pred)
        val_avg_prec = sklearn.metrics.average_precision_score(val_y, val_pred)
        val_mcc = sklearn.metrics.matthews_corrcoef(val_y, val_pred >= thresh)
        val_prec = sklearn.metrics.precision_score(val_y, val_pred >= thresh)
        val_rec = sklearn.metrics.recall_score(val_y, val_pred >= thresh)
        
        train_scores.append([train_acc, train_bal_acc, train_roc_auc, train_avg_prec, train_mcc, train_prec, train_rec, fp_rate, fn_rate])
        val_scores.append(  [val_acc,val_bal_acc, val_roc_auc, val_avg_prec, val_mcc, val_prec, val_rec, fp_rate, fn_rate])
    
    train_avg_scores = np.array(train_scores).mean(axis=0).round(2)
    val_avg_scores = np.array(val_scores).mean(axis=0).round(2)
    
    train_std_scores = np.array(train_scores).std(axis=0).round(2)
    val_std_scores = np.array(val_scores).std(axis=0).round(2)
    
    for i, metric in enumerate(EVAL_METRICS):
        print()
        print(f'Average {metric}: {train_avg_scores[i]}+-{train_std_scores[i]} (training)  {val_avg_scores[i]}+-{val_std_scores[i]} (validation)')
       
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
    
    
