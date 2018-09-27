import pandas as pd
import scipy as sci
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import numpy as np
from sklearn import metrics
from operator import itemgetter

def import_data():
    df_model_raw = pd.read_csv('outParticleInfectiveReportCasesCountSampled.csv',delimiter=',')

    df_model = df_model_raw.ix[:'c', :432]
    df_empirical_raw = pd.read_csv('empiricalData.csv',delimiter=',') #\t
    df_empirical = df_empirical_raw.iloc[:432]
    print(df_model.head())
    print(df_empirical.head())
    #get mean and std
    return df_model, df_empirical

def cal_mean_std(df_model, df_empirical):
    mean = df_empirical['count'].mean()
    #print(mean)
    std = df_empirical['count'].std()
    #print(std)
    return mean, std

def cal_threshold_breakout(mean, std, stdPara):
    threshold_breakout = mean + stdPara * std
    return threshold_breakout

def cal_empirical_breakout(df_empirical, threshold_breakout):
    arr_empirical = sci.array(df_empirical['count'])
    #print(arr_empirical)
    arr_empirical_breakout = [None] * len(arr_empirical)
    for i in range(len(arr_empirical)):
        if (arr_empirical[i] > threshold_breakout):
            arr_empirical_breakout[i] = 1
        else:
            arr_empirical_breakout[i] = 0
    print(arr_empirical_breakout)
    return arr_empirical_breakout

def cal_model_breakout(df_model,threshold_breakout, threshold_probability):
    print(threshold_probability)
    arr_model = sci.array(df_model)
    arr_model_breakout = [[0 for x in range(len(arr_model[0]))] for y in range(len(arr_model))]
    #print(sci.matrix(arr_model_breakout))
    #calculate breakout of the model data
    for i in range(len(arr_model)):
        for j in range(len(arr_model[0])):
            if (arr_model[i][j] > threshold_breakout):
                arr_model_breakout[i][j] = 1
            else:
                arr_model_breakout[i][j] = 0
    #print(arr_model_breakout)
    arr_prediction_count = [0] * len(arr_model[0])
    arr_prediction = [None] * len(arr_model[0])

    for i in range(len(arr_model)):
        for j in range(len(arr_model[0])):
            arr_prediction_count[j] = arr_prediction_count[j] + arr_model_breakout[i][j]

    #print(arr_prediction_count)

    for i in range(len(arr_prediction_count)):
        if (arr_prediction_count[i] >= threshold_probability * len(arr_model)):
            arr_prediction[i] = 1
        else:
            arr_prediction[i] = 0

    print(arr_prediction)
    return arr_prediction

def cal_confusion_matrix(arr_empirical_breakout, arr_prediction):
    cm = confusion_matrix(arr_empirical_breakout,arr_prediction)
    #print(len(arr_empirical_breakout))
    #print(len(arr_prediction))
    print(cm)
    '''plt.matshow(cm)
    plt.title('Confusion matrix')
    plt.colorbar()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.show()'''
    return cm

def cal_TPR_FPR(cm):
    TPR = cm[1][1]/(cm[1][0] + cm[1][1])
    FPR = cm[0][1]/(cm[0][0] + cm[0][1])
    return TPR, FPR

# x is the array of threshold of probability
def cal_array_TPR_FPR(x, arr_empirical_breakout,df_model,threshold_breakout):
    arr_TPR = [None] * len(x)
    arr_FPR = [None] * len(x)
    for i in range(len(x)):
        arr_prediction = cal_model_breakout(df_model,threshold_breakout, x[i])
        cm = cal_confusion_matrix(arr_empirical_breakout, arr_prediction)
        (arr_TPR[i], arr_FPR[i]) = cal_TPR_FPR(cm)
        print("arr_TPR[i]")
        print(arr_TPR[i])
        print("arr_FPR[i]")
        print(arr_FPR[i])
    #get unique arr_TPR, arr_FPR
    arr_TPR_FPR = sci.vstack((arr_TPR, arr_FPR)).T
    print(arr_TPR_FPR)
    arr_TPR_FPR_unique_raw = np.vstack({tuple(row) for row in arr_TPR_FPR})
    arr_TPR_FPR_unique = np.sort(arr_TPR_FPR_unique_raw, axis=0)
    print(arr_TPR_FPR[:, 0], arr_TPR_FPR[:, 1])
    print("arr_TPR_FPR_unique")
    print(arr_TPR_FPR_unique)
    return arr_TPR_FPR_unique[:, 0], arr_TPR_FPR_unique[:, 1]

def cal_AUC_ROC(arr_TPR, arr_FPR):
    auc = 0
    for i in range(len(arr_TPR)-1):
        auc = auc + arr_TPR[i] * (arr_FPR[i+1] - arr_FPR[i])
    return auc

def cal_median_model_data(df_model):
    arr_model = sci.array(df_model)
    arr_median = [None] * len(arr_model[0])
    for i in range(len(arr_model[0])):
        arr_median[i] = np.median(arr_model[:, i])
    print(arr_median)
    return arr_median

def cal_mean_model_data(df_model):
    arr_model = sci.array(df_model)
    arr_mean = [None] * len(arr_model[0])
    for i in range(len(arr_model[0])):
        arr_mean[i] = np.mean(arr_model[:, i])
    print(arr_mean)
    return arr_mean

def main_confusion_matrix():
    (df_model, df_empirical) = import_data()
    (mean, std) = cal_mean_std(df_model, df_empirical)
    stdPara = 1.5
    print('stdPara',stdPara)
    threshold_probability = 0.5
    threshold_breakout = cal_threshold_breakout(mean, std, stdPara)
    print('threshold_breakout',threshold_breakout)
    arr_empirical_breakout = cal_empirical_breakout(df_empirical, threshold_breakout)
    arr_prediction = cal_model_breakout(df_model,threshold_breakout, threshold_probability)
    cm = cal_confusion_matrix(arr_empirical_breakout, arr_prediction)

def main_plot_roc():
    (df_model, df_empirical) = import_data()
    (mean, std) = cal_mean_std(df_model, df_empirical)
    stdPara = 1.5
    print(stdPara)
    threshold_breakout = cal_threshold_breakout(mean, std, stdPara)
    arr_empirical_breakout = cal_empirical_breakout(df_empirical, threshold_breakout)
    x2 = sci.arange(0.1, 1.01, 0.02)
    x1 = sci.arange(0.01, 0.1, 0.02)
    x02 = sci.arange(0.001, 0.01, 0.0002)
    x01 = sci.arange(0.0001, 0.001, 0.0002)
    x00 = sci.arange(0, 0.0001, 0.00002)
    x = sci.concatenate((x00, x01, x02, x1, x2), axis=0)
    print(x)
    (arr_TPR, arr_FPR) = cal_array_TPR_FPR(x, arr_empirical_breakout,df_model,threshold_breakout)
    auc = cal_AUC_ROC(arr_TPR, arr_FPR)
    print("auc is: ")
    print(auc)
    plt.plot(arr_FPR,arr_TPR)
    plt.title('ROC Curve')
    plt.ylabel('TPR')
    plt.xlabel('FPR')
    plt.show()

def main_plot_median_mean():
    (df_model, df_empirical) = import_data()
    arr_median = cal_median_model_data(df_model)
    arr_mean = cal_mean_model_data(df_model)
    plot_median = plt.scatter(sci.array(df_empirical['count']), arr_median, c='green',marker='+',s=50)
    plot_mean = plt.scatter(sci.array(df_empirical['count']), arr_mean, c='red',marker='+',s=50)
    plt.legend((plot_median, plot_mean) ,('median', 'mean'),loc='upper left')
    z_mean = np.polyfit(sci.array(df_empirical['count']), arr_mean, 1)
    z_median = np.polyfit(sci.array(df_empirical['count']), arr_median, 1)
    p_mean = np.poly1d(z_mean)
    p_median = np.poly1d(z_median)
    plt.plot(sci.array(df_empirical['count']),p_mean(sci.array(df_empirical['count'])),"r--")
    plt.plot(sci.array(df_empirical['count']),p_median(sci.array(df_empirical['count'])),"g--")
    print("y_mean=%.6fx+%.6f"%(z_mean[0],z_mean[1]))
    print("y_median=%.6fx+%.6f"%(z_median[0],z_median[1]))
    plt.ylabel('model results of each month')
    plt.xlabel('empirical data of each month')
    plt.show()

def plot_boxplot():
    (df_model, df_empirical) = import_data()
    arr_model = sci.array(df_model)
    arr_empirical = sci.array(df_empirical['count'])
    #plt.boxplot(arr_model[:, :10], '', positions=arr_empirical[:10])
    #plt.boxplot(arr_model, '', positions=arr_empirical)
    arr_total = sci.vstack((arr_empirical, arr_model)).T
    arr_total_ordered = arr_total # correct a bug here
    print(arr_total_ordered)
    arr_y = arr_total_ordered[:, 1:]
    arr_x = arr_total_ordered[:, 0]
    print(arr_x)
    print(arr_y)
    plt.boxplot(arr_y.T, 'o', positions=arr_x, widths=20)
    #plt.boxplot(arr_y.T, 'o', widths=30)
    plt.ylabel('model results')
    plt.xlabel('empirical data')
    print(min(arr_x))
    print(max(arr_x))
    plt.xticks(np.linspace(min(arr_x), max(arr_x)+1, num=1))
    print(np.linspace(min(arr_x), max(arr_x)+1, num=10))
    #plt.locator_params(nbins=100)
    plt.ylim((0, 7000))
    plt.xlim((0, 2300))
    plt.show()



main_confusion_matrix()
#main_plot_roc()
#main_plot_median_mean()
#plot_boxplot()




