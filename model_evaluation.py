import matplotlib.pyplot as plt


def plot_roc_curve(model, X_test, y_test, X_train=None, y_train=None):
    ```
        Plot the ROC curve of a model 
    ```
    # Create matplotlib figure
    plt.figure(figsize=(10,8))
    plt.title('Receiver Operating Characteristic')
    plt.legend(loc = 'lower right')
    plt.plot([0, 1], [0, 1], r--')
    plt.xlim([0, 1])
    plt.ylim([0, 1])
    plt.ylabel('True Positive Rate')
    plt.xlabel('False Positive Rate')  
    
    # Plot Test set
    fpr, tpr, threshold, roc_auc = roc_metrics(model, X_test, y_test)
    plt.plot(fpr, tpr, label = 'AUC (test) : %0.2f' % roc_auc)

    if X_train is not None and y_train is not None:        
        # Plot Train
        fpr, tpr, threshold, roc_auc = roc_metrics(model, X_train, y_train)
        plt.plot(fpr, tpr, label = 'AUC (train) : %0.2f' % roc_auc)
        
    plt.show()

def roc_metrics(model, X, y):
    probs = model.predict_proba(X_test)
    preds = probs[:,1]
    fpr, tpr, threshold = metrics.roc_curve(y_test, preds)
    roc_auc = metrics.auc(fpr, tpr)
    
    return fpr, tpr, threshold, roc_auc
    
# def plot_roc_curve_cross_val(model, X, y):
#     i = 0
#     for train, test in cv.split(X, y):
#         probas_ = classifier.fit(X[train], y[train]).predict_proba(X[test])
#         # Compute ROC curve and area the curve
#         fpr, tpr, thresholds = roc_curve(y[test], probas_[:, 1])
#         tprs.append(interp(mean_fpr, fpr, tpr))
#         tprs[-1][0] = 0.0
#         roc_auc = auc(fpr, tpr)
#         aucs.append(roc_auc)
#         plt.plot(fpr, tpr, lw=1, alpha=0.3,
#                  label='ROC fold %d (AUC = %0.2f)' % (i, roc_auc))

#         i += 1
#         plt.plot([0, 1], [0, 1], linestyle='--', lw=2, color='r',
#                  label='Chance', alpha=.8)

#         mean_tpr = np.mean(tprs, axis=0)
#         mean_tpr[-1] = 1.0
#         mean_auc = auc(mean_fpr, mean_tpr)
#         std_auc = np.std(aucs)
#         plt.plot(mean_fpr, mean_tpr, color='b',
#                  label=r'Mean ROC (AUC = %0.2f $\pm$ %0.2f)' % (mean_auc, std_auc),
#                  lw=2, alpha=.8)

#         std_tpr = np.std(tprs, axis=0)
#         tprs_upper = np.minimum(mean_tpr + std_tpr, 1)
#         tprs_lower = np.maximum(mean_tpr - std_tpr, 0)
#         plt.fill_between(mean_fpr, tprs_lower, tprs_upper, color='grey', alpha=.2,
#                          label=r'$\pm$ 1 std. dev.')

#         plt.xlim([-0.05, 1.05])
#         plt.ylim([-0.05, 1.05])
#         plt.xlabel('False Positive Rate')
#         plt.ylabel('True Positive Rate')
#         plt.title('Receiver operating characteristic example')
#         plt.legend(loc="lower right")
#         plt.show()