import json

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pydot
import seaborn as sns
import sklearn.tree as tree
from pylab import rcParams
from sklearn import metrics
from sklearn.ensemble import AdaBoostClassifier, RandomForestClassifier
from sklearn.feature_selection import RFECV, SelectFromModel
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import (StratifiedKFold, cross_validate,
                                     train_test_split)
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.utils.multiclass import unique_labels
from treeinterpreter import treeinterpreter as ti

import data_preparation
import feature_selection


def classify_group(dataset, feature, groups):
    k_folds = 3
    if len(groups) > 0:
        if len(groups) == 1:
            print('Single group encoding...')
            dataset[feature] = dataset[feature].apply(
                lambda l: True if l == groups[0] else False)
        else:
            print('Multi-group encoding...')
            dataset = dataset[dataset[feature].isin(groups)]
            dataset[feature] = data_preparation.encode_labels(dataset[feature])
    else:
        print('Multi-class encoding...')
        groupnames = dataset[feature].value_counts().to_frame('counts')
        groupnames = groupnames[groupnames['counts'] > 30]
        dataset = dataset[dataset[feature].isin(groupnames.index)]
        dataset[feature] = data_preparation.encode_labels(dataset[feature])
    X = dataset.drop(feature, axis=1)
    y = dataset[feature]
    print('Actual number of incidents included in current dataset: ' + str(len(X)))
    print('Number of classes involved: ' + str(y.nunique()))
    clf = RandomForestClassifier(
        class_weight='balanced', n_jobs=-1, n_estimators=100)
    # X = select_features_rfecv(X, y, clf, k_folds)
    cross_validate_multiple_model_types(X, y, [clf], k_folds)


def select_features_rfecv(X, y, clf, k_folds):
    print('Engaging recursive feature elimination procedure...')
    rfecv = RFECV(estimator=clf, step=1, cv=StratifiedKFold(
        k_folds), scoring='f1_macro', n_jobs=-1)
    rfecv.fit(X, y)
    print("Optimal number of features : %d" % rfecv.n_features_)
    # Plot number of features VS. cross-validation scores
    plt.figure()
    plt.xlabel("Number of features selected")
    plt.ylabel("Cross validation score")
    plt.plot(range(1, len(rfecv.grid_scores_) + 1), rfecv.grid_scores_)
    plt.show()
    mask = list(rfecv.support_)
    X = X[X.columns[mask]]
    return X


def plot_feature_importance(X, y, clf):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
    y_pred = clf.fit(X_train, y_train).predict(X_test)
    np.set_printoptions(precision=2)
    classes = np.unique(unique_labels(y_pred, y_test))
    if len(classes) < 7:
        plot_confusion_matrix(
            y_test, y_pred, classes=classes, title='Confusion matrix')
    # if len(classes) == 2:
    #     plot_roc_curve(X_test, y_test, clf)
    feature_imp = pd.Series(clf.feature_importances_, index=X.columns).sort_values(
        ascending=False).head(20)
    rcParams['figure.figsize'] = 10, 15
    sns.barplot(x=feature_imp, y=feature_imp.index)
    plt.xlabel('Feature importance score')
    plt.ylabel('Features')
    plt.title("Visualising important features")
    plt.show()
    if (clf.__class__ == DecisionTreeClassifier):
        dot_data = StringIO()
        tree.export_graphviz(clf, out_file=dot_data)
        graph = pydot.graph_from_dot_data(dot_data.getvalue())
        graph[0].write_pdf("../../tree.pdf")
    return feature_imp


def plot_roc_curve(X_test, y_test, clf):
    y_pred_proba = clf.predict_proba(X_test)
    fpr, tpr, _ = metrics.roc_curve(y_test,  y_pred_proba)
    auc = metrics.roc_auc_score(y_test, y_pred_proba)
    plt.plot(fpr, tpr, label="data 1, auc=" + str(auc))
    plt.legend(loc=4)
    plt.show()


def plot_confusion_matrix(y_true, y_pred, classes, title=None, cmap=plt.cm.Blues, normalise=True):
    # Compute confusion matrix
    y_true = data_preparation.encode_labels(y_true.tolist())
    y_pred = data_preparation.encode_labels(y_pred.tolist())
    cm = confusion_matrix(y_true, y_pred)
    # Only use the labels that appear in the data
    classes = classes[unique_labels(y_true, y_pred)]
    if normalise:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')
    fig, ax = plt.subplots(figsize=(5, 5))
    im = ax.imshow(cm, interpolation='nearest', cmap=cmap)
    ax.figure.colorbar(im, ax=ax)
    # We want to show all ticks...
    ax.set(xticks=np.arange(cm.shape[1]),
           yticks=np.arange(cm.shape[0]),
           # ... and label them with the respective list entries
           xticklabels=classes, yticklabels=classes,
           title=title,
           ylabel='True label',
           xlabel='Predicted label')

    # Rotate the tick labels and set their alignment.
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right",
             rotation_mode="anchor")

    # Loop over data dimensions and create text annotations.
    fmt = '.2f' if normalise else 'd'
    thresh = cm.max() / 2.
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, format(cm[i, j], fmt),
                    ha="center", va="center",
                    color="white" if cm[i, j] > thresh else "black")
    fig.tight_layout()
    plt.show()


def cross_validate_multiple_model_types(X, y, classifiers, k_folds):
    # classifiers = [
    # DecisionTreeClassifier(),
    # RandomForestClassifier(max_depth=10),
    # AdaBoostClassifier(),
    # KNeighborsClassifier(),
    # MLPClassifier(),
    # GaussianNB(),
    # SVC(),
    # ]
    for clf in classifiers:
        print('Cross-validating (using', k_folds, 'folds): ' + str(clf))
        if y.nunique() > 2:
            print('Multi-class')
            cv_scores = cross_validate(clf, X, y, cv=StratifiedKFold(k_folds), scoring=(
                'accuracy', 'precision_macro', 'recall_macro', 'f1_macro', 'f1_weighted'))
        else:
            print('Binary')
            cv_scores = cross_validate(clf, X, y, cv=StratifiedKFold(k_folds), scoring=(
                'accuracy', 'precision', 'recall', 'f1_macro', 'f1_weighted', 'roc_auc'))
        print('Results: ', json.dumps(
            cv_scores, indent=4, sort_keys=True, default=str))
        print('Average results: ', json.dumps({key: sum(cv_scores[key] / k_folds)
                                               for key in cv_scores}, indent=4, sort_keys=True, default=str))

        if clf.__class__ in [DecisionTreeClassifier, RandomForestClassifier, AdaBoostClassifier]:
            feature_imp = plot_feature_importance(X, y, clf)
        #     # get_interpretation(X, clf)
        #     # feature_imp = feature_selection(X, y, clf)
            return feature_imp


def get_interpretation(X, rf_model):
    for i, row in X.iterrows():
        data_point = pd.DataFrame([row])
        # Once transposed, it will be the column name
        data_point.set_axis(['value_variable'])
        prediction, bias, contributions = ti.predict(rf_model, data_point)
        local_interpretation = data_point.append(pd.DataFrame([[round(c[1], 3) for c in contributions[0]]], columns=data_point.columns.tolist(), index=[
                                                 'contribution_variable'])).T.sort_values('contribution_variable', ascending=False)
        print(local_interpretation)
