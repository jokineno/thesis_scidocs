import json
import pandas as pd
import numpy as np
from collections import defaultdict
from sklearn.metrics import f1_score, accuracy_score
from sklearn.model_selection import GridSearchCV
from lightning.classification import LinearSVC
from scidocs.embeddings import load_embeddings_from_jsonl
from sklearn.preprocessing import LabelEncoder
import pickle
np.random.seed(1)

def get_wiki_class_metrics(data_paths, embeddings_path=None, val_or_test='test', n_jobs=1, task="top"):
    """Run MAG and MeSH tasks.

    Arguments:
        data_paths {scidocs.paths.DataPaths} -- A DataPaths objects that points to
                                                all of the SciDocs files

    Keyword Arguments:
        embeddings_path {str} -- Path to the embeddings jsonl (default: {None})
        val_or_test {str} -- Whether to return metrics on validation set (to tune hyperparams)
                             or the test set (what's reported in SPECTER paper)

    Returns:
        metrics {dict} -- F1 score for both tasks.
    """
    assert val_or_test in ('val', 'test'), "The val_or_test parameter must be one of 'val' or 'test'"
    assert task in ("top", "second")
    embeddings = load_embeddings_from_jsonl(embeddings_path)
    if task=="top":
        X, y = get_X_y_for_classification(embeddings, data_paths.wiki_cls_train, data_paths.wiki_cls_test)
    elif task=="second":
        X, y = get_X_y_for_classification(embeddings, data_paths.wiki_cls_train_second, data_paths.wiki_cls_test_second)
    else:
        raise Exception("Invalid task...")
    
    f1, accuracy = classify(X['train'], y['train'], X["test"], y["test"], n_jobs=n_jobs)

    return {f'wiki_class_{task}': {'f1': f1, "accuracy": accuracy}}


def classify(X_train, y_train, X_test, y_test, n_jobs=1, task="top"):
    """
    Simple classification methods using sklearn framework.
    Selection of C happens inside of X_train, y_train via
    cross-validation. 
    
    Arguments:
        X_train, y_train -- training data
        X_test, y_test -- test data to evaluate on (can also be validation data)

    Returns: 
        F1 on X_test, y_test (out of 100), rounded to two decimal places
    """
    estimator = LinearSVC(loss="squared_hinge", random_state=123)
    print("[*] ============Using ***REAL*** Cross Validation==============")
    Cs = np.logspace(-4, 2, 7)
    svm = GridSearchCV(estimator=estimator, cv=3, param_grid={'C': Cs}, verbose=1, n_jobs=n_jobs)

    svm.fit(X_train, y_train)
    preds = svm.predict(X_test)

    f1 = np.round(100 * f1_score(y_test, preds, average='macro'), 2)
    acc = np.round(100 * accuracy_score(y_test, preds), 2)
    print("f1_score: {}".format(f1))
    print("accuracy: {}".format(acc))
    return f1, acc


def get_X_y_for_classification(embeddings, train_path, test_path):
    """
    Given the directory with train/test/val files for mesh classification
        and embeddings, return data as X, y pair
        
    Arguments:
        embeddings: embedding dict
        mesh_dir: directory where the mesh ids/labels are stored
        dim: dimensionality of embeddings
x
    Returns:"
        X, y: dictionaries of training X and training y
              with keys: 'train', 'val', 'test'
    """
    print("[*] Building train/val/test samples to targets dataset for classification...")
    dim = len(next(iter(embeddings.values())))
    column_types = {"title": str, "page_id": str, "class_name": str, "class_id": int}
    train = pd.read_csv(train_path, sep=";", dtype=column_types)
    test = pd.read_csv(test_path, sep=";", dtype=column_types)
    

    train = train[['page_id', "class_id"]]
    test = test[['page_id', "class_id"]]
    
    print("[*] Shapes of train/test.csv train: {}, test: {}".format(train.shape, test.shape))
    X = defaultdict(list)
    y = defaultdict(list)
    missing_ids = {"train": 0, "test": 0}
    for dataset_name, dataset in zip(['train', 'test'], [train, test]):
        for s2id, class_label in dataset.values:
            if s2id not in embeddings:
                print("[*] s2id ({}) not in embeddings".format(dataset_name))
                X[dataset_name].append(np.zeros(dim))
                missing_ids[dataset_name] += 1
            else:
                X[dataset_name].append(embeddings[s2id])
            y[dataset_name].append(class_label)
        X[dataset_name] = np.array(X[dataset_name])
        y[dataset_name] = np.array(y[dataset_name])

    print("[*] train/test not in embeddings count: {}.".format(missing_ids))
    return X, y
