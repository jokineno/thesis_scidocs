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

def get_wiki_class_metrics(data_paths, embeddings_path=None, val_or_test='test', n_jobs=1, debug=False):
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

    embeddings = load_embeddings_from_jsonl(embeddings_path)

    X, y = get_X_y_for_classification(embeddings, data_paths.wiki_cls_train, data_paths.wiki_cls_val, data_paths.wiki_cls_test)
    f1, accuracy = classify(X['train'], y['train'], X[val_or_test], y[val_or_test], n_jobs=n_jobs, debug=debug)

    return {'wiki_class': {'f1': f1, "accuracy": accuracy}}


def classify(X_train, y_train, X_test, y_test, n_jobs=1, debug=False):
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
    estimator = LinearSVC(loss="squared_hinge", random_state=42)
    print("\n") # For making stdout nice
    if debug:
        # TODO debug and true gives the same results
        print("[*] DEBUG ============Using ***debug*** Cross Validation============== DEBUG")
        Cs = np.logspace(-4, 2, 1)
        svm = GridSearchCV(estimator=estimator, cv=2, param_grid={'C': Cs}, verbose=1, n_jobs=n_jobs)
    else:
        print("[*] ============Using ***REAL*** Cross Validation==============")
        Cs = np.logspace(-4, 2, 7)
        svm = GridSearchCV(estimator=estimator, cv=3, param_grid={'C': Cs}, verbose=1, n_jobs=n_jobs)

    svm.fit(X_train, y_train)
    preds = svm.predict(X_test)

    # read label encoder from label_encoder.pkl
    with open("label_encoder.pkl", "rb") as f:
        print("Loading label encoder from label_encoder.pkl...")
        le = pickle.load(f)
    result_table = pd.DataFrame({"y_test": y_test,
                                 "y_test_desc": le.inverse_transform(y_test),
                                 "preds": preds,
                                 "preds_desc": le.inverse_transform(preds),
                                 "correct": y_test == preds})

    print("correct / all", result_table.correct.sum(),"/", len(result_table), result_table.correct.sum() / len(result_table))
    print("[*] Saving in to result_table.csv...")
    result_table.to_csv("result_table.csv", sep=";", index=False)


    f1 = np.round(100 * f1_score(y_test, preds, average='macro'), 2)
    acc = np.round(100 * accuracy_score(y_test, preds), 2)
    print("f1_score: {}".format(f1))
    print("accuracy: {}".format(acc))
    return f1, acc


def get_X_y_for_classification(embeddings, train_path, val_path, test_path):
    """
    Given the directory with train/test/val files for mesh classification
        and embeddings, return data as X, y pair
        
    Arguments:
        embeddings: embedding dict
        mesh_dir: directory where the mesh ids/labels are stored
        dim: dimensionality of embeddings

    Returns:"
        X, y: dictionaries of training X and training y
              with keys: 'train', 'val', 'test'
    """
    print("[*] Building train/val/test samples to targets dataset for classification...")
    dim = len(next(iter(embeddings.values())))
    column_types = {"title": str, "page_id": str, "class_name": str, "class_id": str}
    train = pd.read_csv(train_path, sep=";", dtype=column_types)
    val = pd.read_csv(val_path, sep=";", dtype=column_types)
    test = pd.read_csv(test_path, sep=";", dtype=column_types)


    le = LabelEncoder()
    le.fit(train['class_id'])
    train['class_label'] = le.transform(train['class_id'])
    test['class_label'] = le.transform(test['class_id'])
    val['class_label'] = le.transform(val['class_id'])
    with open("label_encoder.pkl", "wb") as f:
        print("Saving label_encoder.pkl")
        pickle.dump(le, f)


    train = train[['page_id', "class_label"]]
    test = test[['page_id', "class_label"]]
    val = val[['page_id', "class_label"]]

    print("[*] Shapes of train/test/val.csv train: {}, test: {}, val: {}".format(train.shape, test.shape, val.shape))
    X = defaultdict(list)
    y = defaultdict(list)
    missing_ids = {"train": 0, "val": 0, "test": 0}
    for dataset_name, dataset in zip(['train', 'val', 'test'], [train, val, test]):
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

    print("[*] train/test/val not in embeddings count: {}.".format(missing_ids))
    return X, y
