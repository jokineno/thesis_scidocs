from scidocs.classification import get_wiki_class_metrics
from scidocs.user_activity_and_citations import get_wiki_citations_metrics



def get_wikidocs_metrics(data_paths,
                        classification_embeddings_path,
                        classification_embeddings_path_second,
                        citations_embeddings_path,
                        val_or_test='test',
                        n_jobs=-1,
                        cuda_device=-1):
    """This is the master wrapper that computes the SciDocs metrics given
    three embedding files (jsonl) and some optional parameters.

    Arguments:
        data_paths {scidocs.DataPaths} -- A DataPaths objects that points to
                                          all of the SciDocs files
        classification_embeddings_path {str} -- Path to the embeddings jsonl
                                                for MAG and MeSH tasks
        user_activity_and_citations_embeddings_path {str} -- Path to the embeddings jsonl
                                                             for cocite, cite, coread, coview
        recomm_embeddings_path {str} -- Path to the embeddings jsonl for the recomm task
        n_jobs -- number of parallel jobs for classification related tasks (-1 to use all cpus)
        cuda_device -- cuda device for the recommender model (default is -1, meaning use CPU)

    Keyword Arguments:
        val_or_test {str} -- Whether to return metrics on validation set (to tune hyperparams)
                             or the test set which is what's reported in SPECTER paper
                             (default: 'test')
        cuda_device {int} -- For the recomm pytorch model -> which cuda device to use(default: -1)

    Returns:
        scidocs_metrics {dict} -- SciDocs metrics for all tasks
    """
    assert val_or_test in ('val', 'test'), "The val_or_test parameter must be one of 'val' or 'test'"
    print("===Fetching wikidocs metrics==")
    print("===Running {}===".format(val_or_test))
    wikidocs_metrics = {}

    if classification_embeddings_path:
        wikidocs_metrics.update(get_wiki_class_metrics(data_paths, classification_embeddings_path, val_or_test=val_or_test, n_jobs=n_jobs, task="top"))
    else:
        print("No classifications embeddings path. Not running classification task")

    if classification_embeddings_path_second:
        wikidocs_metrics.update(get_wiki_class_metrics(data_paths, classification_embeddings_path_second, val_or_test=val_or_test, n_jobs=n_jobs, task="second"))

    if citations_embeddings_path:
        wikidocs_metrics.update(get_wiki_citations_metrics(data_paths, citations_embeddings_path, val_or_test=val_or_test))
    else:
        print("No citations embeddings path. Not running citations task")

    print("============DONE============")
    return wikidocs_metrics
