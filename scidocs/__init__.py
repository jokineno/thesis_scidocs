from scidocs.classification import get_mag_mesh_metrics
from scidocs.user_activity_and_citations import get_view_cite_read_metrics


def get_wikidocs_metrics(data_paths,
                        classification_embeddings_path,
                        citations_embeddings_path,
                        val_or_test='test',
                        n_jobs=-1,
                        cuda_device=-1,
                        debug=False):
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
    wikidocs_metrics = {}
    wikidocs_metrics.update(get_mag_mesh_metrics(data_paths, classification_embeddings_path, val_or_test=val_or_test, n_jobs=n_jobs, debug=debug))
    wikidocs_metrics.update(get_view_cite_read_metrics(data_paths, citations_embeddings_path, val_or_test=val_or_test))
    print("============DONE============")
    return wikidocs_metrics
