import os

PROJECT_ROOT_PATH = os.path.abspath(os.path.join(os.getcwd()))
    
class ThesisDataPaths:
    def __init__(self, base_path=None):
        if base_path is None:
            base_path = os.path.join(PROJECT_ROOT_PATH, 'thesis_data')
        self.base_path = base_path


        # =======================THESIS SOURCES START=================
        # THESIS CITATION
        self.cite_val = os.path.join(base_path, 'cite', 'val.qrel')
        self.cite_test = os.path.join(base_path, 'cite', 'test.qrel')

        # THESIS WIKI CLASS CLASSIFICATION - only train and test batches 70-30
        self.wiki_cls_train = os.path.join(base_path, 'wiki_cls', "top", 'train.csv')
        self.wiki_cls_test = os.path.join(base_path, 'wiki_cls', "top", 'test.csv')
        self.wiki_cls_train_second = os.path.join(base_path, 'wiki_cls', "second", 'train.csv')
        self.wiki_cls_test_second = os.path.join(base_path, 'wiki_cls', "second", 'test.csv')

        # =======================THESIS SOURCES END=================