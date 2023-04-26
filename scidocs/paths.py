import os


PROJECT_ROOT_PATH = os.path.abspath(os.path.join(os.getcwd()))
    
    
class DataPaths:
    def __init__(self, base_path=None):
        if base_path is None:
            base_path = os.path.join(PROJECT_ROOT_PATH, 'data')
        self.base_path = base_path
        
        self.cite_val = os.path.join(base_path, 'cite', 'val.qrel')
        self.cite_test = os.path.join(base_path, 'cite', 'test.qrel')
        
        self.cocite_val = os.path.join(base_path, 'cocite', 'val.qrel')
        self.cocite_test = os.path.join(base_path, 'cocite', 'test.qrel')

        self.coread_val = os.path.join(base_path, 'coread', 'val.qrel')
        self.coread_test = os.path.join(base_path, 'coread', 'test.qrel')
        
        self.coview_val = os.path.join(base_path, 'coview', 'val.qrel')
        self.coview_test = os.path.join(base_path, 'coview', 'test.qrel')
        
        self.mag_train = os.path.join(base_path, 'mag', 'train.csv')
        self.mag_val = os.path.join(base_path, 'mag', 'val.csv')
        self.mag_test = os.path.join(base_path, 'mag', 'test.csv')
        
        self.mesh_train = os.path.join(base_path, 'mesh', 'train.csv')
        self.mesh_val = os.path.join(base_path, 'mesh', 'val.csv')
        self.mesh_test = os.path.join(base_path, 'mesh', 'test.csv')

        self.paper_metadata_view_cite_read = os.path.join(base_path, 'paper_metadata_view_cite_read.json')
        self.paper_metadata_mag_mesh = os.path.join(base_path, 'paper_metadata_mag_mesh.json')

        """
        # =======================THESIS SOURCES START=================
        self.thesis_paper_metadata_cross_link = os.path.join(base_path, 'thesis_paper_metadata_cite.json')
        self.thesis_paper_metadata_wikiclass = os.path.join(base_path, 'thesis_paper_metadata_wikiclass.json')

        # THESIS DIRECT CITATION
        self.thesis_direct_link_val = os.path.join(base_path, 'thesis_direct_link', 'val.qrel')
        self.thesis_direct_link_test = os.path.join(base_path, 'thesis_direct_link', 'test.qrel')

        # THESIS WIKI CLASS CLASSIFICATION
        self.wikicls_train = os.path.join(base_path, 'thesis_wikiclass_classification', 'train.csv')
        self.wikicls_val = os.path.join(base_path, 'thesis_wikiclass_classification', 'val.csv')
        self.wikicls_test = os.path.join(base_path, 'thesis_wikiclass_classification', 'test.csv')

        # =======================THESIS SOURCES END=================
        """