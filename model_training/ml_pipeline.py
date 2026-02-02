import tensorflow as tf
from models.baseModel import BaseModel as bm

class MLPipeline:
    def __init__(self, path_to_data):
        self.path_to_data = path_to_data
        self.models = []

    
    def add_model(self, model, focusmodels = True, bad_but_looks_good_equals_bad = None, **params):
        if focusmodels is None and bad_but_looks_good_equals_bad is None:
            pass # do all
        elif bad_but_looks_good_equals_bad is None:
            pass # do one focus and use both good bad* and bad bad*
        elif focusmodels is None:
            pass # do both focus and use specified bad_but_looks_good_equals_bad
        else:
            self.models.append([model, params, focusmodels, bad_but_looks_good_equals_bad])
            # params will be = {'epochs' = 10, 'batch_size' = 32, 'optimizer' = 'adam', 'loss' = 'sparse_categorical_crossentropy', 'metrics' = ['accuracy']}

    def fetch_data(self):
        pass

    def preprocess_data(self):
        pass

    def run_pipline (self):
        for model in self.models:
            bm
            pass

    

    