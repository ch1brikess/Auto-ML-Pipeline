from scr.core.base_pipeline import BaseMLPipeline

class ClassifierPipeline(BaseMLPipeline):
    def __init__(self, algorithm, target_column, save_model, output_columns=None):
        super().__init__(algorithm, 'classification', target_column, save_model, output_columns)