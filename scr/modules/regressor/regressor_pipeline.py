from scr.core.base_pipeline import BaseMLPipeline

class RegressorPipeline(BaseMLPipeline):
    def __init__(self, algorithm, target_column, save_model, output_columns=None):
        super().__init__(algorithm, 'regression', target_column, save_model, output_columns)