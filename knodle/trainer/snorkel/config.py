from knodle.trainer.config import MajorityConfig
from knodle.trainer.knn_denoising.config import KNNConfig


class SnorkelConfig(MajorityConfig):
    def __init__(
            self,
            label_model_num_epochs: int = 50,
            label_model_log_freq: int = 10,
            **kwargs
    ):
        super().__init__(**kwargs)
        self.label_model_num_epochs = label_model_num_epochs
        self.label_model_log_freq = label_model_log_freq


class SnorkelKNNConfig(SnorkelConfig, KNNConfig):
    def __init__(self, **kwargs):
        # use all config parameters in SnorkelConfig and KNNConfig
        super().__init__(**kwargs)