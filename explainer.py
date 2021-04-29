####################
# Explainers
####################

from abc import ABC, abstractmethod

class Explainer(ABC):
    @abstractmethod
    def __init__(self, model, **kwargs):
        pass

    @abstractmethod
    def attribute(self, batch):
        pass

class Saliency(Explainer):
    def __init__(self, model):
        self.model = model
        pass

    def attribute(self, batch):
        pass
