####################
# Explainers
####################

from abc import ABC, abstractmethod

class Explainer(ABC):
    @abstractmethod
    def attribute(self, batch):
        pass

    def load_model(self, model):
        self.model = model
        pass

    def get_name(self):
        return self.name

class Saliency(Explainer):
    def __init__(self):
        pass


    def get_name(self):
        return 'Saliency'

    def attribute(self, batch):
        pass
