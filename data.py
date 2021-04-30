####################
# Data Abstract class
####################

from abc import ABC, abstractmethod
import torch
import numpy as np
import torchvision

class ROARdata(ABC):
    def __init__(self, transforms, n_degradation_steps = 10):

        self.transforms = transforms
        self.n_degradation_steps = n_degradation_steps
        self.size_degradation_steps = 1./n_degradation_steps
        self.data = None
        self.explanations = None

        # self degradation status, starts at 1. -- no corruption, ends with 0. -- full corruption
        self.degradation_status = 1.


    def get_train_data(self):
        return self.data['train']

    def get_valid_data(self):
        return self.data['valid']


    def compute_explanations(self, model, method, **kwargs):
        #TODO make this also distributed

        # Initalise the explainability method
        explainer = method(net)

        # Make batches of training data
        train_batches = partial(
            Batches,
            dataset=self.data['train'],
            shuffle=False,
            drop_last=True,
            max_options=200,
            device=device
        )

        transforms = (Crop(32, 32), FlipLR())
        tbatches = train_batches(batch_size, transforms)
        train_batch_count = len(tbatches)

        # Explain batches with explainer
        for count, batch in enumerate(tbatches):
            explanations = explainer.attribute(batch)
            self.explanations['train_explanations']['data'][count * batch_size : (count + 1 ) * batch_size, ... ] = \
            explanations.data

    def degrade_data(self, **kwargs):
        #TODO do this distributively
        for count, datapoint in enumerate(self.data['train']['data']):
            explanation = self.explanations['train_explanations']['data'][count]
        #TODO please check if explanation is an explanation for that datapoint....
            datapoint = _degrade_image(datapoint,
                                       explanation,
                                       self.degradation_status,
                                       self.size_degradation_steps)

    def _degrade_image(image,
                       explanation,
                       degradation_status,
                       size_degradation_steps):
    #TODO do that
        pass

    @abstractmethod
    def load_data(self, **kwargs):
        pass

class ROARcifar10(ROARdata):
    def load_data(self, root = './data/', **kwargs):
        download = lambda train: torchvision.datasets.CIFAR10(root=root,
                                                              train=train,
                                                              download=True,
                                                              transform= self.transforms)
        self.data = {k: v for k, v in [('train', download(True)), ('valid', download(False))]}

        # Explanations only for train data and should have same number of channels
        self.explanations = {'train_explanations': torch.zeros(self.data['train'].data.shape)
                             }





