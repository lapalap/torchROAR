####################
# Data Abstract class
####################

from abc import ABC, abstractmethod
import torch
import numpy as np
import torchvision

class ROARdata(ABC):
    def __init__(self, transforms, device, n_degradation_steps = 10):

        self.transforms = transforms
        self.n_degradation_steps = n_degradation_steps
        self.size_degradation_steps = 1./n_degradation_steps
        self.data = None
        self.explanations = None
        self.device = device

        # self degradation status, starts at 1. -- no corruption, ends with 0. -- full corruption
        self.degradation_status = 1.


    def get_train_data(self):
        return self.data['train']

    def get_valid_data(self):
        return self.data['valid']


    def compute_explanations(self, model, method, **kwargs):
        # Batch size used for computing the explanations
        BATCH_SIZE = 256

        #TODO make this also distributed

        # Initalise the explainability method
        explainer = method

        # Load model into the explainer
        explainer.load_model(model)

        train_set = self.data['train']
        trainloader = torch.utils.data.DataLoader(train_set,
                                                  batch_size = BATCH_SIZE,
                                                  shuffle=False,
                                                  num_workers=16)

        for i, data in enumerate(trainloader, 0):
            # get the inputs; data is a list of [inputs, labels]
            inputs, labels = data[0].to(self.device), data[1].to(self.device)

            explanations = explainer.attribute(inputs, labels)
            self.explanations['train_explanations'][i * BATCH_SIZE : (i + 1 ) * BATCH_SIZE, ... ] = explanations.data

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





