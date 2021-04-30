from utils import *
from functools import partial
from torch import distributed as dist
import torch.optim as optim
import torchvision.transforms as transforms
from tqdm import tqdm
from torch.nn.parallel import DistributedDataParallel as DDP
import argparse
import pandas as pd

#local imports
from data import ROARcifar10
from explainer import Saliency
from models import ResNet9

# dataset = cifar10(root='./data/') # downloads dataset

def model_train(model,
                dataset,
                distributed,
                train_configs,
                device):

    if distributed:
        pass
        #Do a distributed training
        #
        # # We use pytorch's distributed package with NCCL for inter-gpu communication
        # # Define the process group
        # dist.init_process_group(
        #     backend='nccl',
        #     init_method='env://'
        # )
        #
        # # Set variables for the local worker to determine its rank and the world size
        # rank = dist.get_rank()
        # is_rank0 = rank == 0
        # world_size = dist.get_world_size()
        #
        # # Assign a device for this worker.
        # device = torch.device(
        #     "cuda:{}".format(rank) if torch.cuda.is_available() else "cpu"
        # )
        # torch.random.manual_seed(rank)
        # torch.backends.cudnn.benchmark = True
        #
        # # Wait for GPUS to be initialized
        # torch.cuda.synchronize()
        #
        # # Start timing all processes together
        # dist.barrier()
        # timer = Timer(synch=torch.cuda.synchronize)
        # # Copy the dataset to the GPUs
        # dataset = map_nested(to(device), dataset)
        # dist.barrier()
        # timer()
        # data_transfer_time = timer.total_time
        # if rank == 0:
        #     print(f"Uploaded data to GPUs {data_transfer_time:.3f}s")
        #
        # # Select a shard of the training dataset for this worker, and select all of the validation dataset
        # selector = list(range(rank, len(dataset['train']['data']), world_size))
        # dataset = {'train': {'data': dataset['train']['data'][selector], 'targets': dataset['train']['targets'][selector]},
        #            'valid': dataset['valid']}
        #
        # train_batches = partial(
        #     Batches,
        #     dataset=train_set,
        #     shuffle=True,
        #     drop_last=True,
        #     max_options=200,
        #     device=device
        # )
        #
        # valid_batches = partial(
        #     Batches,
        #     dataset=valid_set,
        #     shuffle=False,
        #     drop_last=False,
        #     device=device
        # )
        #
        # # Model to evaluate after the distributed model is trained
        # local_eval_model = Network(input_whitening_net, label_smoothing_loss(0.2)).half().to(device)
        #
        # # Distributed model to train by all workers
        # distributed_model = Network(input_whitening_net, label_smoothing_loss(0.2)).half().to(device)
        # is_bias = group_by_key(('bias' in k, v) for k, v in trainable_params(distributed_model).items())
        # loss = distributed_model.loss
        #
        # # Make sure all workers start timing here
        # dist.barrier()
        # timer = Timer(torch.cuda.synchronize)
        #
        # # Wrap with distributed data parallel, this introduces hooks to execute all-reduce upon back propagation
        # distributed_model = DDP(distributed_model, device_ids=[rank])
        #
        # if is_rank0:
        #     # Save the model in rank 0 to initialize all the others
        #     with open('initialized.model', 'wb') as f:
        #         torch.save(distributed_model.state_dict(), f)
        #
        # dist.barrier()
        # with open('initialized.model', 'rb') as f:
        #     distributed_model.load_state_dict(torch.load(f))
        #
        # # Data iterators
        # transforms = (Crop(32, 32), FlipLR())
        # tbatches = train_batches(batch_size, transforms)
        # train_batch_count = len(tbatches)
        # vbatches = valid_batches(batch_size)
        #
        # # Construct the learning rate, weight decay and momentum schedules.
        # opt_params = {'lr': lr_schedule(
        #     [0, epochs / warmup_fraction, epochs - ema_epochs],
        #     [0.0, lr_scaler * 1.0, lr_scaler * lr_end_fraction],
        #     batch_size, train_batch_count
        # ),
        #     'weight_decay': Const(5e-4 * lr_scaler * batch_size), 'momentum': Const(0.9)}
        #
        # opt_params_bias = {'lr': lr_schedule(
        #     [0, epochs / warmup_fraction, epochs - ema_epochs],
        #     [0.0, lr_scaler * 1.0 * 64, lr_scaler * lr_end_fraction * 64],
        #     batch_size, train_batch_count
        # ),
        #     'weight_decay': Const(5e-4 * lr_scaler * batch_size / 64), 'momentum': Const(0.9)}
        #
        # opt = SGDOpt(
        #     weight_param_schedule=opt_params,
        #     bias_param_schedule=opt_params_bias,
        #     weight_params=is_bias[False],
        #     bias_params=is_bias[True]
        # )
        #
        # # Train the network
        # distributed_model.train(True)
        # epochs_log = []
        # for epoch in range(epochs):
        #     activations_log = []
        #     for tb in tbatches:
        #         # Forward step
        #         out = loss(distributed_model(tb))
        #         distributed_model.zero_grad()
        #         out['loss'].sum().backward()
        #         opt.step()
        #
        #         # Log activations
        #         activations_log.append(('loss', out['loss'].detach()))
        #         activations_log.append(('acc', out['acc'].detach()))
        #
        #     # Compute the average over the activation logs for the last epoch
        #     res = map_values((lambda xs: to_numpy(torch.cat(xs)).astype(np.float)), group_by_key(activations_log))
        #     train_summary = mean_members(res)
        #     timer()
        #
        #     # Evaluate the model
        #     # Copy the weights to the local model
        #     model_dict = {k[7:]: v for k, v in distributed_model.state_dict().items()}
        #     local_eval_model.load_state_dict(model_dict)
        #     valid_summary = eval_on_batches(local_eval_model, loss, vbatches)
        #     timer(update_total=False)
        #     time_to_epoch_end = timer.total_time + data_transfer_time
        #     epochs_log.append(
        #         {
        #             'valid': valid_summary,
        #             'train': train_summary,
        #             'time': time_to_epoch_end
        #         }
        #     )
        #
        # # Wait until all models finished training
        # dist.barrier()
        # timer()

    else:
        #Standalone training

        train_set = dataset.get_train_data()
        valid_set = dataset.get_valid_data()

        trainloader = torch.utils.data.DataLoader(train_set, batch_size= train_configs['batch_size'],
                                                  shuffle=True, num_workers=16)
        valloader = torch.utils.data.DataLoader(valid_set, batch_size=train_configs['batch_size'],
                                                 shuffle=False, num_workers=16)

        criterion = train_configs['criterion']()
        optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.9)

        for epoch in tqdm(range(train_configs['epochs'])):  # loop over the dataset multiple times
            running_loss = 0.0
            for i, data in enumerate(trainloader, 0):
                # get the inputs; data is a list of [inputs, labels]
                inputs, labels = data[0].to(device), data[1].to(device)

                # zero the parameter gradients
                optimizer.zero_grad()

                # forward + backward + optimize
                outputs = model(inputs)
                print(outputs.shape)
                print(labels.shape)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()

    return model

def eval_model(model, dataset):
    #TODO implement eval
    pass

def eval_roar(xai_method,
              data_configs,
              roar_configs,
              model_configs,
              train_configs,
              device):

    # define model
    model = model_configs['model'](in_channels = data_configs['n_channels'],
                                   num_classes = data_configs['n_classes'])
    model.to(device)

    # get data class
    dataset = data_configs['dataset'](transforms = data_configs['transforms'],
                                      n_degradation_steps = roar_configs['degradation_steps'])

    #load data
    dataset.load_data()


    # train first time
    model = model_train(model = model,
                        dataset = dataset,
                        distributed = roar_configs['distributed'],
                        train_configs = train_configs,
                        device = device
                        )

    # explain train set
    dataset.compute_explanations(model = model,
                                 xai_method = xai_method
                                 )

    # iterate over degraded dataset
    out_dict = {}

    for counter, rate in enumerate(range(roar_configs['degradation_steps'])):
        dataset = dataset.degrade_data()
        out_dict[str(counter)] = {}
        for k in range(roar_configs['runs']):
            model = model_configs['model'](data_configs['n_channels'], data_configs['n_classes'])
            model = model_train(model, train_configs)

            break
            out_dict[str(counter)][str(k)] = eval_model(model, dataset)

        break


    results_df = pd.from_dict(out_dict)
    return results_df

if __name__ == '__main__':

    # We use pytorch's distributed package with NCCL for inter-gpu communication
    # Define the process group
    # dist.init_process_group(
    #     backend='nccl',
    #     init_method='env://'
    # )

    # Set variables for the local worker to determine its rank and the world size
    rank = dist.get_rank()
    is_rank0 = rank == 0
    world_size = dist.get_world_size()

    # Assign a device for this worker.
    device = torch.device(
        "cuda:{}".format(rank) if torch.cuda.is_available() else "cpu"
    )
    torch.random.manual_seed(rank)
    torch.backends.cudnn.benchmark = True

    data_configs = {'dataset' : ROARcifar10,
                    'n_channels': 3,
                    'n_classes': 10,
                    'width': 32,
                    'height': 32,
                    'transforms': transforms.ToTensor(),
                   }

    roar_configs = { 'xai_methods': [Saliency()],
                     'distributed': False,
                     'degradation_steps': 2,
                     'runs': 1
                   }

    model_configs = {'model': ResNet9
    }

    train_configs = {
                    'criterion': nn.CrossEntropyLoss,
                    'lr_scaler': 1.5,
                    'lr_scaler_end_fraction': 0.1,
                    'epochs': 20,
                    'warmup_fraction' : 5,
                    'ema_ep': 2,
                    'batch_size': 256,
                    'runs': 1
    }

    results_dict = {}

    for xai_method in roar_configs['xai_methods']:
        results_dict[xai_method.get_name()] = eval_roar(xai_method,
                                                  data_configs,
                                                  roar_configs,
                                                  model_configs,
                                                  train_configs,
                                                  device)








