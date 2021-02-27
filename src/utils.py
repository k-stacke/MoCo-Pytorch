# -*- coding: utf-8 -*-
import os
import logging
import numpy as np
import time
import random
import pandas as pd
import lmdb

import torch
from torch.utils.data import Dataset
import torch.nn as nn
from torchvision import transforms

from PIL import Image, ImageFilter
from model.simclr_model import Net

def load_simclr(args):
    model = Net(args)
    for param in model.f.parameters():
        param.requires_grad = False
    return model

class GaussianBlur(object):
    """Gaussian blur augmentation: https://github.com/facebookresearch/moco/"""

    def __init__(self, sigma=[.1, 2.]):
        self.sigma = sigma

    def __call__(self, x):
        sigma = random.uniform(self.sigma[0], self.sigma[1])
        x = x.filter(ImageFilter.GaussianBlur(radius=sigma))
        return x

class FixedRandomRotation:
    """Rotate by one of the given angles."""
    def __init__(self, angles):
        self.angles = angles

    def __call__(self, x):
        angle = random.choice(self.angles)
        return transforms.functional.rotate(x, angle)


def load_moco(base_encoder, args):
    """ Loads the pre-trained MoCo model parameters.

        Applies the loaded pre-trained params to the base encoder used in Linear Evaluation,
         freezing all layers except the Linear Evaluation layer/s.

    Args:
        base_encoder (model): Randomly Initialised base_encoder.

        args (dict): Program arguments/commandline arguments.
    Returns:
        base_encoder (model): Initialised base_encoder with parameters from the MoCo query_encoder.
    """
    print("\n\nLoading the model: {}\n\n".format(args.load_checkpoint_dir))

    # Load the pretrained model
    checkpoint = torch.load(args.load_checkpoint_dir, map_location='cuda')

    # rename moco pre-trained keys
    state_dict = checkpoint['moco']
    for k in list(state_dict.keys()):
        # retain only encoder_q up to before the embedding layer
        if k.startswith('module.encoder_q') and not k.startswith('module.encoder_q.fc'):
            # remove prefix
            state_dict[f'module.{k[len("module.encoder_q."):]}'] = state_dict[k]
        # delete renamed or unused k
        del state_dict[k]

    print('Loading state dict with the keys: ', list(state_dict.keys()))
    # Load the encoder parameters
    base_encoder.load_state_dict(state_dict, strict=False)

    return base_encoder


def load_sup(base_encoder, args):
    """ Loads the pre-trained supervised model parameters.

        Applies the loaded pre-trained params to the base encoder used in Linear Evaluation,
         freezing all layers except the Linear Evaluation layer/s.

    Args:
        base_encoder (model): Randomly Initialised base_encoder.

        args (dict): Program arguments/commandline arguments.
    Returns:
        base_encoder (model): Initialised base_encoder with parameters from the supervised base_encoder.
    """
    print("\n\nLoading the model: {}\n\n".format(args.load_checkpoint_dir))

    # Load the pretrained model
    checkpoint = torch.load(args.load_checkpoint_dir)

    # Load the encoder parameters
    base_encoder.load_state_dict(checkpoint['encoder'])

    # freeze all layers but the last fc
    for name, param in base_encoder.named_parameters():
        if name not in ['fc.weight', 'fc.bias']:
            param.requires_grad = False

    # init the fc layer
    init_weights(base_encoder)

    return base_encoder


def init_weights(m):
    '''Initialize weights with zeros
    '''

    # init the fc layer
    #m.fc.weight.data.normal_(mean=0.0, std=0.01)
    #m.fc.bias.data.zero_()
    m.fc.reset_parameters()


class CustomDataset(Dataset):
    """ Creates a custom pytorch dataset.

        - Creates two views of the same input used for unsupervised visual
        representational learning. (SimCLR, Moco, MocoV2)

    Args:
        data (array): Array / List of datasamples

        labels (array): Array / List of labels corresponding to the datasamples

        transforms (Dictionary, optional): The torchvision transformations
            to make to the datasamples. (Default: None)

        target_transform (Dictionary, optional): The torchvision transformations
            to make to the labels. (Default: None)

        two_crop (bool, optional): Whether to perform and return two views
            of the data input. (Default: False)

    Returns:
        img (Tensor): Datasamples to feed to the model.

        labels (Tensor): Corresponding lables to the datasamples.
    """

    def __init__(self, data, labels, transform=None, target_transform=None, two_crop=False):

        # shuffle the dataset
        idx = np.random.permutation(data.shape[0])

        if isinstance(data, torch.Tensor):
            data = data.numpy()  # to work with `ToPILImage'

        self.data = data[idx]

        # when STL10 'unlabelled'
        if not labels is None:
            self.labels = labels[idx]
        else:
            self.labels = labels

        self.transform = transform
        self.target_transform = target_transform
        self.two_crop = two_crop

    def __len__(self):
        return self.data.shape[0]

    def __getitem__(self, index):

        # If the input data is in form from torchvision.datasets.ImageFolder
        if isinstance(self.data[index][0], np.str_):
            # Load image from path
            image = Image.open(self.data[index][0]).convert('RGB')

        else:
            # Get image / numpy pixel values
            image = self.data[index]

        if self.transform is not None:

            # Data augmentation and normalisation
            img = self.transform(image)

        if self.target_transform is not None:

            # Transforms the target, i.e. object detection, segmentation
            target = self.target_transform(target)

        if self.two_crop:

            # Augments the images again to create a second view of the data
            img2 = self.transform(image)

            # Combine the views to pass to the model
            img = torch.cat([img, img2], dim=0)

        # when STL10 'unlabelled'
        if self.labels is None:
            return img, torch.Tensor([0])
        else:
            return img, self.labels[index].long()

def clean_data(img_dirs, dataframe):
    """ Clean the data """
    for img_dir in img_dirs:
        for idx, row in dataframe.iterrows():
            if not os.path.isfile(f'{img_dir}/{row.filename}'):
                print(f"Removing non-existing file from dataset: {img_dir}/{row.filename}")
                dataframe = dataframe.drop(idx)
    return dataframe

def get_dataframes(opt):
    if os.path.isfile(opt.training_data_csv):
        print("reading csv file: ", opt.training_data_csv)
        train_df = pd.read_csv(opt.training_data_csv)
    else:
        raise Exception(f'Cannot find file: {opt.training_data_csv}')

    if os.path.isfile(opt.test_data_csv):
        print("reading csv file: ", opt.test_data_csv)
        test_df = pd.read_csv(opt.test_data_csv)
    else:
        raise Exception(f'Cannot find file: {opt.test_data_csv}')

    #train_df = train_df.sample(10000)
    #test_df = test_df.sample(10000)

    train_df = clean_data(opt.dataset_path, train_df)
    test_df = clean_data(opt.dataset_path, test_df)


    if opt.trainingset_split:
        # Split train_df into train and val
        slide_ids = train_df.slide_id.unique()
        random.shuffle(slide_ids)
        train_req_ids = []
        valid_req_ids = []
        # Take same number of slides from each site
        training_size = int(len(slide_ids)*opt.trainingset_split)
        validation_size = len(slide_ids) - training_size
        train_req_ids.extend([slide_id for slide_id in slide_ids[:training_size]])  # take first
        valid_req_ids.extend([
            slide_id for slide_id in slide_ids[training_size:training_size+validation_size]])  # take last

        print("train / valid / total")
        print(f"{len(train_req_ids)} / {len(valid_req_ids)} / {len(slide_ids)}")

        val_df = train_df[train_df.slide_id.isin(valid_req_ids)] # First, take the slides for validation
        train_df = train_df[train_df.slide_id.isin(train_req_ids)] # Update train_df

    else:
        if os.path.isfile(opt.validation_data_csv):
            print("reading csv file: ", opt.validation_data_csv)
            val_df = pd.read_csv(opt.validation_data_csv)
        else:
            raise Exception(f'Cannot find file: {opt.test_data_csv}')

    if opt.balanced_validation_set:
        print('Use uniform validation set')
        samples_to_take = val_df.groupby('label').size().min()
        val_df = pd.concat([val_df[val_df.label == label].sample(samples_to_take) for label in val_df.label.unique()])

        print('Use uniform test set')
        samples_to_take = test_df.groupby('label').size().min()
        test_df = pd.concat([test_df[test_df.label == label].sample(samples_to_take) for label in test_df.label.unique()])

    return train_df, val_df, test_df


class ImagePatchesDataset(Dataset):
    def __init__(self, dataframe, image_dirs, transform=None, two_crop=False):
        self.dataframe = dataframe
        self.image_dir = image_dirs
        self.transform = transform
        self.two_crop = two_crop

        self.label_enum = {'TUMOR': 1, 'NONTUMOR': 0}
        self.labels = list(dataframe.label.apply(lambda x: self.label_enum[x]))

    def __len__(self):
        return len(self.dataframe.index)

    def __getitem__(self, index):
        row = self.dataframe.iloc[index]

        data_path = random.choice(self.image_dir) # take image from random folder (if multiple folders are present)
        path = f"{data_path}/{row.filename}"
        try:
            image = Image.open(path)
        except IOError:
            print(f"could not open {path}")
            return None

        if self.transform is not None:
            img = self.transform[0](image)
            if self.two_crop:
                img2 = self.transform[-1](image) #can be same or diff transform than first
                # img = torch.cat([img, img2], dim=0)
                img = [img, img2]
        else:
            raise NotImplementedError

        label = self.label_enum[row.label]

        return img, label, row.patch_id, row.slide_id


class LmdbDataset(torch.utils.data.Dataset):
    def __init__(self, lmdb_path, transform, two_crop=False):
        self.cursor_access = False
        self.lmdb_path = lmdb_path
        self.image_dimensions = (224, 224, 3) # size of files in lmdb
        self.transform = transform
        self.two_crop = two_crop

        self._init_db()

    def __len__(self):
        return self.length

    def _init_db(self):
        num_readers = 999

        self.env = lmdb.open(self.lmdb_path,
                             max_readers=num_readers,
                             readonly=1,
                             lock=0,
                             readahead=0,
                             meminit=0)

        self.txn = self.env.begin(write=False)
        self.cursor = self.txn.cursor()

        self.length = self.txn.stat()['entries']
        self.keys = [key for key, _ in self.txn.cursor()] # not so fast...

    def close(self):
        self.env.close()

    def __getitem__(self, index):
        ' cursor in lmdb is much faster than random access '
        if self.cursor_access:
            if not self.cursor.next():
                self.cursor.first()
            image = self.cursor.value()
        else:
            image = self.txn.get(self.keys[index])

        image = np.frombuffer(image, dtype=np.uint8)
        image = image.reshape(self.image_dimensions)
        image = Image.fromarray(image)

        img = self.transform(image)
        if self.two_crop:
            img2 = self.transform(image)
            img = torch.cat([img, img2], dim=0)

        return [img]


def random_split_image_folder(data, labels, n_classes, n_samples_per_class):
    """ Creates a class-balanced validation set from a training set.

        Specifically for the image folder class
    """

    train_x, train_y, valid_x, valid_y = [], [], [], []

    if isinstance(labels, list):
        labels = np.array(labels)

    for i in range(n_classes):
        # get indices of all class 'c' samples
        c_idx = (np.array(labels) == i).nonzero()[0]
        # get n unique class 'c' samples
        valid_samples = np.random.choice(c_idx, n_samples_per_class[i], replace=False)
        # get remaining samples of class 'c'
        train_samples = np.setdiff1d(c_idx, valid_samples)
        # assign class c samples to validation, and remaining to training
        train_x.extend(data[train_samples])
        train_y.extend(labels[train_samples])
        valid_x.extend(data[valid_samples])
        valid_y.extend(labels[valid_samples])

    # torch.from_numpy(np.stack(labels)) this takes the list of class ids and turns them to tensor.long

    return {'train': train_x, 'valid': valid_x}, \
        {'train': torch.from_numpy(np.stack(train_y)), 'valid': torch.from_numpy(np.stack(valid_y))}


def random_split(data, labels, n_classes, n_samples_per_class):
    """ Creates a class-balanced validation set from a training set.
    """

    train_x, train_y, valid_x, valid_y = [], [], [], []

    if isinstance(labels, list):
        labels = np.array(labels)

    for i in range(n_classes):
        # get indices of all class 'c' samples
        c_idx = (np.array(labels) == i).nonzero()[0]
        # get n unique class 'c' samples
        valid_samples = np.random.choice(c_idx, n_samples_per_class[i], replace=False)
        # get remaining samples of class 'c'
        train_samples = np.setdiff1d(c_idx, valid_samples)
        # assign class c samples to validation, and remaining to training
        train_x.extend(data[train_samples])
        train_y.extend(labels[train_samples])
        valid_x.extend(data[valid_samples])
        valid_y.extend(labels[valid_samples])

    if isinstance(data, torch.Tensor):
        # torch.stack transforms list of tensors to tensor
        return {'train': torch.stack(train_x), 'valid': torch.stack(valid_x)}, \
            {'train': torch.stack(train_y), 'valid': torch.stack(valid_y)}
    # transforms list of np arrays to tensor
    return {'train': torch.from_numpy(np.stack(train_x)),
            'valid': torch.from_numpy(np.stack(valid_x))}, \
        {'train': torch.from_numpy(np.stack(train_y)),
         'valid': torch.from_numpy(np.stack(valid_y))}


def sample_weights(labels):
    """ Calculates per sample weights. """
    class_sample_count = np.unique(labels, return_counts=True)[1]
    class_weights = 1. / torch.Tensor(class_sample_count)
    return class_weights[list(map(int, labels))]


def experiment_config(parser, args):
    """ Handles experiment configuration and creates new dirs for model.
    """
    # check number of models already saved in 'experiments' dir, add 1 to get new model number
    # run_dir = os.path.join(os.path.split(os.getcwd())[0], 'experiments')

    # os.makedirs(run_dir, exist_ok=True)

    # run_name = time.strftime("%Y-%m-%d_%H-%M-%S")

    # create all save dirs
    args.model_dir = args.save_dir #os.path.join(run_dir, run_name)

    os.makedirs(args.model_dir , exist_ok=True)

    args.summaries_dir = os.path.join(args.model_dir, 'summaries')
    args.checkpoint_dir = os.path.join(args.model_dir, 'checkpoint.pt')

    if not args.evaluate:
        args.load_checkpoint_dir = args.checkpoint_dir

    os.makedirs(args.summaries_dir, exist_ok=True)

    # save hyperparameters in .txt file
    with open(os.path.join(args.model_dir, 'hyperparams.txt'), 'w') as logs:
        for key, value in vars(args).items():
            logs.write('--{0}={1} \n'.format(str(key), str(value)))

    # save config file used in .txt file
    with open(os.path.join(args.model_dir, 'config.txt'), 'w') as logs:
        # Remove the string from the blur_sigma value list
        config = parser.format_values().replace("'", "")
        # Remove the first line, path to original config file
        config = config[config.find('\n')+1:]
        logs.write('{}'.format(config))

    # reset root logger
    [logging.root.removeHandler(handler) for handler in logging.root.handlers[:]]
    # info logger for saving command line outputs during training
    logging.basicConfig(level=logging.INFO, format='%(message)s',
                        handlers=[logging.FileHandler(os.path.join(args.model_dir, 'trainlogs.txt')),
                                  logging.StreamHandler()])
    return args


def print_network(model, args):
    """ Utility for printing out a model's architecture.
    """
    logging.info('-'*70)  # print some info on architecture
    logging.info('{:>25} {:>27} {:>15}'.format('Layer.Parameter', 'Shape', 'Param#'))
    logging.info('-'*70)

    for param in model.state_dict():
        # p_name = param.split('.')[-2]+'.'+param.split('.')[-1]
        # don't print batch norm layers for prettyness
        # if p_name[:2] != 'BN' and p_name[:2] != 'bn':
        logging.info(
            '{:>25} {:>27} {:>15}'.format(
                param,
                str(list(model.state_dict()[param].squeeze().size())),
                '{0:,}'.format(np.product(list(model.state_dict()[param].size())))
            )
        )
    logging.info('-'*70)

    logging.info('\nTotal params: {:,}\n\nSummaries dir: {}\n'.format(
        sum(p.numel() for p in model.parameters()),
        args.summaries_dir))

    for key, value in vars(args).items():
        if str(key) != 'print_progress':
            logging.info('--{0}: {1}'.format(str(key), str(value)))
