import itertools, os, pickle, sys
from typing import Any, List, Mapping, Optional

from absl import logging
import numpy as np

import torch as pt
import torch.nn.functional as F

from ops.arguments import get_args
from ops.config import get_config, get_path_dict
from ops.cross_validate import cross_validate
from ops.data_pipeline import Data, load_dataset, load_dataset_online, load_dataset_prior
from ops.model import Model
from ops.general_utils import gaussian_blur

logging.get_absl_handler().python_handler.stream = sys.stdout
logging.set_verbosity(logging.INFO)


class Trainer():

    def __init__(self,
                 args: Mapping[str, Any],
                 path_dict: Mapping[str, Any], 
                 state_dict_path: Optional[str] = '') -> None:

        '''
        Train or fine tune model

        Args:
            args: non-tunable hyperparameters
            path_dict: dictionary of paths for opening datasets
            state_dict_path: path to model.state_dict(), if pretrained
        '''
        
        self.args = args
        self.path_dict = path_dict
        self.train_type = args.train_type
        if not self.train_type == 'prior':
            assert args.len_dataset % len(path_dict) == 0
            args.len_block = int(args.len_dataset/len(path_dict))
        self.model = Model(self.args, core_type = args.core_type, pretrained = args.load_pretrained).to(args.device)

        if os.path.exists(state_dict_path):
            self.model.load_state_dict(pt.load(state_dict_path, map_location=f'{args.device}'))
        self.optimizer = pt.optim.Adam(self.model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

        self.data_engine = Data(args)
        self.k_iterator = itertools.cycle(range(args.k))
        self.input_frames = np.zeros((self.args.len_dataset, 1, args.h, args.h))
        self.targets = np.zeros((self.args.len_dataset, args.h, args.h))

        if self.train_type == 'basic':
            k_iterator = itertools.cycle(range(args.k))
            self.input_frames = np.zeros((self.args.len_dataset, 1, args.h, args.h))
            targets = np.zeros((args.len_dataset, args.h, args.h))
        elif self.train_type == 'online':
            self.args.load_every == self.args.batch_size
            self.input_frames = np.zeros((args.batch_size, 1, args.h, args.h))
            self.targets = np.zeros((args.batch_size, args.h, args.h))
        elif self.train_type == 'prior':
            self.args.load_every = self.args.n_episodes


    def load_data(self):
        if self.train_type == 'basic':
            self.input_frames, self.targets = load_dataset(self.args,
                                                           self.path_dict,
                                                           self.model,
                                                           next(self.k_iterator),
                                                           self.input_frames,
                                                           self.targets,
                                                           self.args.high_pass_filter)
        elif self.train_type == 'online':
            self.input_frames, self.targets = load_dataset_online(self.args,
                                                           self.path_dict,
                                                           self.model,
                                                           self.input_frames,
                                                           self.targets)
        elif self.train_type == 'prior':
            self.input_frames, self.targets = load_dataset_prior(self.args,
                                                           self.path_dict)
        return self.input_frames, self.targets


    def train_step(self):

        '''
        Take a single optimization step on a single batch
        '''

        self.optimizer.zero_grad()
        x, reconstruction_target = next(self.data_engine.load(self.input_frames, self.targets))
        out = self.model(x)
        out = pt.clamp(gaussian_blur(pt.sigmoid(out)), 0, 1)
        loss = F.binary_cross_entropy(out, reconstruction_target)
        loss.backward()
        self.optimizer.step()

        return loss


    def cross_validate_k_fold(self, k, t):
        cross_val_ratio = cross_validate_k_fold(self.args,
                                                      self.path_dict,
                                                      self.model,
                                                      k,
                                                      self.args.len_cross_val)        
        return cross_val_ratio


    def cross_validate(self):
        cross_val_ratio = cross_validate(self.args, self.model)
        return cross_val_ratio

    def print_progress(self, t, loss, cross_val_ratio):
        logging.info(f'epoch: {t//self.args.cross_val_every} --- loss: {loss} --- cross val ratio: {cross_val_ratio}')


    def post_train(self, losses: List):

        pt.save(self.model.state_dict(), self.args.state_dict_path)
        pickle.dump(losses, open(f'{self.args.train_result_path}', 'wb'))

    def train(self) -> Mapping[str, Any]:

        self.model.train()
        losses = []

        for t in range(self.args.n_episodes):

            if t % self.args.save_every == 0:
                pt.save(self.model.state_dict(), self.args.state_dict_path)
            
            if t % self.args.load_every == 0:
                if self.train_type == 'basic':
                    logging.info('building dataset')
                    self.args.pixel_cutoff = int(self.args.pixel_cutoff*np.exp(-self.args.T_pixel_cutoff*t/self.args.n_episodes))
                if self.train_type == 'online':
                    self.args.load_every = 1
                if self.train_type == 'prior':
                    logging.info('building dataset')
                    assert self.args.load_every == self.args.n_episodes
                self.input_frames, self.targets = self.load_data()

            loss = self.train_step()
            losses.append(loss.item())
            
            if t % 50 == 0:
                logging.info(f'iteration: {t} --- loss: {loss.item()}')

            if t % self.args.cross_val_every == 0:
                logging.info('cross validate')
                cross_val_ratio = self.cross_validate()
                self.print_progress(t, loss.item(), cross_val_ratio)

        self.post_train(losses)


def main():

    logging.set_verbosity(logging.INFO)
    args = get_args()
    args = get_config(args)

    args.device = pt.device(f'cuda:{args.device_idx}' if pt.cuda.is_available() else 'cpu')
    path_dict = get_path_dict(f'{args.train_dataset_id}')
    
    if not os.path.exists('results'):
        os.makedirs('results')

    if args.train_type == 'basic':
        trainer = Trainer(args, path_dict)
    elif args.train_type == 'online':
        state_dict_path = f'{args.state_dict_path}'
        assert os.path.exists(state_dict_path)
        trainer = Trainer(args, path_dict, state_dict_path)
    elif args.train_type == 'prior':
        state_dict_path = f'{args.state_dict_path}'
        assert os.path.exists(state_dict_path)
        trainer = Trainer(args, path_dict, state_dict_path)

    trainer.train()


if __name__  == '__main__':
    main()