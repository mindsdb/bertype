import numpy

from sklearn.metrics import classification_report

import torch
from torch.utils.data import TensorDataset
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
from torch.utils.data import Subset

from sklearn.model_selection import train_test_split

from bertype.column_info import get_types
from bertype.column_info import get_subtypes
from bertype.column_info import get_modifiers


class PackedSequenceDataset(Dataset):

    def __init__(self, sequences, target, index, n_cols):
        """ Initializer

            packed_seq comes from torch.nn.pack_padded_sequence
            target is a tensor containing target data.
        """
        self.seqs_ = sequences
        self.idx_ = index
        self.n_embs_per_index = n_cols
        self.y_ = target
        self.batch_size_ = 8

    def __len__(self):
        """ Returns number of batches.
        """
        return len(self.y_)

    def __getitem__(self, index):
        """ Returns single batch of data.
        """
        str_idx = index
        end_idx = min(len(self.seqs_), index + self.batch_size_)
        x = self.seqs_[str_idx:end_idx]
        seq_lens = [len(s) for s in x]
        x = torch.nn.utils.rnn.pad_sequence(x, batch_first=True)
        x = torch.nn.utils.rnn.pack_padded_sequence(
                x,
                lengths=seq_lens,
                enforce_sorted=False,
                batch_first=True
            )

        return x, self.y_[str_idx:end_idx]


class BaseRNNClassifier(torch.nn.Module):
    """ Classifies columns into different modifiers.
    """

    def __init__(self, name: str,
                 input_size: int,
                 num_classes: int,
                 device: str = 'cpu'):
        """ Initializer.
        """
        super(BaseRNNClassifier, self).__init__()

        self.rank_ = 0  # torch_dist.get_rank()
        self.world_size_ = 1  # torch_dist.get_world_size()
        self.device_ = torch.device(device)

        # misc. parameters
        self.name = name
        self.in_size = input_size
        self.n_types = num_classes

        # data loaders
        self.dl_train = DataLoader
        self.dl_dev = DataLoader
        self.dl_test = DataLoader

        # neural network
        # recurrent layer
        self.rnn = torch.nn.LSTM(32,
                                 hidden_size=self.in_size,
                                 batch_first=True)
        self.fc1 = torch.nn.Linear(2 * self.in_size, 100)
        self.act1 = torch.nn.ReLU()
        self.fc2 = torch.nn.Linear(100, 100)
        self.act2 = torch.nn.ReLU()
        self.fcf = torch.nn.Linear(100, self.n_types)
        self.final_activation = torch.nn.LogSoftmax(dim=-1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """ Runs model to predict data type.
        """
        _, (h_t, c_t) = self.rnn(x)
        x = torch.cat([h_t, c_t], dim=2)
        x = torch.squeeze(x, dim=0)
        x = self.fc1(x)
        x = self.act1(x)
        x = self.fc2(x)
        x = self.act2(x)
        x = self.fcf(x)
        logits = self.final_activation(x)

        return logits

    def load(self, path: str):
        """ Loads model from disk.
        """
        self.load_state_dict(torch.load(path,
                             map_location=self.device_))

    def save(self, path: str):
        """ Saves model to disk.
        """
        torch.save(self.cpu().state_dict(), path)
        self.to(self.device_)

    def _prepare_data(self,
                      embedding_sequences: list,
                      column_info: list,
                      index: list,
                      embs_per_col: list):
        """ Builds data loaders for training.

            Embeddings is a list of _sequences_.
        """
        # transform to tensor
        emb_seqs_pt = []
        for emb_seq, c_info in zip(embedding_sequences, column_info):
            emb_seqs_pt.append(torch.tensor(emb_seq, dtype=torch.float32))
        col_info_np = numpy.asarray(column_info)
        col_info_pt = torch.tensor(col_info_np, dtype=torch.int64)
        dataset = PackedSequenceDataset(emb_seqs_pt, col_info_pt,
                                        index, embs_per_col)
        print(col_info_np.shape, len(dataset))

        # stratified splitting to somewhat compensate for class imbalance
        train_idx, devtest_idx = \
            train_test_split(numpy.arange(len(dataset)),
                             test_size=0.2,
                             random_state=999,
                             shuffle=True,
                             stratify=col_info_np)
        dev_idx, test_idx = \
            train_test_split(devtest_idx,
                             test_size=0.3,
                             random_state=999,
                             shuffle=True,
                             stratify=col_info_np[devtest_idx])
        ds_train = Subset(dataset, train_idx)
        ds_dev = Subset(dataset, dev_idx)
        ds_test = Subset(dataset, test_idx)

        # setup data loaders
        self.dl_train = DataLoader(ds_train,
                                   shuffle=True,
                                   batch_size=None)
        self.dl_dev = DataLoader(ds_dev,
                                 shuffle=False,
                                 batch_size=None)
        self.dl_test = DataLoader(ds_test,
                                  shuffle=False,
                                  batch_size=None)

    def train_single_epoch(self, optimizer, criterion):
        """ Trains model for a single epoch.
        """
        training_loss = 0.0
        for x, y in self.dl_train:
            x = x.to(self.device_)
            y = y.to(self.device_)
            optimizer.zero_grad()
            y_hat = self(x)
            loss = criterion(y_hat, y)
            loss.backward()
            optimizer.step()
            training_loss += loss.item()

        return training_loss

    def eval_dev_single_epoch(self, criterion):
        """ Trains model for a single epoch.
        """
        dev_loss = 0.0
        for x, y in self.dl_dev:
            x = x.to(self.device_)
            y = y.to(self.device_)
            with torch.no_grad():
                y_hat = self(x)
            loss = criterion(y_hat, y)
            dev_loss += loss.item()

        return dev_loss

    def evaluate(self):
        """ Evaluates model on testing set.
        """
        # final model evaluation
        predictions = []
        ground_truth = []
        for x, y in self.dl_test:
            x = x.to(self.device_)
            y = y.to(self.device_)
            with torch.no_grad():
                y_hat = self(x)
            p = torch.argmax(y_hat, dim=-1).cpu().numpy().ravel()
            t = y.cpu().numpy().ravel()
            ground_truth.append(t)
            predictions.append(p)
        gt = numpy.concatenate(ground_truth)
        pr = numpy.concatenate(predictions)

        return classification_report(gt, pr, zero_division=numpy.nan)

    def fit(self,
            embeddings: list, types: list, index: list, n_cols: list,
            n_epochs: int = 10,
            weights: numpy.ndarray = None,
            evaluate_every: int = 1,
            checkpoint_every: int = 1):
        """ Trains modifier classifier.
        """
        self._prepare_data(embeddings, types, index, n_cols)
        optimizer = torch.optim.Adam(self.parameters(), lr=1e-5)

        w = torch.ones(self.n_types)
        if weights is not None:
            w = torch.tensor(weights, dtype=torch.float32)
        w = w.to(self.device_)
        criterion = torch.nn.NLLLoss(weight=w)

        self.to(self.device_)

        for epoch in range(1, n_epochs + 1):
            self.train()
            train_loss = self.train_single_epoch(optimizer, criterion)
            self.eval()
            dev_loss = self.eval_dev_single_epoch(criterion)
            if epoch % evaluate_every == 0:
                clf_report = self.evaluate()
                print(clf_report)
            # if epoch % checkpoint_every == 0:
            #    self.save(f'./checkpoint_{self.name}_{epoch}.pth')
            print(f'{epoch:05d} {train_loss:+03.4f} {dev_loss:+03.4f}')


class BaseMLPClassifier(torch.nn.Module):
    """ Classifies columns into different modifiers.
    """

    def __init__(self, name: str, input_size: int, num_classes: int,
                 device: str = 'cpu'):
        """ Initializer.
        """
        super(BaseMLPClassifier, self).__init__()

        self.rank_ = 0  # torch_dist.get_rank()
        self.world_size_ = 1  # torch_dist.get_world_size()
        self.device_ = torch.device(device)

        # misc. parameters
        self.name = name
        self.in_size = input_size
        self.n_types = num_classes

        # data loaders
        self.dl_train = DataLoader
        self.dl_dev = DataLoader
        self.dl_test = DataLoader

        # neural network
        # single dense layer followed by softmax (lol)
        self.fc1 = torch.nn.Linear(self.in_size, 100)
        self.act1 = torch.nn.ReLU()
        self.fc2 = torch.nn.Linear(100, 100)
        self.act2 = torch.nn.ReLU()
        self.fcf = torch.nn.Linear(100, self.n_types)
        self.final_activation = torch.nn.LogSoftmax(dim=-1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """ Runs model to predict data type.
        """
        x = self.fc1(x)
        x = self.act1(x)
        x = self.fc2(x)
        x = self.act2(x)
        x = self.fcf(x)
        logits = self.final_activation(x)

        return logits

    def load(self, path: str):
        """ Loads model from disk.
        """
        self.load_state_dict(torch.load(path,
                             map_location=self.device_))

    def save(self, path: str):
        """ Saves model to disk.
        """
        torch.save(self.cpu().state_dict(), path)
        self.to(self.device_)

    def _prepare_data(self, embeddings: list, column_types: list):
        """ Builds data loaders for training.
        """
        X = torch.tensor(embeddings, dtype=torch.float32)
        Y = torch.tensor(column_types, dtype=torch.int64)

        dataset = TensorDataset(X, Y)
        ds_train, ds_dev, ds_test = random_split(dataset, [0.6, 0.2, 0.2])

        # setup data loaders
        self.dl_train = DataLoader(ds_train,
                                   num_workers=2,
                                   shuffle=True,
                                   batch_size=32)
        self.dl_dev = DataLoader(ds_dev,
                                 num_workers=2,
                                 shuffle=False,
                                 batch_size=32)
        self.dl_test = DataLoader(ds_test,
                                  num_workers=2,
                                  shuffle=False,
                                  batch_size=32)

    def train_single_epoch(self, optimizer, criterion):
        """ Trains model for a single epoch.
        """
        training_loss = 0.0
        for x, y in self.dl_train:
            x = x.to(self.device_)
            y = y.to(self.device_)
            optimizer.zero_grad()
            y_hat = self(x)
            loss = criterion(y_hat, y)
            loss.backward()
            optimizer.step()
            training_loss += loss.item()

        return training_loss

    def eval_dev_single_epoch(self, criterion):
        """ Trains model for a single epoch.
        """
        dev_loss = 0.0
        for x, y in self.dl_dev:
            x = x.to(self.device_)
            y = y.to(self.device_)
            with torch.no_grad():
                y_hat = self(x)
            loss = criterion(y_hat, y)
            dev_loss += loss.item()

        return dev_loss

    def evaluate(self):
        """ Evaluates model on testing set.
        """
        # final model evaluation
        predictions = []
        ground_truth = []
        for x, y in self.dl_test:
            x = x.to(self.device_)
            y = y.to(self.device_)
            with torch.no_grad():
                y_hat = self(x)
            p = torch.argmax(y_hat, dim=-1).cpu().numpy().ravel()
            t = y.cpu().numpy().ravel()
            ground_truth.append(t)
            predictions.append(p)
        gt = numpy.concatenate(ground_truth)
        pr = numpy.concatenate(predictions)

        return classification_report(gt, pr, zero_division=numpy.nan)

    def fit(self,
            embeddings: list, types: list,
            n_epochs: int = 10,
            weights: numpy.ndarray = None,
            evaluate_every: int = 1,
            checkpoint_every: int = 1):
        """ Trains modifier classifier.
        """
        self._prepare_data(embeddings, types)
        optimizer = torch.optim.Adam(self.parameters(), lr=2e-4)

        w = torch.ones(self.n_types)
        if weights is not None:
            w = torch.tensor(weights, dtype=torch.float32)
        w = w.to(self.device_)
        criterion = torch.nn.NLLLoss(weight=w)

        self.to(self.device_)

        for epoch in range(1, n_epochs + 1):
            self.train()
            train_loss = self.train_single_epoch(optimizer, criterion)
            self.eval()
            dev_loss = self.eval_dev_single_epoch(criterion)
            if epoch % evaluate_every == 0:
                clf_report = self.evaluate()
                print(clf_report)
            # if epoch % checkpoint_every == 0:
            #    self.save(f'./checkpoint_{self.name}_{epoch}.pth')
            print(f'{epoch:05d} {train_loss:+03.4f} {dev_loss:+03.4f}')


class SimpleTypeClassifier(BaseMLPClassifier):

    def __init__(self, embedding_dim: int, device: str = 'cpu'):
        """ Initializer.
        """
        super(SimpleTypeClassifier, self).__init__(
            'type_classifier',
            embedding_dim,
            len(get_types()),
            device=device
        )


class SimpleSubTypeClassifier(BaseMLPClassifier):

    def __init__(self, type_name: str, embedding_dim: int, device: str = 'cpu'):
        """ Initializer.
        """
        super(SimpleSubTypeClassifier, self).__init__(
            'subtype_classifier',
            embedding_dim,
            len(get_subtypes(type_name)),
            device=device
        )


class SimpleModifierClassifier(BaseRNNClassifier):

    def __init__(self, embedding_dim, device: str = 'cpu'):
        """ Initializer.
        """
        super(SimpleModifierClassifier, self).__init__(
            'modifier_classifier',
            embedding_dim,
            len(get_modifiers()),
            device=device
        )
