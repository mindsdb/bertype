import numpy

from sklearn.metrics import classification_report

import torch
from torch.utils.data import random_split
from torch.utils.data import TensorDataset
from torch.utils.data import DataLoader

from bertype.column_info import get_types
from bertype.column_info import get_subtypes
# from bertype.column_info import get_modifiers


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

    def __init__(self, embedding_dim: int, device: str = 'cpu'):
        """ Initializer.
        """
        super(SimpleSubTypeClassifier, self).__init__(
            'subtype_classifier',
            embedding_dim,
            len(get_subtypes()),
            device=device
        )


# class ColumnModifier(torch.nn.Module):
#     """ Classifies columns into different modifiers.
#     """
#     self.n_types = len(get_modifiers())

#     def __init__(self, dev: str = 'cpu'):
#         """ Initializer.
#         """
#         super(ColumnModifier, self).__init__()

#         self.rank_ = 0  # torch_dist.get_rank()
#         self.world_size_ = 1  # torch_dist.get_world_size()
#         self.device_ = torch.device(dev)
#         self.data_types_ = get_modifiers()

#         # setup one-hot encoder for data types
#         self.oh_encoder_ = OneHotEncoder(
#             sparse_output=False,
#             categories=[self.data_types_])
#         labels = numpy.asarray(self.data_types_)
#         self.oh_encoder_.fit(labels.reshape((-1, 1)))
#         # setup label encoder
#         self.lbl_encoder = LabelEncoder()
#         self.lbl_encoder = self.lbl_encoder.fit(self.data_types_)

#         # neural network
#         self.lstm_ = torch.nn.LSTM(128, 256)
#         self.conv11_ = torch.nn.Conv1d(2, 8, kernel_size=11, stride=2, padding=9)
#         self.conv12_ = torch.nn.Conv1d(8, 8, kernel_size=11, stride=1, padding=9)
#         self.act12_ = torch.nn.LeakyReLU()

#         self.conv21_ = torch.nn.Conv1d(8, 16, kernel_size=9, stride=2, padding=7)
#         self.conv22_ = torch.nn.Conv1d(16, 16, kernel_size=9, stride=1, padding=7)
#         self.act22_ = torch.nn.LeakyReLU()

#         self.conv31_ = torch.nn.Conv1d(16, 32, kernel_size=7, stride=2, padding=5)
#         self.conv32_ = torch.nn.Conv1d(32, 32, kernel_size=7, stride=1, padding=5)
#         self.act32_ = torch.nn.LeakyReLU()

#         self.conv41_ = torch.nn.Conv1d(32, 64, kernel_size=5, stride=2, padding=3)
#         self.conv42_ = torch.nn.Conv1d(64, 64, kernel_size=5, stride=1, padding=3)
#         self.act42_ = torch.nn.LeakyReLU()

#         self.conv51_ = torch.nn.Conv1d(64, 128, kernel_size=3, stride=2, padding=2)
#         self.conv52_ = torch.nn.Conv1d(128, 128, kernel_size=3, stride=2, padding=2)
#         self.act52_ = torch.nn.LeakyReLU()

#         self.pooling_ = torch.nn.MaxPool1d(8)

#         self.linear1_ = torch.nn.Linear(128, self.self.n_types)
#         self.final_activation = torch.nn.Softmax(dim=-1)

#         self.to(self.device_)

#     def forward(self, embeddings: torch.Tensor) -> torch.Tensor:
#         """ Runs model to predict data type.
#         """
#         x = torch.swapaxes(embeddings, 0, 1)
#         _, (h_z, c_z) = self.lstm_(x)
#         z = torch.cat([h_z, c_z])
#         z = torch.swapaxes(z, 0, 1)
#         z = self.conv11_(z)
#         z = self.conv12_(z)
#         z = self.act12_(z)

#         z = self.conv21_(z)
#         z = self.conv22_(z)
#         z = self.act22_(z)

#         z = self.conv31_(z)
#         z = self.conv32_(z)
#         z = self.act32_(z)

#         z = self.conv41_(z)
#         z = self.conv42_(z)
#         z = self.act42_(z)

#         z = self.conv51_(z)
#         z = self.conv52_(z)
#         z = self.act52_(z)

#         z = self.pooling_(z)
#         z = z.squeeze(dim=2)
#         t = self.linear1_(z)

#         type_logits = self.final_activation(t)

#         return type_logits

#     def load(self, path: str):
#         """ Loads model from disk.
#         """
#         self.load_state_dict(torch.load(path,
#                              map_location=self.device_))

#     def save(self, path: str):
#         """ Saves model to disk.
#         """
#         torch.save(self.cpu().state_dict(), path)

#     def decode_modifier(self, encoded):
#         """ Returns name of type from one-hot encoded vector.
#         """
#         return self.oh_encoder_.inverse_transform(encoded).ravel()[0]

#     # def infer_types(self, dataframe: pandas.DataFrame) -> dict:
#     #     """ Performs inference of column types.
#     #     """
#     #     self.eval()

#     #     types = {}
#     #     for col_name in dataframe.columns.to_list():
#     #         col_data_as_str = dataframe[col_name].to_string()
#     #         col_as_sentences = \
#     #             self.embedder_.string_to_sentences(col_data_as_str)
#     #         col_embeddings = self.embedder_.encode(col_as_sentences)
#     #         tp = numpy.zeros(self.self.n_types)
#     #         for x in col_embeddings:
#     #             with torch.no_grad():
#     #                 x = torch.tensor(x, dtype=torch.float32)
#     #                 x = x.view((1, 768)).to(self.device_)
#     #                 type_probs = self(x)
#     #             tp += type_probs.cpu().numpy().ravel()
#     #         tp /= len(col_embeddings)
#     #         types[col_name] = self.decode_type(tp.reshape((1, -1)))

#     #     return types

#     def fit(self,
#             embeddings: list, modifiers: list,
#             n_epochs: int = 10):
#         """ Trains modifier classifier.

#             Identification of modifiers relies on the sequence of embeddings
#             from a single column. For that reason, the data must be treated
#             differently than for the case of a usual classifier.

#             This routine assumes that embeddings and modifiers were stacked
#             in order from their respective data source, that is, that if
#             modifier n is different than modifier n+1, it means the embedding
#             n + 1 comes from a different column than embedding n.
#         """
#         X = []
#         Y = []
#         for embs, modi in zip(embeddings, modifiers):
#             if embs.shape[0] == 0 or embs.shape[1] == 0:
#                 continue
#             x = torch.tensor(embs, dtype=torch.float32)
#             y = torch.tensor(modi, dtype=torch.int64)
#             X.append(x)
#             Y.append(y)
#         mod_lbl_np = numpy.asarray(Y)
#         grp_idx = numpy.arange(len(Y))
#         sss = StratifiedShuffleSplit(n_splits=1, test_size=0.4, random_state=0)
#         train_idx, dev_test_idx = next(
#             sss.split(grp_idx, mod_lbl_np)
#         )
#         dev_idx, test_idx = next(
#             sss.split(dev_test_idx, mod_lbl_np[dev_test_idx])
#         )

#         print("training examples:", len(train_idx))
#         print("dev examples:", len(dev_idx))
#         print("test examples:", len(test_idx))

#         dataset = ColumnClassifierDataset(X, Y)
#         ds_train = Subset(dataset, train_idx)
#         ds_dev = Subset(dataset, dev_idx)
#         ds_test = Subset(dataset, test_idx)
#         # setup data loaders
#         dl_train = DataLoader(ds_train,
#                               num_workers=1, shuffle=True, batch_size=1)
#         dl_dev = DataLoader(ds_dev,
#                             num_workers=1, shuffle=False, batch_size=1)
#         dl_test = DataLoader(ds_test,
#                              num_workers=1, shuffle=False, batch_size=1)

#         metric = MulticlassAccuracy(num_classes=self.self.n_types)
#         metric.to(self.device_)
#         optimizer = torch.optim.Adam(self.parameters(), lr=1e-5)
#         criterion = torch.nn.CrossEntropyLoss()

#         for epoch in range(n_epochs):
#             self.train()
#             epoch_loss = 0.0
#             for x, y in tqdm(dl_train):
#                 x = x.to(self.device_)
#                 y = y.to(self.device_)
#                 optimizer.zero_grad()
#                 y_hat = self(x)
#                 loss = criterion(y_hat, y)
#                 loss.backward()
#                 optimizer.step()
#                 epoch_loss += loss.item()

#             self.eval()
#             metric.reset()
#             dev_loss = 0.0
#             with torch.no_grad():
#                 for x, y in tqdm(dl_dev):
#                     x = x.to(self.device_)
#                     y = y.to(self.device_)
#                     y_hat = self(x)
#                     loss = criterion(y_hat, y)
#                     dev_loss += loss.item()
#                     metric.update(torch.argmax(y_hat, dim=1), y)
#             print(f"epoch = {epoch + 1} loss = {epoch_loss:4.3f} test_loss = {dev_loss:4.3f} acc_dev = {metric.compute():4.3f}")

#         # final model evaluation
#         self.eval()
#         predictions = []
#         ground_truth = []
#         with torch.no_grad():
#             for x, y in dl_test:
#                 x = x.to(self.device_)
#                 y = y.to(self.device_)
#                 y_hat = self(x)
#                 y_oh = torch.nn.functional.one_hot(y, num_classes=self.self.n_types)
#                 ground_truth.append(y_oh.cpu().numpy())
#                 predictions.append(y_hat.cpu().numpy())

#         gt = numpy.concatenate(ground_truth)
#         pr = numpy.concatenate(predictions)

#         gt = self.oh_encoder_.inverse_transform(gt.reshape((-1, self.self.n_types)))
#         pr = self.oh_encoder_.inverse_transform(pr.reshape((-1, self.self.n_types)))

#         print(classification_report(gt, pr))
