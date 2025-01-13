"""
A wrapper for training pytorch neural networks (not for correct
and smooth!)
"""

# pytorch
import torch
import torch.nn

# plotting
import matplotlib.pyplot as plt


class ModelWrapper:
    """
    A class which contains the model life cycle. Will propaply work
    with class inheritances later.
    """
    def __init__(self):
        """
        Initializing the model
        """
        print('Initializing Model...')

    def get_data(self, Dataset):
        """
        Get the graph we would like to train on
        """
        print('Getting data...')
        dataset = Dataset()
        self.data = dataset[0]

    def get_model(self, Model):
        """
        Get the pytorch model.
        """
        print('Setting up model...')
        self.model = Model(n_inputs=self.data.num_features)
        print(self.model)

    def training_step(self):
        """
        One training step.
        """
        out = self.model(self.data.x,
                         self.data.edge_index,
                         self.data.edge_attr)
        loss = self.criterion(self.data.y[self.data.train_mask],
                              out[self.data.train_mask])
        loss.backward()
        self.optimizer.step()
        self.optimizer.zero_grad()

        return loss

    def training(self, learning_rate, n_epoch):
        """
        Training the model
        """
        print('Training...')
        self.criterion = torch.nn.MSELoss()
        self.optimizer = torch.optim.Adam(self.model.parameters(),
                                          lr=learning_rate,
                                          weight_decay=.01)
        for epoch in range(n_epoch):
            loss = self.training_step()
            if (epoch+1) % 50 == 0:
                print(f'epoch: {epoch+1}/{n_epoch};  ' +
                      f'RMSE: {loss.item()**.5:.5f}')



