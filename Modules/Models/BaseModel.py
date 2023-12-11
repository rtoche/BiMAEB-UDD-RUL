import os
import torch
import numpy as np
import pandas as pd
from torch import nn
from torch.utils.data import DataLoader


class BaseModel(nn.Module):
    def __init__(self, model_name: str, metrics_dir: str, regularization: str = None, regularization_weight = 0.005):
        super(BaseModel, self).__init__()
        self.__L1_regularization__ = "L1"
        self.__L2_regularization__ = "L2"
        self.__regularization_weight__ = regularization_weight
        self.__regularization_types__ = [self.__L1_regularization__, self.__L2_regularization__]

        if regularization:
            assert regularization in self.__regularization_types__, "Regularization must be either 'L1' or 'L2'"

        # Create dir to store metrics during training
        if not os.path.exists(metrics_dir):
            os.makedirs(metrics_dir)

        # Define dictionary to track loss
        self.__model_name__ = model_name
        self.__metrics_dir__ = metrics_dir
        self.__regularization__ = regularization

        self.__dict_epochs_key__: str = "epoch"
        self.__dict_training_loss__key__: str = "training_loss"
        self.__dict_testing_loss_key__: str = "testing_loss"
        self.__metrics_csv_file_path__: str = os.path.join(metrics_dir, "metrics.csv")

        self.history = {
            self.__dict_epochs_key__: [],
            self.__dict_training_loss__key__: [],
            # self.__dict_testing_loss_key__: []
        }

    def fit(self,
            train_dataloader: DataLoader,
            test_dataloader,
            epochs: int,
            optimizer,
            loss_function,
            device: str):

        if test_dataloader:
            self.history[self.__dict_testing_loss_key__] = []

        # Move the module2 to the device
        model = self.to(device)
        for epoch in range(0, epochs):
            # Define arrays to store losses for epoch
            training_losses: list = []
            testing_losses = []

            # Train Model
            # Iterate over the batches in the DataLoader
            print(f"Epoch {epoch + 1}/{epochs}")
            print("====================================================")
            model.train()
            total_number_training_batches = len(train_dataloader)
            for batch_count, batch in enumerate(train_dataloader):
                # Get training data and move them to device
                loss: torch.Tensor = self.__do_batch__(batch=batch, loss_function=loss_function, device=device)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                training_losses.append(loss.item())

                if (batch_count % 10) == 0:
                    batch_string = "Batch {}/{}".format(batch_count + 1, total_number_training_batches)
                    BaseModel.__print_batch_output__(leading_string=batch_string, loss=loss.item(), training=True)
            print()

            # Evaluate Model
            if test_dataloader:
                with torch.no_grad():
                    model.eval()
                    total_number_testing_batches = len(test_dataloader)
                    for batch_count, batch in enumerate(test_dataloader):
                        loss = self.__do_batch__(batch=batch, loss_function=loss_function, device=device)
                        testing_losses.append(loss.item())
                        if (batch_count % 10) == 0:
                            batch_string = "Batch {}/{}".format(batch_count + 1, total_number_testing_batches)
                            BaseModel.__print_batch_output__(leading_string=batch_string, loss=loss.item(), training=False)

            # Compute average losses for training/testing and save them
            training_losses: np.ndarray = np.asarray(training_losses)

            if test_dataloader:
                testing_losses = np.asarray(testing_losses)

            # Update the history and get the current batch training/testing loss
            self.__update_history__(epoch=epoch+1, training_losses=training_losses, testing_losses=testing_losses)
            epoch_avg_training_loss = self.history[self.__dict_training_loss__key__][epoch]
            if test_dataloader:
                epoch_avg_testing_loss = self.history[self.__dict_testing_loss_key__][epoch]

            # Print out end of epoch value
            print("\n")
            end_of_epoch_str = "End of Epoch {}".format(epoch + 1)
            BaseModel.__print_batch_output__(leading_string=end_of_epoch_str,
                                             loss=epoch_avg_training_loss, training=True)
            if test_dataloader:
                BaseModel.__print_batch_output__(leading_string=end_of_epoch_str,
                                                 loss=epoch_avg_testing_loss, training=False)
            print()

            # Save history after every epoch
            pd.DataFrame(self.history).to_csv(self.__metrics_csv_file_path__, index=False)

            # Save module2 after epoch
            model_instance_name = self.__model_name__ + "_state_dict"
            model_path = os.path.join(self.__metrics_dir__, self.__model_name__)
            self.save_model(model_path, model_instance_name)

    def save_model(self, model_path, model_name):
        # Create the dir that will store the module2.state_dict()
        if not os.path.exists(model_path):
            os.makedirs(model_path)
        torch.save(self.state_dict(), os.path.join(model_path, model_name))

    def __append_to_history_record__(self, key: str, val_to_append):
        hist_record = self.history[key]
        hist_record.append(val_to_append)
        self.history[key] = hist_record

    def __update_history__(self, epoch: int, training_losses: np.ndarray, testing_losses):
        # Compute average losses and save them
        epoch_avg_training_loss = np.mean(training_losses)
        if len(testing_losses) > 0:
            epoch_avg_testing_loss = np.mean(testing_losses)

        self.__append_to_history_record__(self.__dict_epochs_key__, epoch)
        self.__append_to_history_record__(self.__dict_training_loss__key__, epoch_avg_training_loss)
        if len(testing_losses) > 0:
            self.__append_to_history_record__(self.__dict_testing_loss_key__, epoch_avg_testing_loss)

    def __do_batch__(self, batch: int, loss_function, device: str):
        features, targets = batch

        # Move everything to the device
        features = features.to(device)
        targets = targets.to(device)

        predictions = self(features)
        predictions = predictions.to(device)

        # If a type of regularization is needed
        regularization_penalty = 0
        if self.__regularization__:
            regularization_penalty = self.__get_regularization_penalty__()

        loss = loss_function(predictions, targets) + regularization_penalty

        return loss

    @staticmethod
    def __print_batch_output__(leading_string: str, loss: float, training: bool):
        n_spaces = 20
        leading_string = leading_string.ljust(n_spaces)
        output_type = "Training" if training else "Testing"
        loss_string = "{0: <15}".format(output_type + " Loss:").ljust(15)
        output = "\t{} [{} {:.4f}]".format(leading_string, loss_string, loss)
        print(output)

    def __get_regularization_penalty__(self):
        weight = self.__regularization_weight__
        if self.__L1_regularization__:
            return weight * sum([p.abs().sum() for p in self.parameters()])
        elif self.__L2_regularization__:
            return weight * sum([(p**2).sum() for p in self.parameters()])
        else:
            RuntimeError("Passed regularization has no implementation")

