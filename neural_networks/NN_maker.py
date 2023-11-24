import torch
import numpy as np
import pytorch_lightning as pl
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from pytorch_lightning.callbacks.model_checkpoint import ModelCheckpoint
from torch.utils.data import DataLoader
import os

class BaseFoilModel(pl.LightningModule):

    def __init__(self, train_data, val_data, test_data, Params):
        super().__init__()

        if not (train_data is None and val_data is None and test_data is None):
            # First handle the inputs that are only used when the model is to be trained
            self.train_data = train_data.astype(np.float32)
            self.val_data = val_data.astype(np.float32)
            self.test_data = test_data.astype(np.float32)
            self.input_size = train_data.shape[1] - 1
            self.lr = Params["learning_rate"]
            self.batch_size = Params["batch_size"]
            self.weight_decay = Params['weight_decay']
            self.storage_directory = Params['storage_directory']

            # Create storage
            self.test_predictions = []
            self.test_abs_errors = None
            self.test_rel_errors = None
        
        else:
            self.input_size = Params['input_size']

        # Now handle the inputs that are always needed
        self.layer_size = Params["layer_size"]
        self.num_hidden_layers = Params["num_hidden_layers"]

        # Now create the layers of the model
        self.input_layer = torch.nn.Linear(self.input_size, self.layer_size)
        self.hidden_layers = []
        for i in range(self.num_hidden_layers):
            setattr(self, f'hidden_layer_{i + 1}', torch.nn.Linear(self.layer_size, self.layer_size))
        self.output_layer = torch.nn.Linear(self.layer_size, 1)

    def forward(self, x):
        batch_size = x.shape[0]
        x = x.view(batch_size, -1)
        x = self.input_layer(x)
        x = torch.relu(x)

        for i in range(self.num_hidden_layers):
            x = getattr(self, f'hidden_layer_{i + 1}')(x)
            x = torch.relu(x)

        x = self.output_layer(x)

        return x

    def squared_loss(self, predictions, real_vals):
        return ((predictions - real_vals) ** 2).sum()

    def training_step(self, train_batch, batch_idx):
        x, y = train_batch[:, :-1], train_batch[:, -1].reshape(-1, 1)
        predictions = self.forward(x)
        loss = self.squared_loss(predictions, y)
        self.log('train_loss', loss)
        return loss

    def validation_step(self, val_batch, batch_idx):
        x, y = val_batch[:, :-1], val_batch[:, -1].reshape(-1, 1)
        predictions = self.forward(x)
        loss = self.squared_loss(predictions, y)
        self.log('val_loss', loss)
        self.log('val_mean_rel_error', np.mean(np.abs((predictions.cpu().numpy() - y.cpu().numpy()) / y.cpu().numpy())))
        self.log('val_median_rel_error', np.median(np.abs((predictions.cpu().numpy() - y.cpu().numpy()) / y.cpu().numpy())))
        return loss

    def test_step(self, test_batch, batch_idx):
        x, y = test_batch[:, :-1], test_batch[:, -1].reshape(-1, 1)
        predictions = self.forward(x)
        loss = self.squared_loss(predictions, y)
        self.log('test_loss', loss)
        self.test_predictions = np.concatenate((self.test_predictions, predictions.cpu().numpy().flatten()))

        return [y, predictions]
    
    def on_test_epoch_end(self):
        actual = self.test_data[:, -1]
        predicted = self.test_predictions
        abs_errors = np.abs(actual - predicted)
        rel_errors = np.abs(abs_errors / actual )

        if self.storage_directory is not None:
            np.savetxt(os.path.join(self.storage_directory, 'test_inputs.txt'), self.test_data[:, :-1])
            np.savetxt(os.path.join(self.storage_directory, 'test_targets.txt'), self.test_data[:, -1])
            np.savetxt(os.path.join(self.storage_directory, 'test_predictions.txt'), predicted)
            np.savetxt(os.path.join(self.storage_directory, 'test_abs_errors.txt'), abs_errors)
            np.savetxt(os.path.join(self.storage_directory, 'test_rel_errors.txt'), rel_errors)

        self.test_abs_errors = abs_errors
        self.test_rel_errors = rel_errors

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.lr, weight_decay=self.weight_decay)
        return optimizer
    
    def train_dataloader(self):
        return DataLoader(self.train_data, batch_size=self.batch_size)

    def val_dataloader(self):
        return DataLoader(self.val_data, batch_size=self.batch_size)

    def test_dataloader(self):
        return DataLoader(self.test_data, batch_size=self.batch_size)


class FoilModel:

    def __init__(self, *args, frozen_layers=None):
        """
        To load an existing model simply input the checkpoint path.
        To train a new model input train_data, val_data, test, data, and Params

        frozen_layers=[...ints...] freezes the corresponding layers. 0 indicates the input layer. The output layer cannot be frozen. Only relevant when five inputs are provided
        """
        if len(args) == 1 or len(args) == 5:
            if type(args[0]) != str:
                if len(args) == 1:
                    raise Exception('If only one argument is provided it must be a path to the to-be-loaded checkpoint')
                if len(args) == 5:
                    raise Exception('If five arguments are provided the first one must be a path to the to-be-loaded checkpoint')

            # We must infer the model architecture from the checkpoint:
            state_dict = torch.load(args[0], map_location=torch.device('cpu'))['state_dict']

            count = 0
            for item in state_dict.keys():
                if 'hidden_layer_' in item:
                    count += 1

            Params = {'input_size': state_dict['input_layer.weight'].shape[1],
                        'layer_size': state_dict['input_layer.weight'].shape[0],
                        'num_hidden_layers': count // 2}

            self.model = BaseFoilModel.load_from_checkpoint(args[0], train_data=None, val_data=None, test_data=None, Params=Params)
            self.model_path = args[0]
            self.testable = False

            if frozen_layers is not None:
                frozen_layers = np.ravel(frozen_layers)
                freeze_names = ['input']
                for i in frozen_layers:
                    freeze_names.append(f'hidden_layer_{i + 1}')
                
                if 'hidden_layer_0' in freeze_names:
                    freeze_names.remove('hidden_layer_0')
                    freeze_names.append('input')

                for name, param in self.model.named_parameters():
                    for layer_name in freeze_names:
                        if layer_name in name:
                            param.requires_grad = False

        if len(args) == 4 or len(args) == 5:
            if len(args) == 5:
                args = args[1:]
                if not (type(args[0]) == np.ndarray and type(args[1]) == np.ndarray and type(args[2]) == np.ndarray and type(args[3]) == dict):
                    raise Exception('When providing five inputs the last four must be train_data, val_data, test_data, (all as numpy arrays) and params in that order')

                Params = args[3]
                if Params['seed'] is not None:
                    pl.seed_everything(Params['seed'])

                self.model.train_data = args[0].astype(np.float32)
                self.model.val_data = args[1].astype(np.float32)
                self.model.test_data = args[2].astype(np.float32)
                self.model.input_size = args[0].shape[1] - 1
                self.model.lr = Params["learning_rate"]
                self.model.batch_size = Params["batch_size"]
                self.model.weight_decay = Params['weight_decay']
                self.model.storage_directory = Params['storage_directory']

                # Create storage
                self.model.test_predictions = []
                self.model.test_abs_errors = None
                self.model.test_rel_errors = None

            elif not (type(args[0]) == np.ndarray and type(args[1]) == np.ndarray and type(args[2]) == np.ndarray and type(args[3]) == dict):
                raise Exception('When providing four inputs they must be train_data, val_data, test_data, (all as numpy arrays) and params in that order')
            
            else:
                Params = args[3]
                if Params['seed'] is not None:
                    pl.seed_everything(Params['seed'])

                self.model = BaseFoilModel(*args)
            
            # From here on out things are the same, independent of whether we want to train a new or an existing model
            early_stop_callback = EarlyStopping(monitor='val_loss', min_delta=0.00, patience=Params["patience"], verbose=False, mode='min')
            checkpoint_callback = ModelCheckpoint(save_top_k=1, monitor="val_median_rel_error", mode="min")
        
            self.trainer = pl.Trainer(max_epochs=Params['max_epochs'], min_epochs=1,
                                    callbacks=[early_stop_callback, checkpoint_callback],
                                    default_root_dir=Params['storage_directory'], enable_checkpointing=True)
                
            self.trainer.fit(self.model)
            self.model_path = checkpoint_callback.best_model_path

            self.model = BaseFoilModel.load_from_checkpoint(self.model_path, train_data=args[0], val_data=args[1], test_data=args[2], Params=args[3])

            self.testable = True

        if len(args) not in [1, 4, 5]:
            raise Exception('Either one or four inputs must be provided')
        
        self.model.to('cpu')
        self.model.eval()

    def test(self):
        if self.testable:
            self.model.test_predictions = []
            self.trainer.test(self.model)
            return self.model.test_abs_errors, self.model.test_rel_errors
        
        else:
            print('No test data available')
            return None, None
        
    def __use_model(self, item):
        if np.size(item) == self.model.input_size:
            return np.ravel(self.model(torch.Tensor(item).reshape(1, -1)).detach())
        elif np.size(item) % self.model.input_size == 0:
            return np.ravel(self.model(torch.Tensor(item).detach()).detach())
        else:
            raise Exception(f"Model requires that the input has {self.model.input_size} entries or columns")

    def __getitem__(self, item):
        return self.__use_model(item)


def tune_model(train_data, val_data, test_data, Tuning_Dict, batch_size=64, patience=30, max_epochs=500, seed=999, do_tuning=[True, True, True]):
    """
    Based on the approach used in my paper on chord prediction in Jazz we tune learning rate, weight decay, and model architecture separately.
    Here we will begin with tuning the architecture. The function will directly return the trained model with the best performance
    
    The tuning dict must contain the list of to-be-examined values for number of layers and layer size (will be tuned together),
    learning rate, and weight decay (for those two the first value will be used during the tuning of the model architecture)

    Sample Tuning_Dict:

        Tuning_Dict = {
            'layer_size': [32, 64, 128, 256],
            'num_hidden_layers': [1, 3, 5],
            'learning_rate': [0.01, 0.02],
            'weight_decay': [0.00008, 0.00004, 0.00012]
            }
    """
    Params = {
        "batch_size": batch_size,
        "learning_rate": np.ravel(Tuning_Dict['learning_rate'])[0],
        "weight_decay": np.ravel(Tuning_Dict['weight_decay'])[0],
        "layer_size": None,
        "num_hidden_layers": None,
        "patience": patience,
        "storage_directory": None,
        "max_epochs": max_epochs,
        "seed": seed
        }
    
    results_dict = {}
    
    # First tune the model architecture
    if do_tuning[0]:
        current_best_loss_0 = np.inf
        used_params_tuning_1 = []
        val_losses_tuning_1 = []
        val_median_errors_tuning_1 = []
        val_mean_errors_tuning_1 = []
        for size in Tuning_Dict['layer_size']:
            for num_layers in Tuning_Dict['num_hidden_layers']:
                print(f'Now handling layer size {size} with {num_layers} layers')
                Params['layer_size'] = size
                Params['num_hidden_layers'] = num_layers

                model = FoilModel(train_data, val_data, test_data, Params)
                val_results = model.trainer.validate(model.model)[0]
                performance = val_results['val_loss']
                used_params_tuning_1.append([size, num_layers])
                val_losses_tuning_1.append(performance)
                val_median_errors_tuning_1.append(val_results['val_median_rel_error'])
                val_mean_errors_tuning_1.append(val_results['val_mean_rel_error'])

                if performance < current_best_loss_0:
                    best_params = [size, num_layers]
                    current_best_loss_0 = performance

        Params['layer_size'] = best_params[0]
        Params['num_hidden_layers'] = best_params[1]

        if best_params[0] in [min(Tuning_Dict['layer_size']), max(Tuning_Dict['layer_size'])]:
            print('WARNING: The best layer size was identified to be either the highest or lowest size tested for. Consider adding more options in that direction')

        if best_params[1] == max(Tuning_Dict['num_hidden_layers']):
            print('WARNING: The best number of hidden layers was identified to be the highest number tested for. Consider also testing for higher numbers of layers')

        results_dict['params_tuning_1'] = used_params_tuning_1
        results_dict['losses_tuning_1'] = val_losses_tuning_1
        results_dict['median_errors_tuning_1'] = val_median_errors_tuning_1
        results_dict['mean_errors_tuning_1'] = val_mean_errors_tuning_1

    else:
        Params['layer_size'] = np.ravel(Tuning_Dict['layer_size'])[0]
        Params['num_hidden_layers'] = np.ravel(Tuning_Dict['num_hidden_layers'])[0]
        best_params = [Params['layer_size'], Params['num_hidden_layers']]

    # Now tune the learning rate
    if do_tuning[1]:
        current_best_loss = np.inf
        used_params_tuning_2 = []
        val_losses_tuning_2 = []
        val_median_errors_tuning_2 = []
        val_mean_errors_tuning_2 = []

        for lr in Tuning_Dict['learning_rate']:
            print(f'Now handling learning rate {lr}')
            Params['learning_rate'] = lr

            if do_tuning[0] and lr == Tuning_Dict['learning_rate'][0]:
                # Do not unnecessarily re-tune the same model as before
                performance = current_best_loss_0
                val_median_errors_tuning_2.append(-1)
                val_mean_errors_tuning_2.append(-1)
            else:
                model = FoilModel(train_data, val_data, test_data, Params)
                val_results = model.trainer.validate(model.model)[0]
                performance = val_results['val_loss']
                val_median_errors_tuning_2.append(val_results['val_median_rel_error'])
                val_mean_errors_tuning_2.append(val_results['val_mean_rel_error'])

            used_params_tuning_2.append(lr)
            val_losses_tuning_2.append(performance)

            if performance < current_best_loss:
                best_lr = lr
                current_best_loss = performance

        Params['learning_rate'] = best_lr

        if best_lr in [min(Tuning_Dict['learning_rate']), max(Tuning_Dict['learning_rate'])]:
            print('WARNING: The best learning rate was identified to be either the highest or lowest one tested for. Consider adding more options in that direction')

        results_dict['params_tuning_2'] = used_params_tuning_2
        results_dict['losses_tuning_2'] = val_losses_tuning_2
        results_dict['median_errors_tuning_2'] = val_median_errors_tuning_2
        results_dict['mean_errors_tuning_2'] = val_mean_errors_tuning_2

    else:
        best_lr = Params['learning_rate']

    # And finally tune the weight decay
    if do_tuning[2]:
        current_best_loss = np.inf
        used_params_tuning_3 = []
        val_losses_tuning_3 = []
        val_median_errors_tuning_3 = []
        val_mean_errors_tuning_3 = []

        for decay in Tuning_Dict['weight_decay']:
            print(f'Now handling weight decay {decay}')
            Params['weight_decay'] = decay

            if do_tuning[0] and decay == Tuning_Dict['weight_decay'][0]:
                # Do not unnecessarily re-tune the same model as before
                performance = current_best_loss_0
                val_median_errors_tuning_3.append(-1)
                val_mean_errors_tuning_3.append(-1)
            else:
                model = FoilModel(train_data, val_data, test_data, Params)
                val_results = model.trainer.validate(model.model)[0]
                performance = val_results['val_loss']
                val_median_errors_tuning_3.append(val_results['val_median_rel_error'])
                val_mean_errors_tuning_3.append(val_results['val_mean_rel_error'])
            
            used_params_tuning_3.append(decay)
            val_losses_tuning_3.append(performance)

            if performance < current_best_loss:
                best_decay = decay
                current_best_loss = performance

        if best_decay in [min(Tuning_Dict['weight_decay']), max(Tuning_Dict['weight_decay'])]:
            print('WARNING: The best weight decay was identified to be either the highest or lowest one tested for. Consider adding more options in that direction')

        results_dict['params_tuning_3'] = used_params_tuning_3
        results_dict['losses_tuning_3'] = val_losses_tuning_3
        results_dict['median_errors_tuning_3'] = val_median_errors_tuning_3
        results_dict['mean_errors_tuning_3'] = val_mean_errors_tuning_3

    else:
        best_decay = Params['weight_decay']

    return best_params + [best_lr, best_decay], results_dict