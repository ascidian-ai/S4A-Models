'''
Implementation of the model proposed in:
- Olaf Ronneberger, , Philipp Fischer, and Thomas Brox. "U-Net: Convolutional
Networks for Biomedical Image Segmentation." (2015).

Code adopted from:
https://github.com/PyTorchLightning/lightning-bolts/blob/master/pl_bolts/models/vision/unet.py
'''

from datetime import datetime
import numpy as np
from pathlib import Path
import pickle

import torch
import torch.nn as nn
from torch.nn import functional as F
from torch.optim import lr_scheduler
import torch.optim as optim
import pytorch_lightning as pl

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import Rectangle
import seaborn as sns

from sklearn.metrics._classification import confusion_matrix # added 20220812 Steven Tuften
from torchmetrics.functional import dice # added 20220814 Steven Tuften

########################################################
# ADDED BY ST 17AUG2022 to auto email experiment status
from utils.email import notification as email_notification
########################################################


def get_last_model_checkpoint(path):
    '''
    Browses through the given path and finds the last saved checkpoint of a
    model.

    Parameters
    ----------
    path: str or Path
        The path to search.

    Returns
    -------
    (Path, Path, int): the path of the last model checkpoint file, the path of the
    last optimizer checkpoint file and the corresponding epoch.
    '''
    model_chkp = [c for c in Path(path).glob('model_state_dict_*')]
    optimizer_chkp = [c for c in Path(path).glob('optimizer_state_dict_*')]
    model_chkp_per_epoch = {int(c.name.split('.')[0].split('_')[-1]): c for c in model_chkp}
    optimizer_chkp_per_epoch = {int(c.name.split('.')[0].split('_')[-1]): c for c in optimizer_chkp}

    last_model_epoch = sorted(model_chkp_per_epoch.keys())[-1]
    last_optimizer_epoch = sorted(optimizer_chkp_per_epoch.keys())[-1]

    assert last_model_epoch == last_optimizer_epoch, 'Error: Could not resume training. Optimizer or model checkpoint missing.'

    return model_chkp_per_epoch[last_model_epoch], optimizer_chkp_per_epoch[last_model_epoch], last_model_epoch


class DoubleConv(nn.Module):
    """[ Conv2d => BatchNorm (optional) => ReLU ] x 2."""

    def __init__(self, in_ch: int, out_ch: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.net(x)


class Down(nn.Module):
    """Downscale with MaxPool => DoubleConvolution block."""

    def __init__(self, in_ch: int, out_ch: int):
        super().__init__()
        self.net = nn.Sequential(nn.MaxPool2d(kernel_size=2, stride=2), DoubleConv(in_ch, out_ch))

    def forward(self, x):
        return self.net(x)


class Up(nn.Module):
    """Upsampling (by either bilinear interpolation or transpose convolutions) followed by concatenation of feature
    map from contracting path, followed by DoubleConv."""

    def __init__(self, in_ch: int, out_ch: int, bilinear: bool = False):
        super().__init__()
        self.upsample = None
        if bilinear:
            self.upsample = nn.Sequential(
                nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True),
                nn.Conv2d(in_ch, in_ch // 2, kernel_size=1),
            )
        else:
            self.upsample = nn.ConvTranspose2d(in_ch, in_ch // 2, kernel_size=2, stride=2)

        self.conv = DoubleConv(in_ch, out_ch)

    def forward(self, x1, x2):
        x1 = self.upsample(x1)

        # Pad x1 to the size of x2
        diff_h = x2.shape[2] - x1.shape[2]
        diff_w = x2.shape[3] - x1.shape[3]

        x1 = F.pad(x1, [diff_w // 2, diff_w - diff_w // 2, diff_h // 2, diff_h - diff_h // 2])

        # Concatenate along the channels axis
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)

class UNet(pl.LightningModule):
    def __init__(self, run_path, linear_encoder, learning_rate=1e-3, parcel_loss=False,
                 class_weights=None, crop_encoding=None, checkpoint_epoch=None,
                 num_layers=3, num_bands=4, img_dim=61, window_len=6):
        '''
        Parameters:
        -----------
        run_path: str or Path
            The path to export results into.
        linear_encoder: dict
            A dictionary mapping the true labels to the given labels.
            True labels = the labels in the mappings file.
            Given labels = labels ranging from 0 to len(true labels), which the
            true labels have been converted into.
        learning_rate: float, default 1e-3
            The initial learning rate.
        parcel_loss: boolean, default False
            If True, then a custom loss function is used which takes into account
            only the pixels of the parcels. If False, then all image pixels are
            used in the loss function.
        class_weights: dict, default None
            Weights per class to use in the loss function.
        crop_encoding: dict, default None
            A dictionary mapping class ids to class names.
        checkpoint_epoch: int, default None
            The epoch loaded for testing.
        num_layers: int, default 3
            The number of layers to use in each path.
        num_bands: int, default 4
            The number of image bands or layers.
        img_dim: int, default 61
            The number of pixels in each row of the input image.
        window_len: int, default 6
            The length of the rolling window to be used (in months). Default 6.
        '''
        if num_layers < 1:
            raise ValueError(f"num_layers = {num_layers}, expected: num_layers > 0")

        self.num_layers = num_layers

        super(UNet, self).__init__()

        self.img_dim = img_dim
        self.linear_encoder = linear_encoder
        self.parcel_loss = parcel_loss

        self.epoch_train_losses = []
        self.epoch_valid_losses = []
        self.avg_train_losses = []
        self.avg_val_losses = []
        self.best_loss = None

        self.starttime = datetime.now()
        self.endtime = datetime.now()
        self.duration = 0.0

        self.num_discrete_labels = len(set(linear_encoder.values()))
        self.confusion_matrix = np.zeros([self.num_discrete_labels, self.num_discrete_labels])  # 20220812 ST changed to np.zeros from torch.zeros

        self.class_weights = class_weights
        self.checkpoint_epoch = checkpoint_epoch

        if class_weights is not None:
            class_weights_tensor = torch.tensor([class_weights[k] for k in sorted(class_weights.keys())]).cuda()

            if self.parcel_loss:
                self.lossfunction = nn.NLLLoss(ignore_index=0, weight=class_weights_tensor, reduction='sum')
            else:
                self.lossfunction = nn.NLLLoss(ignore_index=0, weight=class_weights_tensor)
        else:
            if self.parcel_loss:
                self.lossfunction = nn.NLLLoss(ignore_index=0, reduction='sum')
            else:
                self.lossfunction = nn.NLLLoss(ignore_index=0)

        # Extract label keys and values sorted by label key
        self.crop_encoding = crop_encoding
        self.label_values = []
        self.label_keys = []
        for k in sorted(self.linear_encoder.keys()):
            self.label_values.append(self.crop_encoding[k])
            self.label_keys.append(k)

        self.label_dict = {i: key for i, key in enumerate(self.label_values)}

        self.run_path = Path(run_path)

        input_channels = num_bands * window_len   # bands * time steps (Window lengths ie. number of rolling months)

        # Encoder
        # -------
        layers = [DoubleConv(input_channels, 64)]

        feats = 64
        for _ in range(num_layers - 1):
            layers.append(Down(feats, feats * 2))
            feats *= 2

        # Decoder
        # --------
        for _ in range(num_layers - 1):
            layers.append(Up(feats, feats // 2, False))
            feats //= 2

        layers.append(nn.Conv2d(feats, self.num_discrete_labels, kernel_size=1))
        layers.append(nn.LogSoftmax(dim=1))

        self.layers = nn.ModuleList(layers)

        # Hyperparameters
        self.learning_rate = learning_rate
        self.save_hyperparameters()

        if self.training:
            # Export metrics in text file
            self.metrics_file = self.run_path / f"training_metrics_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
            self.metrics_file.unlink(missing_ok=True)  # Delete file if present

            with open(self.metrics_file, "a") as f:
                f.write('Mode,Epoch,Start Time,End Time,Duration (HH:MM:ss),Duration (sec),Loss,Learning Rate\n')

        self.dice_score = []

    def forward(self, x):
        xi = [self.layers[0](x)]

        # Down path
        for layer in self.layers[1:self.num_layers]:
            xi.append(layer(xi[-1]))

        # Up path
        for i, layer in enumerate(self.layers[self.num_layers:-2]):
            xi[-1] = layer(xi[-1], xi[-2-i])

        xi[-1] = self.layers[-2](xi[-1])

        # Softmax
        return self.layers[-1](xi[-1])


    def configure_optimizers(self):
        optimizer = optim.Adam(self.parameters(), lr=self.learning_rate)
        pla_lr_scheduler = {
            'scheduler': lr_scheduler.ReduceLROnPlateau(optimizer,
                                                        factor=0.5,
                                                        patience=4,
                                                        verbose=True),
            'monitor': 'val_loss'
        }
        return [optimizer], [pla_lr_scheduler]

    def convertPatchesToTiles(self, patches, tile_dim=366):
        # patches are of shape (B,H,W)
        patches = patches.cpu().detach().numpy()
        patch_dim = patches.shape[-1]  # Get patch width
        num_tiles = patches.size // (tile_dim * tile_dim)

        tiles = np.empty(shape=(0, tile_dim, tile_dim))
        for t in range(num_tiles):
            f_dim = (tile_dim // patch_dim)
            f_dim_sq = f_dim * f_dim
            batch_tile = patches[(f_dim_sq * t):(f_dim_sq * (t + 1))]

            batch_re = batch_tile.swapaxes(0, 1)
            batch_re = batch_re.reshape(tile_dim, tile_dim)

            tile = np.empty(shape=(0, tile_dim))
            for i in range(0, f_dim):
                for j in range(0, patch_dim):
                    tile = np.vstack((tile, batch_re[i + f_dim * j]))

            tiles = np.vstack((tiles, [tile]))
        return tiles

    def DumpImages(self, batch, batch_predictions, batch_idx, dice_score):
        dice_score = np.nan_to_num(np.array(dice_score), nan=0.0, posinf=0.0, neginf=0.0)
        dice_score_avg = np.average(dice_score, axis=0)

        # if batch_idx <= 5: # Use this to Test
        if dice_score_avg > 0.2:
            print(f" | Dice Score(Avg): {dice_score_avg:.4f}")
            batch_inputs = batch['medians'].cpu().detach()  # (B, T, C, H, W)
            batch_labels = batch['labels'].to(torch.long).cpu().detach()  # (B, H, W)
            batch_preds = batch_predictions.to(torch.long).cpu().detach()  # (B, H, W)
            b, t, c, h, w = batch_inputs.size()

            # Create plot
            self.testrun_path = Path(self.run_path / f'tile_images')
            self.testrun_path.mkdir(exist_ok=True, parents=True)

            # Generate Tiles for Ground Truth
            tiles = self.convertPatchesToTiles(batch_labels, tile_dim=366)
            preds = self.convertPatchesToTiles(batch_preds, tile_dim=366)

            # convert RGB components of (B, T, C, H, W) to grayscale (B, H, W)
            # batchred = batch_inputs
            # rgbs = self.convertPatchesToTiles(batchred, tile_dim=366)

            num_tiles = len(tiles)

            fig, axs = plt.subplots(nrows=num_tiles, ncols=2, figsize=(12, 12))
            fig.suptitle(f'Batch #{batch_idx + 1}\nDice Score (Avg) = {dice_score_avg:0.3f}', fontsize=18)
            plt.setp(axs.flat, xticks=[], yticks=[])

            plottitle_font = {'size': '14'}
            cmap = "Paired"

            if num_tiles > 1:
                for i in range(num_tiles):
                    tile = tiles[i, :, :]
                    pred = preds[i, :, :]
                    axs[i, 0].imshow(tile, cmap=cmap)
                    im = axs[i, 1].imshow(pred, cmap=cmap)
                    axs[i, 0].set_title(f'Tile #{i + 1}\nGround Truth', fontdict=plottitle_font)
                    axs[i, 1].set_title(f'Tile #{i + 1}\nPrediction', fontdict=plottitle_font)
            else:
                tile = tiles[0, :, :]
                pred = preds[0, :, :]
                axs[0].imshow(tile, cmap=cmap)
                im = axs[1].imshow(pred, cmap=cmap)
                axs[0].set_title(f'Ground Truth', fontdict=plottitle_font)
                axs[1].set_title(f'Prediction', fontdict=plottitle_font)

            ## create patches as legend
            colors = [im.cmap(im.norm(label)) for label in self.label_dict]
            patches = [mpatches.Patch(color=colors[i], label=self.label_dict[i]) for i in range(len(self.label_dict))]
            plt.legend(handles=patches, loc='center left', bbox_to_anchor=(1.1, 0.5))
            fig.tight_layout()
            fig.savefig(self.testrun_path / f'batch_{batch_idx}_tiles.png', dpi=fig.dpi)
            plt.close(fig)

        return

    def training_step(self, batch, batch_idx):
        inputs = batch['medians']  # (B, T, C, H, W)

        label = batch['labels']  # (B, H, W)
        label = label.to(torch.long)

        # Concatenate time series along channels dimension
        b, t, c, h, w = inputs.size()
        inputs = inputs.view(b, -1, h, w)   # (B, T * C, H, W)

        pred = self(inputs)  # (B, K, H, W)

        if self.parcel_loss:
            parcels = batch['parcels']  # (B, H, W)
            parcels_K = parcels[:, None, :, :].repeat(1, pred.size(1), 1, 1)  # (B, K, H, W)

            # Note: a new masked array must be created in order to avoid inplace
            # operations on the label/pred variables. Otherwise the optimizer
            # will throw an error because it requires the variables to be unchanged
            # for gradient computation

            mask = (parcels) & (label != 0)
            mask_K = (parcels_K) & (label[:, None, :, :].repeat(1, pred.size(1), 1, 1) != 0)

            label_masked = label.clone()
            label_masked[~mask] = 0

            pred_masked = pred.clone()
            pred_masked[~mask_K] = 0

            label = label_masked.clone()
            pred = pred_masked.clone()

            loss = self.lossfunction(pred, label)

            loss = loss / parcels.sum()
        else:
            loss = self.lossfunction(pred, label)

        # Compute total loss for current batch
        loss_aver = loss.item() * inputs.shape[0]

        self.epoch_train_losses.append(loss_aver)

        # torch.nn.utils.clip_grad_value_(self.parameters(), clip_value=10.0)

        return {'loss': loss}


    def validation_step(self, batch, batch_idx):
        inputs = batch['medians']  # (B, T, C, H, W)

        label = batch['labels']  # (B, H, W)
        label = label.to(torch.long)

        # Concatenate time series along channels dimension
        b, t, c, h, w = inputs.size()
        inputs = inputs.view(b, -1, h, w)   # (B, T * C, H, W)

        pred = self(inputs)  # (B, K, H, W)

        if self.parcel_loss:
            parcels = batch['parcels']  # (B, H, W)
            parcels_K = parcels[:, None, :, :].repeat(1, pred.size(1), 1, 1)  # (B, K, H, W)

            # Note: a new masked array must be created in order to avoid inplace
            # operations on the label/pred variables. Otherwise the optimizer
            # will throw an error because it requires the variables to be unchanged
            # for gradient computation

            mask = (parcels) & (label != 0)
            mask_K = (parcels_K) & (label[:, None, :, :].repeat(1, pred.size(1), 1, 1) != 0)

            label_masked = label.clone()
            label_masked[~mask] = 0

            pred_masked = pred.clone()
            pred_masked[~mask_K] = 0

            label = label_masked.clone()
            pred = pred_masked.clone()

            loss = self.lossfunction(pred, label)

            loss = loss / parcels.sum()
        else:
            loss = self.lossfunction(pred, label)

        # Compute total loss for current batch
        loss_aver = loss.item() * inputs.shape[0]

        self.epoch_valid_losses.append(loss_aver)

        return {'val_loss': loss}


    def test_step(self, batch, batch_idx):
        inputs = batch['medians']  # (B, T, C, H, W)
        label = batch['labels'].to(torch.long)  # (B, H, W)

        # Concatenate time series along channels dimension
        b, t, c, h, w = inputs.size()
        inputs = inputs.view(b, -1, h, w)   # (B, T * C, H, W)

        pred = self(inputs).to(torch.long)  # (B, K, H, W)

        # Reverse the logarithm of the LogSoftmax activation
        pred = torch.exp(pred)

        # Clip predictions larger than the maximum possible label
        pred = torch.clamp(pred, 0, max(self.linear_encoder.values()))

        if self.parcel_loss:
            parcels = batch['parcels']  # (B, H, W)
            parcels_K = parcels[:, None, :, :].repeat(1, pred.size(1), 1, 1)  # (B, K, H, W)

            mask = (parcels) & (label != 0)
            mask_K = (parcels_K) & (label[:, None, :, :].repeat(1, pred.size(1), 1, 1) != 0)
            label[~mask] = 0
            pred[~mask_K] = 0

            pred_sparse = pred.argmax(axis=1)

            label = label.cpu().detach().flatten() # flatten after numpy conversion is faster
            prediction = pred_sparse.cpu()  #.detach().flatten() # flatten after numpy conversion is faster
            pred = prediction.detach().flatten() # flatten after numpy conversion is faster

        # added 20220812 Steven Tuften
        # Replace bespoke Confusion Matrix calculation with sklearn method to speed up by order of magnitude!
        cm_delta = confusion_matrix(label, pred, labels=range(self.num_discrete_labels)) # include label_keys so 12x12 returned
        self.confusion_matrix = self.confusion_matrix + cm_delta

        # added 20220815 Dice Score
        step_dice_score = dice(pred, label, num_classes=self.num_discrete_labels,
                               multiclass=True, zero_division=0,
                               average='none',ignore_index=0)
        self.dice_score.append(step_dice_score.numpy())

        self.DumpImages(batch=batch, batch_predictions=prediction, batch_idx=batch_idx, dice_score=step_dice_score)

        return


    def training_epoch_end(self, outputs):
        # Calculate average loss over an epoch
        train_loss = np.nanmean(self.epoch_train_losses)
        self.avg_train_losses.append(train_loss)
        self.log('loss', train_loss, prog_bar=True)

        # Clear list to track next epoch
        self.epoch_train_losses = []

        self.endtime = datetime.now()
        self.duration = self.endtime - self.starttime

        # Export metrics in text file
        with open(self.metrics_file, "a") as f:
            f.write(f'TRAIN,{self.current_epoch},'
                    f'{self.starttime.strftime("%Y-%m-%d %H:%M:%S")},{self.endtime.strftime("%Y-%m-%d %H:%M:%S")},'
                    f'{self.duration},{self.duration.total_seconds()},'
                    f'{train_loss},{self.learning_rate}\n')

        self.starttime = datetime.now()

        # Send Email Status update
        messagebody = f"""\
        Date/Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
        Status: TRAINING EPOCH END
        Epoch: {self.current_epoch}
        Duration: {self.duration}
        Loss: {train_loss}"""
        try:
            email_notification("DL Experiment | TRAINING EPOCH END", messagebody)
        except:
            print("SMTP Server connection interrupted")

    def validation_epoch_end(self, outputs):
        # Calculate average loss over an epoch
        valid_loss = np.nanmean(self.epoch_valid_losses)
        self.avg_val_losses.append(valid_loss)
        self.log('val_loss', valid_loss, prog_bar=True)

        # Clear list to track next epoch
        self.epoch_valid_losses = []

        self.endtime = datetime.now()
        self.duration = self.endtime - self.starttime

        # Export metrics in text file
        with open(self.metrics_file, "a") as f:
            f.write(f'VALIDATION,{self.current_epoch},'
                    f'{self.starttime.strftime("%Y-%m-%d %H:%M:%S")},{self.endtime.strftime("%Y-%m-%d %H:%M:%S")},'
                    f'{self.duration},{self.duration.total_seconds()},'
                    f'{valid_loss},"N/A"\n')

        self.starttime = datetime.now()

        # Send Email Status update
        messagebody = f"""\
        Date/Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
        Status: VALIDATION EPOCH END
        Epoch: {self.current_epoch}
        Duration: {self.duration}
        Loss: {valid_loss}"""
        try:
            email_notification("DL Experiment | VALIDATION EPOCH END", messagebody)
        except:
            print("SMTP Server connection interrupted")

    def test_epoch_end(self, outputs):
        self.testrun_path = Path(self.run_path / f'testrun_{datetime.now().strftime("%Y%m%d_%H%M%S")}')
        self.testrun_path.mkdir(exist_ok=True, parents=True)

        #self.confusion_matrix = self.confusion_matrix.cpu().detach().numpy() # Convert to ndarray
        self.confusion_matrix = self.confusion_matrix[1:, 1:]  # Drop zero label
        self.dice_score = np.array(self.dice_score)
        self.dice_score = self.dice_score[:, 1:]  # Drop zero label
        self.dice_score = np.nan_to_num(self.dice_score, nan=0.0, posinf=0.0, neginf=0.0)

        # Calculate metrics and confusion matrix
        fp = self.confusion_matrix.sum(axis=0) - np.diag(self.confusion_matrix)
        fn = self.confusion_matrix.sum(axis=1) - np.diag(self.confusion_matrix)
        tp = np.diag(self.confusion_matrix)
        tn = self.confusion_matrix.sum() - (fp + fn + tp)

       # Sensitivity, hit rate, recall, or true positive rate
        tpr = tp / (tp + fn)
        # Specificity or true negative rate
        tnr = tn / (tn + fp)
        # Precision or positive predictive value
        ppv = tp / (tp + fp)
        # Negative predictive value
        npv = tn / (tn + fn)
        # Fall out or false positive rate
        fpr = fp / (fp + tn)
        # False negative rate
        fnr = fn / (tp + fn)
        # False discovery rate
        fdr = fp / (tp + fp)
        # F1-score
        f1 = (2 * ppv * tpr) / (ppv + tpr)

        # Overall accuracy
        accuracy = (tp + tn) / (tp + fp + fn + tn)

        self.endtime = datetime.now()
        self.duration = self.endtime - self.starttime

        # Export metrics in text file
        metrics_file = self.testrun_path / f"evaluation_metrics_epoch{self.checkpoint_epoch}.csv"

        # Delete file if present
        metrics_file.unlink(missing_ok=True)

        with open(metrics_file, "a") as f:
            f.write('TIMING\n')
            f.write(f'Start,{self.starttime.strftime("%Y-%m-%d %H:%M:%S")}\n')
            f.write(f'End,{self.endtime.strftime("%Y-%m-%d %H:%M:%S")}\n')
            f.write(f'Duration (HH:MM:ss),"{self.duration}"\n')
            f.write(f'Duration (sec),{self.duration.total_seconds()}\n')

            f.write('\nTEST RESULTS\n')
            row = 'Class'
            for k in sorted(self.linear_encoder.keys()):
                if k == 0: continue
                row += f',{k} ({self.crop_encoding[k]})'
            f.write(row + '\n')

            class_samples = self.confusion_matrix.sum(axis=1)
            row = "class samples"
            for i in class_samples:
                row += f',{i:.4f}'
            f.write(row + '\n')

            row = 'tn'
            for i in tn:
                row += f',{i}'
            f.write(row + '\n')

            row = 'tp'
            for i in tp:
                row += f',{i}'
            f.write(row + '\n')

            row = 'fn'
            for i in fn:
                row += f',{i}'
            f.write(row + '\n')

            row = 'fp'
            for i in fp:
                row += f',{i}'
            f.write(row + '\n')

            row = "specificity"
            for i in tnr:
                row += f',{i:.4f}'
            f.write(row + '\n')

            row = "precision"
            for i in ppv:
                row += f',{i:.4f}'
            f.write(row + '\n')

            row = "recall"
            for i in tpr:
                row += f',{i:.4f}'
            f.write(row + '\n')

            row = "accuracy"
            for i in accuracy:
                row += f',{i:.4f}'
            f.write(row + '\n')

            row = "f1"
            for i in f1:
                row += f',{i:.4f}'
            f.write(row + '\n')

            dice_score_avg = np.average(self.dice_score, axis=0)
            row = "dice score"
            for i in dice_score_avg:
                row += f',{i:.4f}'
            f.write(row + '\n')

            f.write('\nWEIGHTED METRICS\n')
            row = 'weighted accuracy,weighted macro-f1,weighted precision,weighted dice score'
            f.write(row + '\n')

            class_samples = self.confusion_matrix.sum(axis=1)
            weighted_acc = ((accuracy * class_samples) / class_samples.sum()).sum()
            weighted_f1 = ((f1 * class_samples) / class_samples.sum()).sum()
            weighted_ppv = ((ppv * class_samples) / class_samples.sum()).sum()
            weighted_dice = ((dice_score_avg * class_samples) / class_samples.sum()).sum()
            f.write(f'{weighted_acc:.4f},{weighted_f1:.4f},{weighted_ppv:.4f},{weighted_dice:.4f}\n')

        # Normalize each row of the confusion matrix because class imbalance is
        # high and visualization is difficult
        row_mins = self.confusion_matrix.min(axis=1)
        row_maxs = self.confusion_matrix.max(axis=1)
        cm_norm =  (self.confusion_matrix - row_mins[:, None]) / (row_maxs[:, None] - row_mins[:, None])
        cm_norm = np.nan_to_num(cm_norm, nan=0.0, posinf=0.0, neginf=0.0) # Replace invalid values with 0

        # Convert CM to show % of ground truth assigned to each prediction
        row_totals = self.confusion_matrix.sum(axis=1)
        cm_pct = self.confusion_matrix / row_totals[:, None]
        cm_pct = np.nan_to_num(cm_pct, nan=0.0, posinf=0.0, neginf=0.0) # Replace invalid values with 0

        # Export Confusion Matrix
        # Replace invalid values with 0
        self.confusion_matrix = np.nan_to_num(self.confusion_matrix, nan=0.0, posinf=0.0, neginf=0.0)

        # Create plot of Percentage Confusion Matrix
        fig, ax = plt.subplots(1, 1, figsize=(12, 12))
        sns.heatmap(cm_pct, annot=True, ax=ax, cmap="Blues", fmt=".1%", annot_kws={'fontsize': 12, 'weight': 'bold'}, cbar=False)

        # Labels, title and ticks
        label_font = {'size': '24'}
        ax.set_xlabel('Predicted labels', fontdict=label_font, labelpad=10)
        ax.set_ylabel('Observed labels', fontdict=label_font, labelpad=10)

        ax.set_xticks(list(np.arange(0.5, len(self.linear_encoder.keys()) - 1 + 0.5)))
        ax.set_yticks(list(np.arange(0.5, len(self.linear_encoder.keys()) - 1 + 0.5)))

        ax.xaxis.set_ticks_position('none')
        ax.yaxis.set_ticks_position('none')

        ax.set_xticklabels([f'{self.crop_encoding[k]} ({k})' for k in sorted(self.linear_encoder.keys()) if k != 0],
                           fontsize=14, rotation=45, ha='right', rotation_mode='anchor') # rotation='vertical'
        ax.set_yticklabels([f'{self.crop_encoding[k]} ({k})' for k in sorted(self.linear_encoder.keys()) if k != 0],
                           fontsize=14, rotation='horizontal')

        ax.tick_params(axis='both', which='major')

        title_font = {'size': '21'}
        ax.set_title('Normalised Confusion Matrix\n(by observed class)\n', fontdict=title_font)

        # For Percent CM
        for i in range(len(self.linear_encoder.keys())):
            ax.axhline(i, color="white", lw=2)

        for i in range(len(self.linear_encoder.keys()) - 1):
            ax.add_patch(Rectangle((i + 0.02, i + 0.04), 0.96, 0.92, fill=False, edgecolor='red', lw=4))

        plt.savefig(self.testrun_path / f'cm_pct_epoch{self.checkpoint_epoch}.png', dpi=fig.dpi, bbox_inches='tight', pad_inches=0.5)
        np.save(self.testrun_path / f'cm_pct_epoch{self.checkpoint_epoch}.npy', cm_pct)

        # Create plot of Confusion Matrix
        fig, ax = plt.subplots(1, 1, figsize=(7, 7))
        sns.heatmap(self.confusion_matrix, annot=False, ax=ax, cmap="Blues", fmt="g", annot_kws={'fontsize': 6})

        # Labels, title and ticks
        label_font = {'size': '18'}
        ax.set_xlabel('Predicted labels', fontdict=label_font, labelpad=10)
        ax.set_ylabel('Observed labels', fontdict=label_font, labelpad=10)

        ax.set_xticks(list(np.arange(0.5, len(self.linear_encoder.keys()) - 1 + 0.5)))
        ax.set_yticks(list(np.arange(0.5, len(self.linear_encoder.keys()) - 1 + 0.5)))

        ax.xaxis.set_ticks_position('none')
        ax.yaxis.set_ticks_position('none')

        ax.set_xticklabels([f'{self.crop_encoding[k]} ({k})' for k in sorted(self.linear_encoder.keys()) if k != 0], fontsize=8, rotation='vertical')
        ax.set_yticklabels([f'{self.crop_encoding[k]} ({k})' for k in sorted(self.linear_encoder.keys()) if k != 0], fontsize=8, rotation='horizontal')

        ax.tick_params(axis='both', which='major')

        title_font = {'size': '21'}
        ax.set_title('Confusion Matrix', fontdict=title_font)

        for i in range(len(self.linear_encoder.keys()) - 1):
            ax.add_patch(Rectangle((i, i), 1, 1, fill=False, edgecolor='red', lw=2))

        plt.savefig(self.testrun_path / f'cm_epoch{self.checkpoint_epoch}.png', dpi=fig.dpi, bbox_inches='tight', pad_inches=0.5)
        np.save(self.testrun_path / f'cm_epoch{self.checkpoint_epoch}.npy', self.confusion_matrix)


        # Create plot of Normalised Confusion Matrix

        # Create plot
        fig, ax = plt.subplots(1, 1, figsize=(7, 7))
        sns.heatmap(cm_norm, annot=False, ax=ax, cmap="Blues", fmt=".1%", annot_kws={'fontsize': 6})

        # Labels, title and ticks
        label_font = {'size': '18'}
        ax.set_xlabel('Predicted labels', fontdict=label_font, labelpad=10)
        ax.set_ylabel('Observed labels', fontdict=label_font, labelpad=10)

        ax.set_xticks(list(np.arange(0.5, len(self.linear_encoder.keys()) - 1 + 0.5)))
        ax.set_yticks(list(np.arange(0.5, len(self.linear_encoder.keys()) - 1 + 0.5)))

        ax.xaxis.set_ticks_position('none')
        ax.yaxis.set_ticks_position('none')

        ax.set_xticklabels([f'{self.crop_encoding[k]} ({k})' for k in sorted(self.linear_encoder.keys()) if k != 0], fontsize=8, rotation='vertical')
        ax.set_yticklabels([f'{self.crop_encoding[k]} ({k})' for k in sorted(self.linear_encoder.keys()) if k != 0], fontsize=8, rotation='horizontal')

        ax.tick_params(axis='both', which='major')

        title_font = {'size': '21'}
        ax.set_title('Confusion Matrix', fontdict=title_font)

        for i in range(len(self.linear_encoder.keys()) - 1):
            ax.add_patch(Rectangle((i, i), 1, 1, fill=False, edgecolor='red', lw=2))

        plt.savefig(self.testrun_path / f'cm_norm_epoch{self.checkpoint_epoch}.png', dpi=fig.dpi, bbox_inches='tight', pad_inches=0.5)
        np.save(self.testrun_path / f'cm_norm_epoch{self.checkpoint_epoch}.npy', cm_norm)

        # Save Linear Encoder
        pickle.dump(self.linear_encoder, open(self.testrun_path / f'linear_encoder_epoch{self.checkpoint_epoch}.pkl', 'wb'))

        # Send Email Status update
        messagebody = f"""\
        Date/Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
        Status: TEST EPOCH END
        Duration: {self.duration}
        Number of Heads: {self.num_heads}
        weighted accuracy | weighted macro-f1 | weighted precision | weighted dice score
        {weighted_acc:.4f} | {weighted_f1:.4f} | {weighted_ppv:.4f} | {weighted_dice:.4f}
        """

        try:
            email_notification("DL Experiment | TEST EPOCH END", messagebody)
        except:
            print("SMTP Server connection interrupted")