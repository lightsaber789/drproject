import numpy as np
import torch
from torchvision.utils import make_grid

from aptos.base import BaseTrainer
from torch.cuda.amp import GradScaler

import torch.nn.functional as F
class Trainer(BaseTrainer):
    """
    Trainer class

    Note:
        Inherited from BaseTrainer.
    """
    def __init__(self, model, loss, metrics, optimizer, resume, config, device,
                 data_loader, valid_data_loader=None, lr_scheduler=None):
        super().__init__(model, loss, metrics, optimizer, resume, config, device)
        self.data_loader = data_loader
        self.valid_data_loader = valid_data_loader
        self.do_validation = self.valid_data_loader is not None
        self.lr_scheduler = lr_scheduler
        self.log_step = int(np.sqrt(len(data_loader)))
        self.scaler = GradScaler()

        self.noise_std = config['training']['noise_std']

    def _eval_metrics(self, output, target):
        with torch.no_grad():
            acc_metrics = []
            for i, metric in enumerate(self.metrics):
                score = metric(output, target)
                acc_metrics.append(score)
                if not hasattr(score, '__len__'):  # hacky way to avoid logging conf matrix
                    self.writer.add_scalar(f'{metric.__name__}', acc_metrics[-1])
            return acc_metrics

#bad qwk because its instable, this function was for model fusion
    def _train_epochfuckthis(self, epoch):
        self.model.train()
        total_loss = 0.0
        running_outputs = []
        running_targets = []

        # More aggressive mixed-resolution parameters
        resize_prob = 0.7  # 70% chance to downsample (higher for more memory savings)
        low_res_size = 224  # Standard ImageNet size that works well with EfficientNet

        # Gradient accumulation parameters
        accum_steps = 2  # Process 2 batches before updating weights
        batch_accum = 0

        sample_batch = None

        for bidx, (data, target) in enumerate(self.data_loader):
            data, target = data.to(self.device), target.to(self.device).float()

            # Store first batch for logging
            if bidx == 0:
                sample_batch = (data.detach().cpu(), target.detach().cpu())

            # Random resolution selection with higher probability of downsampling
            if torch.rand(1).item() < resize_prob:
                data = F.interpolate(
                    data,
                    size=(low_res_size, low_res_size),
                    mode='bilinear',
                    align_corners=False
                )

            # Forward pass with mixed precision
            with torch.cuda.amp.autocast(enabled=True):
                output = self.model(data)

                # Ensure proper output shape (batch_size, 1)
                output = output.flatten(start_dim=1)
                if output.size(1) > 1:
                    output = output.mean(dim=1, keepdim=True)

                # Ensure target shape matches
                if target.dim() == 1:
                    target = target.unsqueeze(1)

                # Add noise to target
                noise = torch.normal(mean=0.0, std=self.noise_std, size=target.shape).to(self.device)
                loss = self.loss(output, target + noise) / accum_steps

            # Gradient accumulation
            loss.backward()
            batch_accum += 1

            if batch_accum % accum_steps == 0:
                # Clip gradients to prevent explosion
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                self.optimizer.step()
                self.optimizer.zero_grad()
                total_loss += loss.item() * accum_steps  # Scale back up

                # Store outputs for metrics
                running_outputs.append(output.detach().cpu())
                running_targets.append(target.detach().cpu())

            if bidx % self.log_step == 0:
                self._log_batch(
                    epoch,
                    bidx,
                    target.size(0),
                    len(self.data_loader),
                    loss.item() * accum_steps  # Show actual loss magnitude
                )

            # Clear memory
            del data, target, output, noise, loss
            torch.cuda.empty_cache()
            gc.collect()

        # Handle any remaining accumulated batches
        if batch_accum % accum_steps != 0:
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
            self.optimizer.step()
            self.optimizer.zero_grad()
            total_loss += loss.item() * (batch_accum % accum_steps)

        # Metric calculation and logging
        outputs = torch.cat(running_outputs).squeeze().numpy() if running_outputs else np.array([])
        targets = torch.cat(running_targets).squeeze().numpy() if running_targets else np.array([])

        self.writer.set_step(epoch - 1)
        if epoch == 1 and sample_batch is not None:
            sample_data = sample_batch[0]
            if sample_data.shape[1] == 6:
                self.writer.add_image('input_RGB', make_grid(sample_data[:, :3], nrow=8, normalize=True))
                self.writer.add_image('input_HSV', make_grid(sample_data[:, 3:], nrow=8, normalize=True))
            else:
                #self.writer.add_image('input', make_grid(sample_data, nrow=8, normalize=True))
                self.writer.add_image('input', make_grid(sample_data[:, :3], nrow=8, normalize=True))

        total_metrics = self._eval_metrics(outputs, targets) if len(outputs) > 0 else {}
        total_loss /= len(self.data_loader)
        self.writer.add_scalar('total_loss', total_loss)

        log = {
            'loss': total_loss,
            'metrics': total_metrics,
            'resolution_mix': f"{resize_prob * 100}% {low_res_size}px"
        }

        if self.do_validation:
            self.logger.debug('Starting validation...')
            val_log = self._valid_epoch(epoch)
            log = {**log, **val_log}

        if self.lr_scheduler is not None:
            self.lr_scheduler.step()

        return log

    def _train_epochBatch16_img256(self, epoch):

        self.model.train()

        total_loss = 0
        outputs = np.zeros(self.data_loader.n_samples)
        targets = np.zeros(self.data_loader.n_samples)
        assert outputs.shape[0] == len(self.data_loader) * self.data_loader.batch_sampler.batch_size

        for bidx, (data, target) in enumerate(self.data_loader):
            print(f"Input data shape before model: {data.shape}")
            data, target = data.to(self.device), target.to(self.device).float()
            # self.logger.info(f'target: {target}')
            self.optimizer.zero_grad()
            output = self.model(data)
            print("LOOK at output shape")
            print(output.shape)
            means = torch.zeros_like(target)
            noise = torch.normal(mean=means, std=self.noise_std)

            #TODO UNCOMMENT THIS WHEN USING MSE_LOSS
            if target.dim() == 1 and output.dim() == 2:
                target = target.unsqueeze(1)
                noise = noise.unsqueeze(1)  # <- important

            loss = self.loss(output, target + noise)

            loss.backward()
            self.optimizer.step()

            total_loss += loss.item()
            bs = target.size(0)
            outputs[bidx * bs:(bidx + 1) * bs] = output.cpu().detach().squeeze(1).numpy()
            targets[bidx * bs:(bidx + 1) * bs] = target.cpu().detach().numpy().squeeze()
            #targets[bidx * bs:(bidx + 1) * bs] = target.cpu().detach().numpy()

            if bidx % self.log_step == 0:
                self._log_batch(epoch, bidx, bs, len(self.data_loader), loss.item())

        # tensorboard logging
        self.writer.set_step(epoch - 1)
        if epoch == 1:  # only log images once to save time
            if data.shape[1] == 6:  # 6-channel input
                # Option 1: Just show first 3 channels (RGB)
                rgb_data = data[:, :3, :, :]
                self.writer.add_image(
                    'input_RGB', make_grid(rgb_data.cpu(), nrow=8, normalize=True))

                # Option 2: Show middle 3 channels (HSV combinations)
                hsv_data = data[:, 3:, :, :]
                self.writer.add_image(
                    'input_HSV', make_grid(hsv_data.cpu(), nrow=8, normalize=True))
            else:
                # Regular 3-channel image
                self.writer.add_image(
                    'input', make_grid(data.cpu(), nrow=8, normalize=True))

        total_metrics = self._eval_metrics(outputs, targets)
        total_loss /= len(self.data_loader)
        self.writer.add_scalar('total_loss', total_loss)
        log = {
            'loss': total_loss,
            'metrics': total_metrics
        }

        if self.do_validation:
            self.logger.debug('Starting validation...')
            val_log = self._valid_epoch(epoch)
            log = {**log, **val_log}

        if self.lr_scheduler is not None:
            self.lr_scheduler.step()

        return log

    def _log_batch(self, epoch, batch_idx, batch_size, n_batches, loss):
        n_complete = batch_idx * batch_size
        n_samples = batch_size * n_batches
        percent = 100.0 * batch_idx / n_batches
        msg = f'Train Epoch: {epoch} [{n_complete}/{n_samples} ({percent:.0f}%)] Loss: {loss:.6f}'
        self.logger.debug(msg)

    def _valid_epoch(self, epoch):
        """
        Validate after training an epoch

        :return: A log that contains information about validation

        Note:
            The validation metrics in log must have the key 'val_metrics'.
        """
        self.model.eval()
        total_val_loss = 0
        outputs = np.zeros(len(self.valid_data_loader.batch_sampler.sampler))
        targets = np.zeros(len(self.valid_data_loader.batch_sampler.sampler))
        with torch.no_grad():
            for bidx, (data, target) in enumerate(self.valid_data_loader):
                data, target = data.to(self.device), target.to(self.device).float()
                output = self.model(data)

                #TODO uncomment this for MSE_LOSS
                loss = self.loss(output, target.unsqueeze(1))

               #loss = self.loss(output, target)
                total_val_loss += loss.item()
                bs = target.size(0)
                outputs[bidx * bs:(bidx + 1) * bs] = output.cpu().detach().squeeze(1).numpy()
                targets[bidx * bs:(bidx + 1) * bs] = target.cpu().detach().numpy().squeeze()
               # targets[bidx * bs:(bidx + 1) * bs] = target.cpu().detach().numpy()


        self.writer.set_step((epoch - 1), 'valid')
        total_val_metrics = self._eval_metrics(outputs, targets)
        total_val_loss /= len(self.valid_data_loader)
        self.writer.add_scalar('total_loss', total_val_loss)

        if data.shape[1] == 6:
            rgb_data = data[:, :3, :, :]  # Take only first 3 channels
            self.writer.add_image('input', make_grid(rgb_data.cpu(), nrow=8, normalize=True))
        else:
            self.writer.add_image('input', make_grid(data.cpu(), nrow=8, normalize=True))

        return {
            'val_loss': total_val_loss,
            'val_metrics': total_val_metrics
        }
