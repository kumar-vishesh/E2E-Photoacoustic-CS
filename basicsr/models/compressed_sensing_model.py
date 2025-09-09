# ------------------------------------------------------------------------
# Vishesh Kumar, 2025 
# ------------------------------------------------------------------------
# This is the compressed sensing model for photoacoustic imaging.
# ------------------------------------------------------------------------
# Based on image resoration_model from basisr & NAFNet
# ------------------------------------------------------------------------


import importlib
import torch
import torch.nn.functional as F
from collections import OrderedDict
from copy import deepcopy
from os import path as osp
from tqdm import tqdm
import numpy as np
import matplotlib.pyplot as plt

from basicsr.models.archs import define_network
from basicsr.models.modules.cs_frontend import CSFrontend
from basicsr.models.base_model import BaseModel
from basicsr.utils import get_root_logger, imwrite, tensor2img
from basicsr.utils.dist_util import get_dist_info
import os

loss_module = importlib.import_module('basicsr.models.losses')
metric_module = importlib.import_module('basicsr.metrics')

class E2ECompressedSensing(BaseModel):
    """Base Deblur model for single image deblur."""

    def __init__(self, opt):
        super(E2ECompressedSensing, self).__init__(opt)

        # define network
        self.net_g = define_network(deepcopy(opt['network_g']))
        self.net_g = self.model_to_device(self.net_g)

        # add compression+upsample frontend
        comp_cfg = opt.get('compression', {})
        self.frontend = CSFrontend(comp_cfg).to(self.device)

        # load pretrained models
        paths = self.opt.get('path', {})
        load_path = paths.get('pretrain_network_g', None)  # keep same YAML key
        if load_path is not None:
            self._load_weights(
                load_path,
                strict_g=paths.get('strict_load_g', True),
                strict_fe=paths.get('strict_load_fe', True)
            )


        if self.is_train:
            self.init_training_settings()

        self.scale = int(opt['scale'])

    def init_training_settings(self):
        self.net_g.train()
        self.frontend.train()
        train_opt = self.opt['train']

        # define losses
        if train_opt.get('pixel_opt'):
            pixel_type = train_opt['pixel_opt'].pop('type')
            cri_pix_cls = getattr(loss_module, pixel_type)
            self.cri_pix = cri_pix_cls(**train_opt['pixel_opt']).to(
                self.device)
        else:
            self.cri_pix = None

        if train_opt.get('perceptual_opt'):
            percep_type = train_opt['perceptual_opt'].pop('type')
            cri_perceptual_cls = getattr(loss_module, percep_type)
            self.cri_perceptual = cri_perceptual_cls(
                **train_opt['perceptual_opt']).to(self.device)
        else:
            self.cri_perceptual = None

        if self.cri_pix is None and self.cri_perceptual is None:
            raise ValueError('Both pixel and perceptual losses are None.')

        # set up optimizers and schedulers
        self.setup_optimizers()
        self.setup_schedulers()

    def setup_optimizers(self):
        train_opt = self.opt['train']
        optim_params = []

        for k, v in self.net_g.named_parameters():
            if v.requires_grad:
                optim_params.append(v)
        for k, v in self.frontend.named_parameters():
            if v.requires_grad:
                optim_params.append(v)

        optim_type = train_opt['optim_g'].pop('type')
        if optim_type == 'Adam':
            self.optimizer_g = torch.optim.Adam([{'params': optim_params}],
                                                **train_opt['optim_g'])
        elif optim_type == 'SGD':
            self.optimizer_g = torch.optim.SGD(optim_params,
                                               **train_opt['optim_g'])
        elif optim_type == 'AdamW':
            self.optimizer_g = torch.optim.AdamW([{'params': optim_params}],
                                                **train_opt['optim_g'])
            pass
        else:
            raise NotImplementedError(
                f'optimizer {optim_type} is not supperted yet.')
        self.optimizers.append(self.optimizer_g)

    def feed_data(self, data, is_val=False):
        self.lq = data['lq'].to(self.device)
        if 'gt' in data:
            self.gt = data['gt'].to(self.device)

    def grids(self):
        b, c, h, w = self.gt.size()
        self.original_size = (b, c, h, w)

        assert b == 1
        if 'crop_size_h' in self.opt['val']:
            crop_size_h = self.opt['val']['crop_size_h']
        else:
            crop_size_h = int(self.opt['val'].get('crop_size_h_ratio') * h)

        if 'crop_size_w' in self.opt['val']:
            crop_size_w = self.opt['val'].get('crop_size_w')
        else:
            crop_size_w = int(self.opt['val'].get('crop_size_w_ratio') * w)


        crop_size_h, crop_size_w = crop_size_h // self.scale * self.scale, crop_size_w // self.scale * self.scale
        #adaptive step_i, step_j
        num_row = (h - 1) // crop_size_h + 1
        num_col = (w - 1) // crop_size_w + 1

        import math
        step_j = crop_size_w if num_col == 1 else math.ceil((w - crop_size_w) / (num_col - 1) - 1e-8)
        step_i = crop_size_h if num_row == 1 else math.ceil((h - crop_size_h) / (num_row - 1) - 1e-8)

        scale = self.scale
        step_i = step_i//scale*scale
        step_j = step_j//scale*scale

        parts = []
        idxes = []

        i = 0  # 0~h-1
        last_i = False
        while i < h and not last_i:
            j = 0
            if i + crop_size_h >= h:
                i = h - crop_size_h
                last_i = True

            last_j = False
            while j < w and not last_j:
                if j + crop_size_w >= w:
                    j = w - crop_size_w
                    last_j = True
                parts.append(self.lq[:, :, i // scale :(i + crop_size_h) // scale, j // scale:(j + crop_size_w) // scale])
                idxes.append({'i': i, 'j': j})
                j = j + step_j
            i = i + step_i

        self.origin_lq = self.lq
        self.lq = torch.cat(parts, dim=0)
        self.idxes = idxes

    def grids_inverse(self):
        preds = torch.zeros(self.original_size)
        b, c, h, w = self.original_size

        count_mt = torch.zeros((b, 1, h, w))
        if 'crop_size_h' in self.opt['val']:
            crop_size_h = self.opt['val']['crop_size_h']
        else:
            crop_size_h = int(self.opt['val'].get('crop_size_h_ratio') * h)

        if 'crop_size_w' in self.opt['val']:
            crop_size_w = self.opt['val'].get('crop_size_w')
        else:
            crop_size_w = int(self.opt['val'].get('crop_size_w_ratio') * w)

        crop_size_h, crop_size_w = crop_size_h // self.scale * self.scale, crop_size_w // self.scale * self.scale

        for cnt, each_idx in enumerate(self.idxes):
            i = each_idx['i']
            j = each_idx['j']
            preds[0, :, i: i + crop_size_h, j: j + crop_size_w] += self.outs[cnt]
            count_mt[0, 0, i: i + crop_size_h, j: j + crop_size_w] += 1.

        self.output = (preds / count_mt).to(self.device)
        self.lq = self.origin_lq

    def _pack_weights(self):
        return {
            "net_g": self.net_g.state_dict(),
            "frontend": self.frontend.state_dict(),
        }

    def _load_weights(self, path, strict_g=True, strict_fe=True):
        ckpt = torch.load(path, map_location="cpu")
        if not (isinstance(ckpt, dict) and "net_g" in ckpt and "frontend" in ckpt):
            raise RuntimeError(
                f"Unified checkpoint expected with keys ['net_g','frontend'], got keys: {list(ckpt.keys())}"
            )
        mg, ug = self.net_g.load_state_dict(ckpt["net_g"], strict=strict_g)
        mf, uf = self.frontend.load_state_dict(ckpt["frontend"], strict=strict_fe)
        if mg or ug or mf or uf:
            print("[load] net_g missing:", mg, "unexpected:", ug,
                "| frontend missing:", mf, "unexpected:", uf)


    def optimize_parameters(self, current_iter, tb_logger):
        self.optimizer_g.zero_grad()

        if self.opt['train'].get('mixup', False):
            self.mixup_aug()

        # Use CSFrontend for compression + upsampling
        x_recon = self.frontend(self.lq)
        preds = self.net_g(x_recon)

        if not isinstance(preds, list):
            preds = [preds]

        self.output = preds[-1]

        l_total = 0
        loss_dict = OrderedDict()

        # pixel loss
        if self.cri_pix:
            l_pix = 0.
            for pred in preds:
                l_pix += self.cri_pix(pred, self.gt)
            l_total += l_pix
            loss_dict['l_pix'] = l_pix

        # perceptual loss
        if self.cri_perceptual:
            l_percep, l_style = self.cri_perceptual(self.output, self.gt)
            if l_percep is not None:
                l_total += l_percep
                loss_dict['l_percep'] = l_percep
            if l_style is not None:
                l_total += l_style
                loss_dict['l_style'] = l_style

        l_total = l_total + 0. * sum(p.sum() for p in self.net_g.parameters())

        l_total.backward()
        use_grad_clip = self.opt['train'].get('use_grad_clip', True)
        if use_grad_clip:
            for group in self.optimizer_g.param_groups:
                torch.nn.utils.clip_grad_norm_(group['params'], 0.01)

        self.optimizer_g.step()

        self.log_dict = self.reduce_loss_dict(loss_dict)

    def test(self):
        self.net_g.eval()
        self.frontend.eval()
        with torch.no_grad():
            n = len(self.lq)
            outs = []
            m = self.opt['val'].get('max_minibatch', n)
            i = 0
            while i < n:
                j = i + m
                if j >= n:
                    j = n

                inputs = self.lq[i:j]
                # Use CSFrontend for compression + upsampling
                x_recon = self.frontend(inputs)
                pred = self.net_g(x_recon)
                if isinstance(pred, list):
                    pred = pred[-1]
                outs.append(pred.detach().cpu())
                i = j

            self.output = torch.cat(outs, dim=0)
        self.net_g.train()
        self.frontend.train()
    
    def nondist_validation(self, dataloader, current_iter, tb_logger, save_img, rgb2bgr, use_image):
        dataset_name = dataloader.dataset.opt['name']
        with_metrics = self.opt['val'].get('metrics') is not None

        if with_metrics:
            self.metric_results = {metric: 0.0 for metric in self.opt['val']['metrics'].keys()}
            metric_pixel_counts = {metric: 0 for metric in self.opt['val']['metrics'].keys()}

        pbar = tqdm(total=len(dataloader), unit='image', desc='Validating')
        for idx, val_data in enumerate(dataloader):
            img_name = osp.splitext(osp.basename(val_data['lq_path'][0]))[0]

            self.feed_data(val_data, is_val=True)
            self.test()

            visuals = self.get_current_visuals()
            result = visuals['result']  # (N, C, H, W) or (C, H, W)
            gt = visuals.get('gt')      # may be None

            # Save images if requested
            if save_img:
                sr_img = tensor2img([result], rgb2bgr=rgb2bgr)
                if 'gt' in visuals:
                    gt_img = tensor2img([gt], rgb2bgr=rgb2bgr)

                save_dir = self.opt['path']['visualization']
                if self.opt['is_train']:
                    save_path = osp.join(save_dir, img_name, f'{img_name}_{current_iter}.png')
                    save_gt_path = osp.join(save_dir, img_name, f'{img_name}_{current_iter}_gt.png')
                else:
                    save_path = osp.join(save_dir, dataset_name, f'{img_name}.png')
                    save_gt_path = osp.join(save_dir, dataset_name, f'{img_name}_gt.png')

                imwrite(sr_img, save_path)
                if 'gt' in visuals:
                    imwrite(gt_img, save_gt_path)

            # Metric computation
            if with_metrics and gt is not None:
                for name, opt_ in deepcopy(self.opt['val']['metrics']).items():
                    metric_type = opt_.pop('type')
                    metric_fn = getattr(metric_module, metric_type)

                    if use_image:
                        sr_img = tensor2img([result], rgb2bgr=rgb2bgr, out_type=np.float32, min_max=(-1, 1))
                        gt_img = tensor2img([gt], rgb2bgr=rgb2bgr, out_type=np.float32, min_max=(-1, 1))
                        metric_val = metric_fn(sr_img, gt_img, **opt_)
                        pixel_count = sr_img.size if opt_.get('average_over_pixels', True) else 1
                    else:
                        # Handle batching like in epoch_summary
                        if result.ndim == 4:
                            batch_size = result.size(0)
                            for b in range(batch_size):
                                pred = result[b]
                                gt_b = gt[b] if gt.ndim == 4 else gt
                                val = metric_fn(pred, gt_b, **opt_)
                                px = torch.numel(pred) if opt_.get('average_over_pixels', True) else 1
                                self.metric_results[name] += val * px
                                metric_pixel_counts[name] += px
                            continue
                        else:
                            val = metric_fn(result, gt, **opt_)
                            px = torch.numel(result) if opt_.get('average_over_pixels', True) else 1

                        self.metric_results[name] += val * px
                        metric_pixel_counts[name] += px

            pbar.update(1)
            pbar.set_description(f'Test {img_name}')
        pbar.close()

        # Final metric averaging
        if with_metrics:
            for name in self.metric_results:
                count = metric_pixel_counts[name] if metric_pixel_counts[name] > 0 else 1
                self.metric_results[name] /= count

            self._log_validation_metric_values(current_iter, dataset_name, tb_logger, self.metric_results)

        return 0.

#    def nondist_validation(self, *args, **kwargs):
#        logger = get_root_logger()
#        logger.warning('nondist_validation is not implemented. Run dist_validation.')
#        self.dist_validation(*args, **kwargs)
#        return

    def _log_validation_metric_values(self, current_iter, dataset_name,
                                      tb_logger, metric_dict):
        log_str = f'Validation {dataset_name}, \t'
        for metric, value in metric_dict.items():
            if metric == 'mse':
                log_str += f'\t # {metric}: {value:.6e}'  # scientific notation
            else:
                log_str += f'\t # {metric}: {value:.4f}'
        logger = get_root_logger()
        logger.info(log_str)

        log_dict = OrderedDict()
        for metric, value in metric_dict.items():
            log_dict[f'm_{metric}'] = value

        self.log_dict = log_dict

    def get_current_visuals(self):
        out_dict = OrderedDict()
        out_dict['lq'] = self.lq.detach().cpu()
        # Compression matrix
        A = self.frontend.get_matrix()
        out_dict['A'] = A.detach().cpu() if hasattr(A, 'detach') else torch.tensor(A).cpu()
        # Compressed Ax (use frontend's get_compressed for consistency)
        with torch.no_grad():
            Ax = self.frontend.get_compressed(self.lq)
            x_upsample = self.frontend(self.lq)
        
        out_dict['Ax'] = Ax.detach().cpu()
        out_dict['x_upsample'] = x_upsample.detach().cpu()
        out_dict['result'] = self.output.detach().cpu()

        # Ground truth
        if hasattr(self, 'gt'):
            out_dict['gt'] = self.gt.detach().cpu()
        return out_dict

    def save(self, epoch, current_iter):
        is_best = isinstance(current_iter, str) and current_iter.startswith("best_model")
        save_dir = os.path.join(self.opt['path']['models'], 'best_models') if is_best else self.opt['path']['models']
        os.makedirs(save_dir, exist_ok=True)

        stem = f"{current_iter}" if is_best else f"model_{current_iter}"
        weights_path = osp.join(save_dir, f"{stem}.pth")
        torch.save(self._pack_weights(), weights_path)

        if not is_best:
            self.save_training_state(epoch, current_iter)




    def epoch_summary(self, dataloader, metrics_cfg, rgb2bgr=True, use_image=True):
        """Compute average metrics over the dataloader using the given metric configs."""
        self.net_g.eval()

        if not metrics_cfg:
            return {}

        results = {name: 0.0 for name in metrics_cfg}
        pixel_counts = {name: 0 for name in metrics_cfg}

        with torch.no_grad():
            for data in tqdm(dataloader, desc='[Eval]', unit='batches'):
                self.feed_data(data, is_val=True)
                self.test()

                visuals = self.get_current_visuals()
                if 'result' not in visuals or 'gt' not in visuals:
                    continue

                for name, cfg in deepcopy(metrics_cfg).items():
                    metric_type = cfg.pop('type')
                    metric_fn = getattr(metric_module, metric_type)
                    average = cfg.pop('average_over_pixels', True)

                    if use_image:
                        sr_img = tensor2img([visuals['result']], rgb2bgr=rgb2bgr, out_type=np.float32, min_max=(-1, 1))
                        gt_img = tensor2img([visuals['gt']], rgb2bgr=rgb2bgr, out_type=np.float32, min_max=(-1, 1))
                        metric_val = metric_fn(sr_img, gt_img, **cfg)
                        pixel_counts[name] += sr_img.size if average else 1
                        results[name] += metric_val * (sr_img.size if average else 1)
                    else:
                        preds = visuals['result']  # shape (N, C, H, W)
                        gts = visuals['gt']
                        batch_size = preds.size(0)

                        for b in range(batch_size):
                            pred = preds[b]
                            gt = gts[b]
                            metric_val = metric_fn(pred, gt, **cfg)

                            pixel_counts[name] += torch.numel(pred) if average else 1
                            results[name] += metric_val * (torch.numel(pred) if average else 1)

        # Final averaging
        for name in results:
            if pixel_counts[name] > 0:
                results[name] /= pixel_counts[name]

        return results