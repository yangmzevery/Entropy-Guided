import torch
import torch.nn as nn
import logging

from .mmce import MMCE_weighted
from .flsd import FocalLossAdaptive

from torch.nn import functional as F

# AdaFocal
import collections, math

# from https://github.com/torrvision/focal_calibration/blob/main/Losses/focal_loss.py
class FocalLoss(nn.Module):
    def __init__(self, gamma=0, **kwargs):
        super(FocalLoss, self).__init__()

        self.gamma = gamma
        logging.info("using gamma={}".format(gamma))

    def forward(self, input, target):

        target = target.view(-1,1)

        logpt = torch.nn.functional.log_softmax(input, dim=1)
        logpt = logpt.gather(1,target)
        logpt = logpt.view(-1)
        pt = logpt.exp()

        loss = -1 * (1-pt)**self.gamma * logpt
        
        return loss.mean()

class CrossEntropy(nn.Module):
    def __init__(self, **kwargs) -> None:
        super(CrossEntropy, self).__init__()
        self.criterion = nn.CrossEntropyLoss()

    def forward(self, input, target):
        return self.criterion(input, target)

class LabelSmoothingLoss(nn.Module):
    def __init__(self, alpha=0.1, dim=-1, **kwargs):
        super(LabelSmoothingLoss, self).__init__()
        self.confidence = 1.0 - alpha
        self.alpha = alpha
        self.dim = dim

    def forward(self, pred, target):
        pred = pred.log_softmax(dim=self.dim)
        num_classes = pred.shape[self.dim]
        with torch.no_grad():
            # true_dist = pred.data.clone()
            true_dist = torch.zeros_like(pred)
            true_dist.fill_(self.alpha / (num_classes - 1))
            true_dist.scatter_(1, target.data.unsqueeze(1), self.confidence)
        return torch.mean(torch.sum(-true_dist * pred, dim=self.dim))

class MDCA(torch.nn.Module):
    def __init__(self, **kwargs):
        super(MDCA,self).__init__()

    def forward(self , output, target):
        output = torch.softmax(output, dim=1)
        # [batch, classes]
        loss = torch.tensor(0.0).cuda()
        batch, classes = output.shape
        for c in range(classes):
            avg_count = (target == c).float().mean()
            avg_conf = torch.mean(output[:,c])
            loss += torch.abs(avg_conf - avg_count)
        denom = classes
        loss /= denom
        return loss

class ClassficationAndMDCA(nn.Module):
    def __init__(self, loss="NLL+MDCA", alpha=0.1, beta=1.0, gamma=1.0, **kwargs):
        super(ClassficationAndMDCA, self).__init__()
        
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        if "NLL" in loss:
            self.classification_loss = nn.CrossEntropyLoss()
        elif "FL" in loss:
            self.classification_loss = FocalLoss(gamma=self.gamma)
        else:
            self.classification_loss = LabelSmoothingLoss(alpha=self.alpha) 
        self.MDCA = MDCA()

    def forward(self, logits, targets):
        loss_cls = self.classification_loss(logits, targets)
        loss_cal = self.MDCA(logits, targets)
        return loss_cls + self.beta * loss_cal

class BrierScore(nn.Module):
    def __init__(self, **kwargs):
        super(BrierScore, self).__init__()

    def forward(self, logits, target):
        
        target = target.view(-1,1)
        target_one_hot = torch.FloatTensor(logits.shape).to(target.get_device())
        target_one_hot.zero_()
        target_one_hot.scatter_(1, target, 1)

        pt = torch.softmax(logits, dim=1)
        squared_diff = (target_one_hot - pt) ** 2

        loss = torch.sum(squared_diff) / float(logits.shape[0])
        return loss

class DCA(nn.Module):
    def __init__(self, beta=1.0, **kwargs):
        super().__init__()
        self.beta = beta
        self.cls_loss = nn.CrossEntropyLoss()

    def forward(self, logits, targets):
        output = torch.softmax(logits, dim=1)
        conf, pred_labels = torch.max(output, dim = 1)
        calib_loss = torch.abs(conf.mean() -  (pred_labels == targets).float().mean())
        return self.cls_loss(logits, targets) + self.beta * calib_loss

class MMCE(nn.Module):
    def __init__(self, beta=2.0, **kwargs):
        super().__init__()
        self.beta = beta
        self.mmce = MMCE_weighted()
        self.cls_loss = nn.CrossEntropyLoss()

    def forward(self, logits, targets):
        cls = self.cls_loss(logits, targets)
        calib = self.mmce(logits, targets)
        return cls + self.beta * calib

class FLSD(nn.Module):
    def __init__(self, gamma=3.0, **kwargs):
        super().__init__()
        self.gamma = gamma
        self.criterion = FocalLossAdaptive(gamma=self.gamma)

    def forward(self, logits, targets):
        return self.criterion.forward(logits, targets)



class X_Loss(nn.Module):
    """
    Args:
        logits (Tensor): Raw outputs from the model with shape (batch_size, num_classes).
        labels (Tensor): One-hot encoded labels with shape (batch_size, num_classes).
    Returns:
        Tensor: Computed loss.
    """
    def __init__(self, reduction='mean', **kwargs):
        super().__init__()
        self.reduction = reduction

    def forward(self, logits, labels):
        
        batch_size, num_classes = logits.shape

        # Detect label format and convert it to one-hot if needed
        labels = self._convert_labels(labels, num_classes)

        # Apply softmax to convert logits to probabilities
        probabilities = F.softmax(logits, dim=1).detach()  # Shape: [batch_size, num_classes]

        # Identify the class of each sample for grouping (for both hard label and soft label)
        class_indices = labels.argmax(dim=1)  # Shape: [batch_size]

        # Create a mask to group samples by class
        masks = torch.eye(num_classes, device=logits.device)[class_indices.long()]  # Shape: [batch_size, num_classes]

        # Compute the target probabilities
        target_probs = probabilities * masks  # Shape: [batch_size, num_classes]

        # Find the maximum target probability for each class
        max_target_probs, max_indices = target_probs.max(dim=0)  # Shape: [num_classes]

        # Create soft labels
        soft_labels = torch.zeros_like(probabilities)  # Initialize soft labels with zeros

        # Handle presented classes in the batch
        valid_classes = max_target_probs > 0  # Boolean mask for valid classes

        if valid_classes.any():
            valid_class_indices = valid_classes.nonzero(as_tuple=True)[0]  # Indices of valid classes
            # Create a mask for the valid classes across the batch
            valid_class_mask = masks[:, valid_class_indices]  # Shape: [batch_size, num_valid_classes]
            # Aggregate probabilities for valid classes
            selected_probs = (valid_class_mask.T @ probabilities) / valid_class_mask.sum(dim=0, keepdim=True).T  # Shape: [num_valid_classes, num_classes]
            # Assign the averaged probabilities to the corresponding classes in soft_labels
            soft_labels += valid_class_mask @ selected_probs

        # Update labels with the computed soft labels
        labels += soft_labels

        # Apply softmax to logits
        log_probs = F.log_softmax(logits, dim=1)  # Shape: [batch_size, num_classes]

        # Compute the loss
        loss = -torch.sum(labels * log_probs, dim=1)  # Shape: [batch_size]

        # Apply reduction (mean or sum)
        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        else:
            return loss  # No reduction

    def _convert_labels(self, labels, num_classes):
        """
        Automatically detect the label format and prioritise one-hot encoded labels.
        If labels are integer indices, convert them to one-hot encoded format.
        Args:
            labels (Tensor): Labels, either one-hot encoded or as integer indices.
        Returns:
            Tensor: One-hot encoded labels.
        """
        if labels.dim() == 2 and labels.size(1) == num_classes:  # One-hot encoded labels
            return labels.float()  # Prioritise and return as-is
        elif labels.dim() == 1:  # Integer labels
            # Convert to one-hot encoded format
            return F.one_hot(labels, num_classes=num_classes).float()
        else:
            raise ValueError("Labels must be either one-hot encoded or integer indices.")



class DualFocalLoss(nn.Module):
    def __init__(self, gamma=0, size_average=True, **kwargs):
        super(DualFocalLoss, self).__init__()
        self.gamma = gamma
        self.size_average = size_average


    def forward(self, input, target):
        if input.dim()>2:
            input = input.view(input.size(0),input.size(1),-1)  # N,C,H,W => N,C,H*W
            input = input.transpose(1,2)    # N,C,H*W => N,H*W,C
            input = input.contiguous().view(-1,input.size(2))   # N,H*W,C => N*H*W,C
        target = target.view(-1,1)

        logp_k = F.log_softmax(input, dim=1)
        softmax_logits = logp_k.exp()
        logp_k = logp_k.gather(1, target)
        logp_k = logp_k.view(-1)
        p_k = logp_k.exp()  # p_k: probility at target label
        p_j_mask = torch.lt(softmax_logits, p_k.reshape(p_k.shape[0], 1)) * 1  # mask all logit larger and equal than p_k
        p_j = torch.topk(p_j_mask * softmax_logits, 1)[0].squeeze()

        loss = -1 * (1 - p_k + p_j) ** self.gamma * logp_k

        if self.size_average: return loss.mean()
        else: return loss.sum()



class OLS_loss(nn.Module):
    """
    Online Label Smoothing (OLS) Loss
    Args:
        numberofclass (int): Number of classes.
        device (torch.device): Device to store tensors.
        reduction (str): Reduction method ('mean' or 'sum').
    """
    def __init__(self, numberofclass, device, reduction='mean'):
        super().__init__()
        self.numberofclass = numberofclass
        self.device = device
        self.reduction = reduction
        self.loss_lams = torch.zeros(numberofclass, numberofclass, dtype=torch.float32).to(device)
        self.loss_lams.requires_grad = False
        self.cur_epoch_lams = torch.zeros(numberofclass, numberofclass, dtype=torch.float32).to(device)
        self.cur_epoch_cnt = torch.zeros(numberofclass, dtype=torch.float32).to(device)
        self.cur_epoch_lams.requires_grad = False
        self.cur_epoch_cnt.requires_grad = False

    def reset_epoch_state(self):
        """Reset per-epoch state variables."""
        self.cur_epoch_lams.zero_()
        self.cur_epoch_cnt.zero_()

    def update_loss_lams(self, output, target):
        """Update the class-level soft label matrix based on model predictions."""
        with torch.no_grad():
            logits = torch.softmax(output, dim=1)
            sort_args = torch.argsort(logits, dim=1, descending=True)
            for k in range(output.shape[0]):
                if target[k] != sort_args[k, 0]:  # Update only if the top prediction is correct
                    continue
                self.cur_epoch_lams[target[k]] += logits[k]
                self.cur_epoch_cnt[target[k]] += 1

    def forward(self, output, target):
        """Compute the OLS loss."""
        # Update dynamic soft labels
        self.update_loss_lams(output, target)

        # Combine hard and soft losses
        loss = 0.5 * F.cross_entropy(output, target) + 0.5 * self.soft_cross_entropy(output, target)

        return loss

    def normalise_loss_lams(self):
        """Normalise the class-level soft label matrix at the end of the epoch."""
        for cls in range(self.numberofclass):
            if self.cur_epoch_cnt[cls].max() < 0.5:
                self.loss_lams[cls] = 1. / self.numberofclass  # Default uniform distribution
            else:
                self.loss_lams[cls] = self.cur_epoch_lams[cls] / self.cur_epoch_cnt[cls]

    def soft_cross_entropy(self, output, target):
        """Compute soft cross-entropy loss using dynamic soft labels."""
        target_prob = torch.zeros_like(output)
        batch = output.shape[0]
        for k in range(batch):
            target_prob[k] = self.loss_lams[target[k]]  # Use class-specific soft labels
        log_like = -torch.nn.functional.log_softmax(output, dim=1)
        loss = torch.sum(torch.mul(log_like, target_prob)) / batch
        return loss



class AdaFocal(nn.Module):
    def __init__(self, args, device, **kwargs):            
        super(AdaFocal, self).__init__()
        
        self.args = args
        self.num_bins = args.num_bins
        self.lamda = args.adafocal_lambda
        self.gamma_initial = args.adafocal_gamma_initial
        self.switch_pt = args.adafocal_switch_pt
        self.gamma_max = args.adafocal_gamma_max
        self.gamma_min = args.adafocal_gamma_min
        self.update_gamma_every = args.update_gamma_every
        self.device = device
        # self.save_path = save_path

        # This initializes the bin_stats variable
        self.bin_stats = collections.defaultdict(dict)
        for bin_no in range(self.num_bins):
            self.bin_stats[bin_no]['lower_boundary'] = bin_no*(1/self.num_bins)
            self.bin_stats[bin_no]['upper_boundary'] = (bin_no+1)*(1/self.num_bins)
            self.bin_stats[bin_no]['gamma'] = self.gamma_initial


    # This function updates the bin statistics which are used by the Adafocal loss at every epoch.
    def update_bin_stats(self, val_adabin_dict):
        for bin_no in range(self.num_bins):
            # This is the Adafocal gamma update rule
            prev_gamma = self.bin_stats[bin_no]['gamma']
            exp_term = val_adabin_dict[bin_no]['calibration_gap']
            if prev_gamma > 0:
                next_gamma = prev_gamma * math.exp(self.lamda*exp_term)
            else:
                next_gamma = prev_gamma * math.exp(-self.lamda*exp_term)    
            # This switches between focal and inverse-focal loss when required.
            if abs(next_gamma) < self.switch_pt:
                if next_gamma > 0:
                    next_gamma = -self.switch_pt
                else:
                    next_gamma = self.switch_pt
            self.bin_stats[bin_no]['gamma'] = max(min(next_gamma, self.gamma_max), self.gamma_min) # gamma-clipping
            self.bin_stats[bin_no]['lower_boundary'] = val_adabin_dict[bin_no]['lower_bound']
            self.bin_stats[bin_no]['upper_boundary'] = val_adabin_dict[bin_no]['upper_bound']
        # This saves the "bin_stats" to a text file.
        # save_file = os.path.join(self.args.save_path, "val_bin_stats.txt")
        # save_file = os.path.join(save_path, "val_bin_stats.txt")
        # with open(save_file, "a") as write_file:
        #     json.dump(self.bin_stats, write_file)
        #     write_file.write("\n")
        
        return

    # This function selects the gammas for each sample based on which bin it falls into.
    def get_gamma_per_sample(self, pt):
        gamma_list = []
        batch_size = pt.shape[0]
        for i in range(batch_size):
            pt_sample = pt[i].item()
            for bin_no, stats in self.bin_stats.items():
                if bin_no==0 and pt_sample < stats['upper_boundary']:
                    break
                elif bin_no==self.num_bins-1 and pt_sample >= stats['lower_boundary']:
                    break
                elif pt_sample >= stats['lower_boundary'] and pt_sample < stats['upper_boundary']:
                    break
            gamma_list.append(stats['gamma'])
        return torch.tensor(gamma_list).to(self.device)

    # This computes the loss value to be returned for back-propagation.
    def forward(self, input, target):
        if input.dim() > 2:
            input = input.view(input.size(0),input.size(1),-1)  # N,C,H,W => N,C,H*W
            input = input.transpose(1,2)    # N,C,H*W => N,H*W,C
            input = input.contiguous().view(-1,input.size(2))   # N,H*W,C => N*H*W,C
        target = target.view(-1,1)
        logpt = F.log_softmax(input, dim=1)
        logpt = logpt.gather(1,target)
        logpt = logpt.view(-1)
        pt = logpt.exp()

        gamma = self.get_gamma_per_sample(pt)
        gamma_sign = torch.sign(gamma)
        gamma_mag = torch.abs(gamma)
        pt = gamma_sign * pt
        loss = -1 * ((1 - pt + 1e-20)**gamma_mag) * logpt # 1e-20 added for numerical stability 
 
        # return loss.sum()
        return loss.mean()


loss_dict = {
    "focal_loss" : FocalLoss,
    "cross_entropy" : CrossEntropy,
    "LS" : LabelSmoothingLoss,
    "NLL+MDCA" : ClassficationAndMDCA,
    "LS+MDCA" : ClassficationAndMDCA,
    "FL+MDCA" : ClassficationAndMDCA,
    "brier_loss" : BrierScore,
    "NLL+DCA" : DCA,
    "MMCE" : MMCE,
    "FLSD" : FLSD,
    "XLoss": X_Loss,
    "MDCA": MDCA,
    "DFL": DualFocalLoss,
    "OLS": OLS_loss,
    "AdaFocal": AdaFocal
}


