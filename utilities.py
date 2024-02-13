import torch
from torch.autograd import Variable
import torch.nn.functional as F
import matplotlib.pyplot as plt
# loss functions
# apply softmax on network output for dice, not CE
from torch import nn, Tensor
import numpy as np

softmax_helper = lambda x: F.softmax(x, 1)

def sum_tensor(inp, axes, keepdim=False):
    axes = np.unique(axes).astype(int)
    if keepdim:
        for ax in axes:
            inp = inp.sum(int(ax), keepdim=True)
    else:
        for ax in sorted(axes, reverse=True):
            inp = inp.sum(int(ax))
    return inp


def get_tp_fp_fn_tn(net_output, gt, axes=None, mask=None, square=False):
    """
    net_output must be (b, c, x, y(, z)))
    gt must be a label map (shape (b, 1, x, y(, z)) OR shape (b, x, y(, z))) or one hot encoding (b, c, x, y(, z))
    if mask is provided it must have shape (b, 1, x, y(, z)))
    :param net_output:
    :param gt:
    :param axes: can be (, ) = no summation
    :param mask: mask must be 1 for valid pixels and 0 for invalid pixels
    :param square: if True then fp, tp and fn will be squared before summation
    :return:
    """
    if axes is None:
        axes = tuple(range(2, len(net_output.size())))

    shp_x = net_output.shape
    shp_y = gt.shape

    with torch.no_grad():
        if len(shp_x) != len(shp_y):
            gt = gt.view((shp_y[0], 1, *shp_y[1:]))

        if all([i == j for i, j in zip(net_output.shape, gt.shape)]):
            # if this is the case then gt is probably already a one hot encoding
            y_onehot = gt
        else:
            gt = gt.long()
            y_onehot = torch.zeros(shp_x)
            if net_output.device.type == "cuda":
                y_onehot = y_onehot.cuda(net_output.device.index)
            y_onehot.scatter_(1, gt, 1)

    tp = net_output * y_onehot
    fp = net_output * (1 - y_onehot)
    fn = (1 - net_output) * y_onehot
    tn = (1 - net_output) * (1 - y_onehot)

    if mask is not None:
        tp = torch.stack(tuple(x_i * mask[:, 0] for x_i in torch.unbind(tp, dim=1)), dim=1)
        fp = torch.stack(tuple(x_i * mask[:, 0] for x_i in torch.unbind(fp, dim=1)), dim=1)
        fn = torch.stack(tuple(x_i * mask[:, 0] for x_i in torch.unbind(fn, dim=1)), dim=1)
        tn = torch.stack(tuple(x_i * mask[:, 0] for x_i in torch.unbind(tn, dim=1)), dim=1)

    if square:
        tp = tp ** 2
        fp = fp ** 2
        fn = fn ** 2
        tn = tn ** 2

    if len(axes) > 0:
        tp = sum_tensor(tp, axes, keepdim=False)
        fp = sum_tensor(fp, axes, keepdim=False)
        fn = sum_tensor(fn, axes, keepdim=False)
        tn = sum_tensor(tn, axes, keepdim=False)

    return tp, fp, fn, tn


class SoftDiceLoss(nn.Module):
    def __init__(self, apply_nonlin=None, batch_dice=False, do_bg=True, smooth=1.):
        """
        """
        super(SoftDiceLoss, self).__init__()

        self.do_bg = do_bg
        self.batch_dice = batch_dice
        self.apply_nonlin = apply_nonlin
        self.smooth = smooth

        print("batch_dice: {}\ndo_bg: {}\n".format(self.batch_dice, self.do_bg))

    def forward(self, x, y, loss_mask=None):
        shp_x = x.shape

        if self.batch_dice:
            axes = [0] + list(range(2, len(shp_x)))
        else:
            axes = list(range(2, len(shp_x)))

        #print("[SAUMDEBUG]\naxes: {}\n".format(axes))
        if self.apply_nonlin is not None:
            x = self.apply_nonlin(x)
            #print("[SAUMDEBUG]\napply_nonlin called")

        tp, fp, fn, _ = get_tp_fp_fn_tn(x, y, axes, loss_mask, False)

        nominator = 2 * tp + self.smooth
        denominator = 2 * tp + fp + fn + self.smooth

        dc = nominator / (denominator + 1e-8)

        if not self.do_bg:
            if self.batch_dice:
                dc = dc[1:]
            else:
                dc = dc[:, 1:]
        dc = dc.mean()
        #print("[SAUMDEBUG]\ndc without manipulation: {}\n -dc: {}\n".format(dc, -dc))
        return -dc


class RobustCrossEntropyLoss(nn.CrossEntropyLoss):
    """
    this is just a compatibility layer because my target tensor is float and has an extra dimension
    """
    def forward(self, input: Tensor, target: Tensor) -> Tensor:
        if len(target.shape) == len(input.shape):
            assert target.shape[1] == 1
            target = target[:, 0]
        return super().forward(input, target.long())

class DC_and_CE_loss(nn.Module):
    def __init__(self, soft_dice_kwargs, ce_kwargs, aggregate="sum", square_dice=False, weight_ce=1, weight_dice=1,
                 log_dice=False, ignore_label=None):
        """
        CAREFUL. Weights for CE and Dice do not need to sum to one. You can set whatever you want.
        :param soft_dice_kwargs:
        :param ce_kwargs:
        :param aggregate:
        :param square_dice:
        :param weight_ce:
        :param weight_dice:
        """
        super(DC_and_CE_loss, self).__init__()
        if ignore_label is not None:
            assert not square_dice, 'not implemented'
            ce_kwargs['reduction'] = 'none'
        self.log_dice = log_dice
        self.weight_dice = weight_dice
        self.weight_ce = weight_ce
        self.aggregate = aggregate
        self.ce = RobustCrossEntropyLoss(**ce_kwargs)

        self.ignore_label = ignore_label

        self.dc = SoftDiceLoss(apply_nonlin=softmax_helper, **soft_dice_kwargs)

    def forward(self, net_output, target):
        """
        target must be b, c, x, y(, z) with c=1
        :param net_output:
        :param target:
        :return:
        """
        if self.ignore_label is not None:
            assert target.shape[1] == 1, 'not implemented for one hot encoding'
            mask = target != self.ignore_label
            target[~mask] = 0
            mask = mask.float()
        else:
            mask = None

        dc_loss = self.dc(net_output, target, loss_mask=mask) if self.weight_dice != 0 else 0
        if self.log_dice:
            dc_loss = -torch.log(-dc_loss)

        ce_loss = self.ce(net_output, target[:, 0].long()) if self.weight_ce != 0 else 0
        if self.ignore_label is not None:
            ce_loss *= mask[:, 0]
            ce_loss = ce_loss.sum() / mask.sum()

        if self.aggregate == "sum":
            result = self.weight_ce * ce_loss + self.weight_dice * dc_loss
            #print("[SAUMDEBUG]\nDC_and_CE_loss\nweight_ce: {}\nce_loss: {}\nweight_dice: {}\ndc_loss: {}\n".format(self.weight_ce, ce_loss, self.weight_dice, dc_loss))
        else:
            raise NotImplementedError("nah son") # reserved for other stuff (later)
        return result



def torch_dice_fn(pred, target): #pytorch tensors NCDHW
    pred = torch.argmax(softmax_helper(pred),dim=1)
    num = pred.size(0)
    m1 = pred.view(num, -1).float()  # Flatten
    m2 = target.view(num, -1).float()  # Flatten
    intersection = (m1 * m2).sum().float()

    return (2. * intersection) / (m1.sum() + m2.sum())

# not doing softmax because for ce binary training, num_classes = 1
# ideally for bce you should do some threshold on the values to get pred in binary form. 
def torch_dice_fn_bce(pred, target): #pytorch tensors NCDHW
    num = pred.size(0)
    m1 = pred.view(num, -1).float()  # Flatten
    m2 = target.view(num, -1).float()  # Flatten
    intersection = (m1 * m2).sum().float()

    return (2. * intersection) / (m1.sum() + m2.sum())

# loss function for aleatoric [regression]
# from the paper: https://proceedings.neurips.cc/paper/2017/file/2650d6089a6d640c5e85b2b88265dc2b-Paper.pdf
# from the code: https://github.com/hmi88/what/blob/master/WHAT_src/loss/mse_var.py
class MSE_VAR(nn.Module):
    def __init__(self, var_weight=1.):
        super(MSE_VAR, self).__init__()
        self.var_weight = var_weight

    def forward(self, mu, log_var, label):
        log_var = self.var_weight * log_var

        loss1 = torch.mul(torch.exp(-log_var), (mu - label) ** 2)
        loss2 = log_var
        loss = .5 * (loss1 + loss2)
        return loss.mean()

# loss function for aleatoric [classification]
# from the paper: https://proceedings.neurips.cc/paper/2017/file/2650d6089a6d640c5e85b2b88265dc2b-Paper.pdf
# own implementation
# inspired by: https://github.com/kyle-dorman/bayesian-neural-network-blogpost (for the loss) and https://github.com/geyang/variational_autoencoder_pytorch (for the reparametrization trick)
# Unsure when the cross-entropy should be applied --- before averaging across T samples or after; doing it after for now
class Aleatoric_Classification_Loss(nn.Module):
    def __init__(self, T):
        super(Aleatoric_Classification_Loss, self).__init__()
        self.crossentropy = torch.nn.CrossEntropyLoss(size_average = False, reduce=False, reduction=None)
        self.T = T
        self.softmax = torch.nn.Softmax(dim=1)
        self.elu = torch.nn.ELU()

    def forward(self, mu, sigma, label, device): # mu and sigma are NCHW

        ans = []
        undistorted_loss = self.crossentropy(mu,label)
        for t in range(self.T):
            epsilon = torch.normal(0., 1., size=mu.shape).to(device)
            sampled_pred = mu + sigma*epsilon
            sampled_pred = self.softmax(sampled_pred) # along channel dim
            #ans.append(self.crossentropy(sampled_pred, label))
            ans.append(sampled_pred)
        
        sampled_pred = torch.mean(torch.stack(ans, dim=0), dim=0) # NCHW
        distorted_loss = self.crossentropy(sampled_pred,label)

        diff = -self.elu(undistorted_loss - distorted_loss)
        lossval = diff * undistorted_loss

        return lossval.mean()#*0.5 + sigma.mean()*0.5
        



def truncated_normal_(tensor, mean=0, std=1):
    size = tensor.shape
    tmp = tensor.new_empty(size + (4,)).normal_()
    valid = (tmp < 2) & (tmp > -2)
    ind = valid.max(-1, keepdim=True)[1]
    tensor.data.copy_(tmp.gather(-1, ind).squeeze(-1))
    tensor.data.mul_(std).add_(mean)

def init_weights(m):
    if type(m) == nn.Conv2d or type(m) == nn.ConvTranspose2d:
        nn.init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='relu')
        #nn.init.normal_(m.weight, std=0.001)
        #nn.init.normal_(m.bias, std=0.001)
        truncated_normal_(m.bias, mean=0, std=0.001)

def init_weights_orthogonal_normal(m):
    if type(m) == nn.Conv2d or type(m) == nn.ConvTranspose2d:
        nn.init.orthogonal_(m.weight)
        truncated_normal_(m.bias, mean=0, std=0.001)
        #nn.init.normal_(m.bias, std=0.001)

def l2_regularisation(m):
    l2_reg = None

    for W in m.parameters():
        if l2_reg is None:
            l2_reg = W.norm(2)
        else:
            l2_reg = l2_reg + W.norm(2)
    return l2_reg

def save_mask_prediction_example(mask, pred, iter):
	plt.imshow(pred[0,:,:],cmap='Greys')
	plt.savefig('images/'+str(iter)+"_prediction.png")
	plt.imshow(mask[0,:,:],cmap='Greys')
	plt.savefig('images/'+str(iter)+"_mask.png")