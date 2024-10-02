import math, time
import random
from itertools import chain
import torch
import torch.nn.functional as F
from torch import nn
from base import BaseModel
from utils.helpers import *
from utils.losses import *
from models.decoder import *
from models.encoder import *


class CCFF(BaseModel):
    def __init__(self, num_classes, conf, loss_l=None, loss_alg=None, len_unsper=None, testing=False, pretrained=True):
        self.num_classes = num_classes
        if not testing:
            assert (loss_l is not None)

        super(CCFF, self).__init__()
        self.method = conf['method']

        # Supervised and unsupervised losses
        self.loss_l = loss_l
        self.loss_alg = loss_alg

        # confidence masking (sup mat)
        if self.method != 'supervised':
            self.confidence_thr = conf['confidence_thr']
            print('thr: ', self.confidence_thr)

        # Create the model
        self.encoder = Encoder_ResNet50(pretrained=pretrained)
        self.scSE_Module = scSE_Module(15)
        # The main encoder
        upscale = 8
        num_out_ch = 2048
        decoder_in_ch = num_out_ch // 4
        self.decoder = Decoder(upscale, decoder_in_ch, num_classes=num_classes)
        self.concatDecoder = ConcatDecoder(upscale, decoder_in_ch, num_classes=num_classes)

    def forward(self, A_l=None, B_l=None, target_l=None, \
                WA_ul=None, WB_ul=None, SA_ul=None, SB_ul=None, target_ul=None, CTCD=None):
        if not self.training:
            return self.decoder(self.encoder(A_l, B_l))
        input_size = (A_l.size(2), A_l.size(3))

        if self.method == 'supervised':
            out_feat_l = self.encoder(A_l, B_l)
            out_l = self.decoder(out_feat_l)
            loss_l = self.loss_l(out_l, target_l)
            if out_l.shape != A_l.shape:
                out_l = F.interpolate(out_l, size=input_size, mode='bilinear', align_corners=True)
            outs = {'pred_l': out_l}
            curr_losses = {'loss_l': loss_l}
            total_loss = loss_l
            return total_loss, curr_losses, outs

        else:
            #Supervised loss
            out_feat_l=self.encoder(A_l,B_l)
            out_l = self.decoder(out_feat_l)
            loss_l_1 = self.loss_l(out_l, target_l)
            # Ensemble Supervised Loss
            WA_l_Cross = self.scSE_Module(CTCD(A_l))
            WB_l_Cross = self.scSE_Module(CTCD(B_l))
            WAB_l = torch.cat((WA_l_Cross, WB_l_Cross), dim=1)
            Cross_l = self.concatDecoder(out_feat_l.detach_(), WAB_l)
            loss_l_2 = self.loss_l(Cross_l, target_l)
            # Overall supervised Loss
            loss_l=loss_l_1+loss_l_2

            # Unsupervised loss
            
            weak_feat_ul = self.encoder(WA_ul, WB_ul)
            weak_out_ul = self.decoder(weak_feat_ul)
            strong_feat_ul = self.encoder(SA_ul, SB_ul)
            strong_out_ul = self.decoder(strong_feat_ul)

            # Ensemble Unsupervised Loss
            WA_ul_Cross = self.scSE_Module(CTCD(WA_ul))
            WB_ul_Cross = self.scSE_Module(CTCD(WB_ul))
            WAB_ul = torch.cat((WA_ul_Cross, WB_ul_Cross), dim=1)
            Cross_ul = self.concatDecoder(weak_feat_ul.detach_(), WAB_ul)

            Cross_prob_ul = F.softmax(Cross_ul.detach_(), dim=1)
            max_probs_Cross, target_ul_Cross = torch.max(Cross_prob_ul, dim=1)
            mask_Cross = max_probs_Cross.ge(self.confidence_thr).float()
            loss_ul_cls_2 = (F.cross_entropy(weak_out_ul, target_ul_Cross, reduction='none') * mask_Cross).mean()

            # Standard Unsupervised Loss
            weak_prob_ul = F.softmax(weak_out_ul.detach_(), dim=1)
            max_probs, target_ul = torch.max(weak_prob_ul, dim=1)
            mask = max_probs.ge(self.confidence_thr).float()
            loss_ul_cls_1 = (F.cross_entropy(strong_out_ul, target_ul, reduction='none') * mask).mean()

            # Overall Consistency Loss
            loss_ul_cls = loss_ul_cls_1 + loss_ul_cls_2

            # Feature-Alignment Loss
            loss_ul_alg = self.loss_alg(weak_prob_ul, strong_feat_ul, self.confidence_thr)

            # Overall Unsupervised Loss
            loss_ul = loss_ul_cls + loss_ul_alg

            # record loss
            curr_losses = {'loss_l': loss_l}
            curr_losses['loss_ul'] = loss_ul
            curr_losses['loss_ul_cls'] = loss_ul_cls
            curr_losses['loss_ul_alg'] = loss_ul_alg

            if weak_out_ul.shape != WA_ul.shape:
                out_l = F.interpolate(out_l, size=input_size, mode='bilinear', align_corners=True)
                weak_out_ul = F.interpolate(weak_out_ul, size=input_size, mode='bilinear', align_corners=True)
            outs = {'pred_l': out_l, 'pred_ul': weak_out_ul}

            # Compute the unsupervised loss
            total_loss = loss_l + loss_ul

            return total_loss, curr_losses, outs

    def get_backbone_params(self):
        return self.encoder.get_backbone_params()

    def get_other_params(self):
        return chain(self.encoder.get_module_params(), self.decoder.parameters(), self.concatDecoder.parameters(),
                     self.scSE_Module.parameters())

