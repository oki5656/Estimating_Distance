#============================================
__author__ = "Sachin Mehta"
__maintainer__ = "Sachin Mehta"
#============================================
import numpy as np
import torch

class MIOU(object):
    def __init__(self, num_classes=21):
        self.num_classes = num_classes
        self.epsilon = 1e-6

    def get_iou(self, output, target):
        #print("output.shape",output.shape)
        if isinstance(output, tuple):
            output = output[0]
        _, pred = torch.max(output, 1)#maxの第２引数は軸、ここではaxis=1:row?

        # histc in torch is implemented only for cpu tensors, so move your tensors to CPU
        if pred.device == torch.device('cuda'):
            pred = pred.cpu()#############################ここが怪しい
        if target.device == torch.device('cuda'):
            target = target.cpu()#########################ここが怪しい
        ##########################
        #print("type(pred)",type(pred),"  type(target)",type(target))#,"  type(inter)",type(inter))
        pred = pred.type(torch.ByteTensor)
        target = target.type(torch.ByteTensor)

        # shift by 1 so that 255 is 0
        pred += 1#pred.shape=(batch_size,480,640)   type(pred)=torch.Tensor
        target += 1#target.shape=(batch_size,480,640)    type(target)=torch.Tensor
        
        pred = pred * (target > 0)
        inter = pred * (pred == target)#inter.shape=(batch_size,480,640)   type(inter)=torch.Tensor
        area_inter = torch.histc(inter.float(), bins=self.num_classes, min=1, max=self.num_classes)
        area_pred = torch.histc(pred.float(), bins=self.num_classes, min=1, max=self.num_classes)
        area_mask = torch.histc(target.float(), bins=self.num_classes, min=1, max=self.num_classes)
        area_union = area_pred + area_mask - area_inter + self.epsilon
        #type(area_union)=torch.Tensor  area_union.shape=torch.Size([2]) len(area_union)=2)

        #print("type(pred)",type(pred),"  type(target)",type(target),"  type(inter)",type(inter))
        #print("pred.shape",pred.shape," target.shape",target.shape," inter.shape",inter.shape)
        #print("area_union.shape",len(area_union))

        return area_inter.numpy(), area_union.numpy()#たぶんndarrayに変換しているだけ


if __name__ == '__main__':
    from utilities.utils import AverageMeter
    inter = AverageMeter()
    union = AverageMeter()
    a = torch.Tensor(1, 21, 224, 224).random_(254, 256)
    b = torch.Tensor(1, 21, 224, 224).random_(254, 256 )

    m = MIOU()
    print(m.get_iou(a, b))
