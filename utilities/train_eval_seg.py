#============================================
__author__ = "Sachin Mehta"
__maintainer__ = "Sachin Mehta"
#============================================

import torch
from utilities.utils import AverageMeter
import time
from utilities.metrics.segmentation_miou import MIOU
from utilities.print_utils import *
from torch.nn.parallel import gather

def train_seg(model, dataset_loader, optimizer, criterion, num_classes, epoch, device='cuda'):
    losses = AverageMeter()#AverageMeter()はクラス。ここはインスタンスを作成している
    #AverageMeter : Computes and stores the average and current value
    """
class AverageMeter(object):
    Computes and stores the average and current value
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count
    """
    batch_time = AverageMeter()
    inter_meter = AverageMeter()
    union_meter = AverageMeter()
    end = time.time()
    model.train()

    miou_class = MIOU(num_classes=num_classes)

    for i, (inputs, target) in enumerate(dataset_loader):# enumerate?インデックスと要素を取得できる
        inputs = inputs.to(device=device)
        target = target.to(device=device)

        outputs = model(inputs)

        if device == 'cuda':
            loss = criterion(outputs, target).mean()
            if isinstance(outputs, (list, tuple)):
                target_dev = outputs[0].device
                outputs = gather(outputs, target_device=target_dev)
        else:
            loss = criterion(outputs, target)

        inter, union = miou_class.get_iou(outputs, target)

        inter_meter.update(inter)#.update()は集合をappendする感じ
        union_meter.update(union)

        losses.update(loss.item(), inputs.size(0))

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if i % 10 == 0:  # print after every 100 batches
            iou = inter_meter.sum / (union_meter.sum + 1e-10)
            miou = iou.mean() * 100
            print_log_message("train Epoch: %d[%d/%d]\t\tBatch Time:%.4f\t\tLoss:%.4f\t\tmiou:%.4f" %
                  (epoch, i, len(dataset_loader), batch_time.avg, losses.avg, miou))

    iou = inter_meter.sum / (union_meter.sum + 1e-10)
    miou = iou.mean() * 100
    return miou, losses.avg


def val_seg(model, dataset_loader, criterion=None, num_classes=21, device='cuda'):
    model.eval()
    inter_meter = AverageMeter()#AverageMeter()はクラス。ここはインスタンスを作成している
    #AverageMeter : Computes and stores the average and current value
    union_meter = AverageMeter()
    batch_time = AverageMeter()
    end = time.time()

    miou_class = MIOU(num_classes=num_classes)

    if criterion:
        losses = AverageMeter()

    with torch.no_grad():#評価を行うために勾配を計算しないようにしている？　多分よく使われる
        for i, (inputs, target) in enumerate(dataset_loader):#enumerateを用いるとインデックス番号、要素を順に取得できる
            # iはbatchごと(batchsize=16ならi++ごとに画像16枚)に回っている
            # 
            inputs = inputs.to(device=device)
            target = target.to(device=device)
            outputs = model(inputs)

            if criterion:
                if device == 'cuda':
                    loss = criterion(outputs, target).mean()
                    if isinstance(outputs, (list, tuple)):
                        target_dev = outputs[0].device
                        outputs = gather(outputs, target_device=target_dev)
                else:
                    loss = criterion(outputs, target)
                losses.update(loss.item(), inputs.size(0))

            inter, union = miou_class.get_iou(outputs, target)
            inter_meter.update(inter)
            union_meter.update(union)

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

            if i % 10 == 0:  # print after every 100 batches    ←１０じゃね？
            # epochの中の10batchesごとの変化を見るためにある
            # 10batch=batchsize*batch=16*10=160 つまりi==0の時だけしか表示されていない
                iou = inter_meter.sum / (union_meter.sum + 1e-10)
                miou = iou.mean() * 100
                loss_ = losses.avg if criterion is not None else 0
                print_log_message("val10batch  [%d/%d]\t\tBatch Time:%.4f\t\tLoss:%.4f\t\tmiou:%.4f" %
                      (i, len(dataset_loader), batch_time.avg, loss_, miou))###################################

    iou = inter_meter.sum / (union_meter.sum + 1e-10)
    miou = iou.mean() * 100

    print_info_message('Mean IoU: {0:.2f}'.format(miou))#######################################################
    if criterion:
        return miou, losses.avg
    else:
        return miou, 0