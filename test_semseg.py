import torch
import glob
import os
import cv2
import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from argparse import ArgumentParser
from PIL import Image
from torchvision.transforms import functional as F
from tqdm import tqdm
from utilities.print_utils import *
from transforms.classification.data_transforms import MEAN, STD
from utilities.utils import model_parameters, compute_flops

from commons.general_details import segmentation_models, segmentation_datasets
from model.weight_locations.segmentation import model_weight_map#weight file 読み込み
from data_loader.segmentation.mydataset import MY_DATASET_CLASS_LIST


def relabel(img):
    '''
    This function relabels the predicted labels so that cityscape dataset can process
    :param img:
    :return:
    '''
    img[img == 19] = 255
    img[img == 18] = 33
    img[img == 17] = 32
    img[img == 16] = 31
    img[img == 15] = 28
    img[img == 14] = 27
    img[img == 13] = 26
    img[img == 12] = 25
    img[img == 11] = 24
    img[img == 10] = 23
    img[img == 9] = 22
    img[img == 8] = 21
    img[img == 7] = 20
    img[img == 6] = 19
    img[img == 5] = 17
    img[img == 4] = 13
    img[img == 3] = 12
    img[img == 2] = 11
    img[img == 1] = 8
    img[img == 0] = 7
    img[img == 255] = 0
    return img


def data_transform(img, im_size):
    img = img.resize(im_size, Image.BILINEAR)
    img = F.to_tensor(img)  # convert to tensor (values between 0 and 1)
    img = F.normalize(img, MEAN, STD)  # normalize the tensor
    return img


def evaluate(args, model, image_list, device, IMG, ver, hor):
    im_size = tuple(args.im_size)

    # get color map for my dataset
    if args.dataset == 'mydataset':
        from utilities.color_map import MYColormap
        cmap = MYColormap().get_color_map_my()
    else:
        cmap = None
    
    # get color map for pascal dataset ######################################
    model.eval()
    for i, imgName in tqdm(enumerate(image_list)):
        #print("imgName=",imgName)
        img = Image.open(imgName).convert('RGB')#imgNameはtest_semseg.pyから画像までのパス
        img = Image.fromarray(np.uint8(IMG))
        w, h = img.size

        img = data_transform(img, im_size)
        img = img.unsqueeze(0)  # add a batch dimension
        img = img.to(device)
        img_out = model(img)##################################################
        img_out = img_out.squeeze(0)  # remove the batch dimension
        img_out = img_out.max(0)[1].byte()  # get the label map
        img_out = img_out.to(device='cpu').numpy()

        img_out = Image.fromarray(img_out)
        img_out = img_out.resize((w, h), Image.NEAREST)

        if args.dataset == 'mydataset':
            img_out.putpalette(cmap)

        # save the segmentation mask
        name = imgName.split('/')[-1]
        img_extn = imgName.split('.')[-1]
        name = '{}/{}'.format(args.savedir, name.replace(img_extn, 'png'))
        img_out.save(name)
        segmap=img_out
        img_out=np.asarray(img_out)

        # Apply KMeans
        ############ config #############
        rate=0.5#リサイズ
        #################################


        ########## 読み込み&リサイズ&フィルタリング #########
        img = cv2.medianBlur(img_out, 7)
        height = img.shape[0]
        width = img.shape[1]
        #print("height,width=",height,width)#1512 2016 or 2016 1512
        HEIGHT=rate*height
        WIDTH=rate*width
        img = cv2.resize(img , (int(WIDTH), int(HEIGHT)))

        ###### cv2.kmeansに合うように変換 ######
        f_list=list(zip(*np.where(img == 1)))#################
        f_list=np.asarray(f_list)
        f_list=np.float32(f_list)

        # define criteria and apply kmeans()
        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 5, 1.0)
        ret,label,center=cv2.kmeans(f_list,2,None,criteria,10,cv2.KMEANS_RANDOM_CENTERS)

        if int(ver) == 0 and int(hor) == 0:
            ver,hor=50,63
        else:
            ver,hor=int(ver),int(hor)

        # Now separate the data, Note the flatten()
        A,B = f_list[label.ravel()==0],f_list[label.ravel()==1]
        S1,S2=A.shape[0]/(WIDTH*HEIGHT),B.shape[0]/(WIDTH*HEIGHT)#画像に対するボール（個々）の面積の割合
        L1,L2=np.sqrt(0.001135/S1),np.sqrt(0.001135/S2)#0.001135=0.00454*0.25=S0*0.5^2
        x1=np.sqrt(hor**2+ver**2)#iphoneのカメラの画角
        x2=np.sqrt((center[0][0]-center[1][0])**2+(center[0][1]-center[1][1])**2)#取得した画像内の距離
        x3=np.sqrt(WIDTH**2+HEIGHT**2)#にゅうりょく画像の対角線MAX距離
        theta=x1*x2/x3
        ans=np.sqrt(L1**2+L2**2-2*L1*L2*np.cos(np.radians(theta)))

        return ans , segmap

def main(image,v,h):
    parser = ArgumentParser()
    # mdoel details
    parser.add_argument('--model', default="espnetv2", choices=segmentation_models, help='Model name')
    parser.add_argument('--weights-test', default='', help='Pretrained weights directory.')
    parser.add_argument('--s', default=2.0, type=float, help='scale')
    # dataset details
    parser.add_argument('--data-path', default="vision_datasets/mydataset/", help='Data directory')
    parser.add_argument('--dataset', default='mydataset', choices=segmentation_datasets, help='Dataset name')
    # input details
    parser.add_argument('--im-size', type=int, nargs="+", default=[640,480], help='Image size for testing (W x H)')
    parser.add_argument('--split', default='val', choices=['val', 'test'], help='data split')
    parser.add_argument('--model-width', default=224, type=int, help='Model width')
    parser.add_argument('--model-height', default=224, type=int, help='Model height')
    parser.add_argument('--channels', default=3, type=int, help='Input channels')
    parser.add_argument('--num-classes', default=1000, type=int,
                        help='ImageNet classes. Required for loading the base network')
    args = parser.parse_args()

    if not args.weights_test:
        #from model.weight_locations.segmentation import model_weight_map#weight file 読み込み
        model_key = '{}_{}'.format(args.model, args.s)
        dataset_key = '{}_{}x{}'.format(args.dataset, args.im_size[0], args.im_size[1])
        assert model_key in model_weight_map.keys(), '{} does not exist'.format(model_key)
        assert dataset_key in model_weight_map[model_key].keys(), '{} does not exist'.format(dataset_key)
        args.weights_test = model_weight_map[model_key][dataset_key]['weights']
        if not os.path.isfile(args.weights_test):
            print('weight file does not exist: {}'.format(args.weights_test))
    # set-up results path
    if args.dataset == 'mydataset':
        ################################# result folder name  ###########################################
        args.savedir = '{}_{}/results/comp6_{}_cls'.format('results', args.dataset, args.split)
        ######################################################################################################
    else:
        print('{} dataset not yet supported'.format(args.dataset))

    if not os.path.isdir(args.savedir):
        os.makedirs(args.savedir)
    # This key is used to load the ImageNet weights while training. So, set to empty to avoid errors
    args.weights = ''


    # read all the images in the folder
    if args.dataset == 'mydataset':
        seg_classes = len(MY_DATASET_CLASS_LIST)
        data_file = os.path.join(args.data_path,'{}.txt'.format(args.split))
        #print("data file=",data_file)
        if not os.path.isfile(data_file):
            print('{} file does not exist'.format(data_file))
        image_list = []
        with open(data_file, 'r') as lines:
            for line in lines:
                rgb_img_loc = '{}/{}/{}'.format(args.data_path, 'images', line.split(',')[0])#############
                print("rgb_img_loc=",rgb_img_loc)
                if not os.path.isfile(rgb_img_loc):
                    print('{} image file does not exist'.format(rgb_img_loc))
                image_list.append(rgb_img_loc)
    else:
        print('{} dataset not yet supported'.format(args.dataset))

    print('# of images for testing: {}'.format(len(image_list)))

    if args.model == 'espnetv2':
        from model.segmentation.espnetv2 import espnetv2_seg
        args.classes = seg_classes
        model = espnetv2_seg(args)
    else:
        print('{} network not yet supported'.format(args.model))
        exit(-1)

    # mdoel information
    num_params = model_parameters(model)
    flops = compute_flops(model, input=torch.Tensor(1, 3, args.im_size[0], args.im_size[1]))
    print('FLOPs for an input of size {}x{}: {:.2f} million'.format(args.im_size[0], args.im_size[1], flops))
    print('# of parameters: {}'.format(num_params))

    if args.weights_test:
        print('Loading model weights')
        weight_dict = torch.load(args.weights_test, map_location=torch.device('cpu'))
        model.load_state_dict(weight_dict)
        print('Weight loaded successfully')
        print("loaded:",args.weights_test)
    else:
        print('weight file does not exist or not specified. Please check: {}', format(args.weights_test))

    num_gpus = torch.cuda.device_count()
    device = 'cuda' if num_gpus > 0 else 'cpu'
    model = model.to(device=device)
    #print("image_list",image_list)
    ans=evaluate(args, model, image_list, device=device, IMG=image, ver=v, hor=h)####################
    return ans
