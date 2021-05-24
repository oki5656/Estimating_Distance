# ============================================
__author__ = "Sachin Mehta"
__maintainer__ = "Sachin Mehta"
# ============================================

model_weight_map = {}
# key is of the form <model-name_model-scale>


#ESPNetv2
espnetv2_scales = [2.0, 0.5, 1.0, 1.5, 2.0]
for scale in espnetv2_scales:
    model_weight_map['espnetv2_{}'.format(scale)] = {
        'mydataset_640x480':
        {
            #################################   用変更   ###################################
            #'weights': 'results_segmentation/model_espnetv2_mydataset/s_2.0_sch_hybrid_loss_ce_res_640_sc_1.0_2.0/20210514-181637/espnetv2_{}_640_best.pth'.format(scale)
            'weights': 'results_segmentation/model_espnetv2_mydataset/s_2.0_sch_hybrid_loss_ce_res_640_sc_1.0_2.0/20210423-145501/espnetv2_{}_640_best.pth'.format(scale)
            #rootからのパスを記入
        },
        'pascal_256x256':
        {
            'weights': 'model/segmentation/model_zoo/espnetv2/espnetv2_s_{}_pascal_256x256.pth'.format(scale)
        },
        'pascal_384x384':
        {
            'weights': 'model/segmentation/model_zoo/espnetv2/espnetv2_s_{}_pascal_384x384.pth'.format(scale)
        },
        'city_1024x512': {
            'weights': 'model/segmentation/model_zoo/espnetv2/espnetv2_s_{}_city_1024x512.pth'.format(scale)
        },
        'city_512x256': {
            'weights': 'model/segmentation/model_zoo/espnetv2/espnetv2_s_{}_city_512x256.pth'.format(scale)
        }
        
    }

#DiCENet

dicenet_scales = [0.5, 0.75, 1.0, 1.25, 1.5, 1.75]
for scale in dicenet_scales:
    model_weight_map['dicenet_{}'.format(scale)] = {
        'pascal_256x256':
        {
            'weights': 'model/segmentation/model_zoo/dicenet/dicenet_s_{}_pascal_256x256.pth'.format(scale)
        },
        'pascal_384x384':
        {
            'weights': 'model/segmentation/model_zoo/dicenet/dicenet_s_{}_pascal_384x384.pth'.format(scale)
        },
        'city_1024x512': {
            'weights': 'model/segmentation/model_zoo/dicenet/dicenet_s_{}_city_1024x512.pth'.format(scale)
        },
        'city_512x256': {
            'weights': 'model/segmentation/model_zoo/dicenet/dicenet_s_{}_city_512x256.pth'.format(scale)
        }
    }
