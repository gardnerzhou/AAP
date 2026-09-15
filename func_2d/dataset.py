""" train and test dataset

author jundewu
"""
import os

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset
import json

class SUNSEG(Dataset):
    def __init__(self, args, data_path ,raw_data_path,  transform = None, transform_msk = None, mode = 'train',prompt = 'click', json = False):
        if json :
            self.data_path = raw_data_path
            image_paths = self.get_frames_from_csv(os.path.join('/data/whl/Medical-SAM2-main/func_2d/sunseg_split', mode+'_frames.json'))
            self.images = [os.path.join(self.data_path , f) for f in image_paths if f.endswith('.png') or f.endswith('.jpg') ]
            self.gts = [f.replace('Frame','GT').replace('.jpg','.png') for f in self.images]
            self.images = sorted(self.images)
            self.gts = sorted(self.gts)
            if mode == "train":
                self.gts_weak = [f.replace('GT','Scribble') for f in self.gts]
            if mode == "train":
                self.gts_weak = sorted(self.gts_weak)
                self.filter_files_train()
            else:
                self.filter_files()
        else:
            self.data_path = data_path
            img_path = os.path.join(data_path,'images')
            self.images = [os.path.join(img_path , f) for f in os.listdir(img_path) if f.endswith('.jpg') or f.endswith('.png')]
            gt_path = os.path.join(data_path,'annotations')
            self.gts = [os.path.join(gt_path , f) for f in os.listdir(gt_path) if f.endswith('.png') or f.endswith('.jpg')]
            if mode == "train":
                gt_weak_path = os.path.join(data_path,'scribble')
                self.gts_weak = [os.path.join(gt_weak_path , f) for f in os.listdir(gt_weak_path) if f.endswith('.png') or f.endswith('.jpg')]
            self.images = sorted(self.images)
            self.gts = sorted(self.gts)
        
            if mode == "train":
                self.gts_weak = sorted(self.gts_weak)
                self.filter_files_train()
            else:
                self.filter_files()
        
        self.mode = mode
        self.prompt = prompt
        self.img_size = args.image_size
        self.mask_size = args.out_size

        self.transform = transform
        self.transform_msk = transform_msk

    def __len__(self):
        return len(self.images)
    
    def filter_files(self):
        assert len(self.images) == len(self.gts)
        self.images = self.images
        self.gts = self.gts
        
    def get_frames_from_csv(self, csv_path):
        frame_paths = []
        frame_num = 0
        with open(csv_path) as csv:
            data_dict = json.load(csv)
        case_list = list(data_dict.keys())
        for case in case_list:
            frame_paths.extend(data_dict[case][0])
            frame_num += data_dict[case][1]
        assert len(frame_paths) == frame_num, 'len(frame_paths) != frame_num'

        return frame_paths
    
    def filter_files_train(self):
        assert len(self.images) == len(self.gts) == len(self.gts_weak)
        self.images = self.images
        self.gts = self.gts
        self.gts_weak = self.gts_weak
        
    def __getitem__(self, index):

        """Get the images"""
        img_path = self.images[index]
        name = img_path.split('/')[-1]

        # raw image and raters path
        # img_path = os.path.join(subfolder, name )
        # multi_rater_cup_path = [os.path.join(subfolder, name + '_seg_cup_' + str(i) + '_cropped.jpg') for i in range(1, 8)]

        # img_path = os.path.join(subfolder, name + '.jpg')
        # multi_rater_cup_path = [os.path.join(subfolder, name + '_seg_cup_' + str(i) + '.png') for i in range(1, 8)]
        # print(name)
        # raw image and rater images
        img = Image.open(img_path).convert('RGB')
        raw_img = img.copy()
        raw_img=torch.as_tensor((self.transform_msk(raw_img) ).float(), dtype=torch.float32)
        
        gt_path = self.gts[index]
        gt = Image.open(gt_path).convert('1')
        
        if self.mode == "train":
            gt_weak_path = self.gts_weak[index]
            gt_weak = Image.open(gt_weak_path).convert('L')        
        # gt = (gt==255)
        # multi_rater_cup = [Image.open(path).convert('L') for path in multi_rater_cup_path]

        # apply transform
        if self.transform:
            # é‡ç½®åŒæ­¥å‡ ä½•å˜æ¢çš„å‚æ•°ï¼ˆå¦‚æžœæœ‰çš„è¯ï¼‰
            from func_2d.augmentation import SynchronizedGeometricTransform
            SynchronizedGeometricTransform.reset()
            
            state = torch.get_rng_state()
            img = self.transform(img)
            gt = torch.as_tensor((self.transform_msk(gt) ).float(), dtype=torch.float32)
            if self.mode == "train":
                gt_weak = torch.as_tensor((self.transform_msk(gt_weak) ).float(), dtype=torch.float32)
            
            # multi_rater_cup = [torch.as_tensor((self.transform(single_rater) >=0.5).float(), dtype=torch.float32) for single_rater in multi_rater_cup]
            # multi_rater_cup = torch.stack(multi_rater_cup, dim=0)

            torch.set_rng_state(state)
            
            # æ¸…ç†åŒæ­¥å‡ ä½•å˜æ¢çš„å‚æ•°
            SynchronizedGeometricTransform.reset()

        # find init click and apply majority vote
        
        # if self.prompt == 'click':

        #     point_label_cup, pt_cup = random_click(np.array(gt), point_label = 1)
        #     pt_cup = pt_cup[1:]
            
            # selected_rater_mask_cup_ori = multi_rater_cup.mean(axis=0)
            # selected_rater_mask_cup_ori = (selected_rater_mask_cup_ori >= 0.5).float() 


            # selected_rater_mask_cup = F.interpolate(selected_rater_mask_cup_ori.unsqueeze(0), size=(self.mask_size, self.mask_size), mode='bilinear', align_corners=False).mean(dim=0) # torch.Size([1, mask_size, mask_size])
            # selected_rater_mask_cup = (selected_rater_mask_cup >= 0.5).float()


            # # Or use any specific rater as GT
            # point_label_cup, pt_cup = random_click(np.array(multi_rater_cup[0, :, :, :].squeeze(0)), point_label = 1)
            # selected_rater_mask_cup_ori = multi_rater_cup[0,:,:,:]
            # selected_rater_mask_cup_ori = (selected_rater_mask_cup_ori >= 0.5).float() 

            # selected_rater_mask_cup = F.interpolate(selected_rater_mask_cup_ori.unsqueeze(0), size=(self.mask_size, self.mask_size), mode='bilinear', align_corners=False).mean(dim=0) # torch.Size([1, mask_size, mask_size])
            # selected_rater_mask_cup = (selected_rater_mask_cup >= 0.5).float()


        # è®°å½•æ–‡ä»¶åä¸Žå®Œæ•´è·¯å¾„ï¼Œæ–¹ä¾¿è°ƒè¯•æ—¶ç²¾ç¡®å®šä½åˆ°æ ·æœ¬
        image_meta_dict = {
            'filename_or_obj': name,
            'filepath': img_path,
        }
        if self.mode == "train":
            result = {
                'image':img,
                # 'p_label': point_label_cup,
                # 'pt':pt_cup, 
                'mask': gt, 
                'mask_weak':gt_weak,
                'image_meta_dict':image_meta_dict,
                # 'raw_'
            }
            return result
        else:
            return {
                'image':img,
                # 'p_label': point_label_cup,
                # 'pt':pt_cup, 
                'mask': gt, 
                # 'mask_weak':gt_weak,
                'image_meta_dict':image_meta_dict,
                'raw_image':raw_img,
            }


class ISIC(Dataset):
    """ISIC 2017/2018 æ•°æ®é›†ï¼šä»Žæ–‡ä»¶å¤¹²È="24ôÁÉ½µÁÐ(€€€€€€€Í•±˜¹¥µ}Í¥é”€ô…ÉÌ¹¥µ…•}Í¥é”(€€€€€€€Í•±˜¹µ…Í­}Í¥é”€ô…ÉÌ¹½ÕÑ}Í¥é”(€€€€€€€Í•±˜¹ÑÉ…¹Í™½É´€ôÑÉ…¹Í™½É´(€€€€€€€Í•±˜¹ÑÉ…¹Í™½Éµ}µÍ¬€ôÑÉ…¹Í™½Éµ}µÍ¬((€€€€€€€¥µ…•Í}‘¥È€ô½Ì¹Á…Ñ ¹©½¥¸¡Í•±˜¹‘…Ñ…}Á…Ñ °€¥µ…”œ¤(€€€€€€€Ñ}‘¥È€ô½Ì¹Á…Ñ ¹©½¥¸¡Í•±˜¹‘…Ñ…}Á…Ñ °€Ðœ¤(€€€€€€€ÍÉ¥‰‰±•}‘¥È€ô½Ì¹Á…Ñ ¹©½¥¸¡Í•±˜¹‘…Ñ…}Á…Ñ °€ÍÉ¥‰‰±”œ¤((€€€€€€€¥˜¹½Ð½Ì¹Á…Ñ ¹¥Í‘¥È¡¥µ…•Í}‘¥È¤è(€€€€€€€€€€€É…¥Í”¥±•9½Ñ½Õ¹‘ÉÉ½È¡˜‰A…¹É•…ÍPƒ–nû–?žn»–öW’â7–¶c–r èí¥µ…•Í}‘¥Éôˆ¤(€€€€€€€¥˜¹½Ð½Ì¹Á…Ñ ¹¥Í‘¥È¡Ñ}‘¥È¤è(€€€€€€€€€€€É…¥Í”¥±•9½Ñ½Õ¹‘ÉÉ½È¡˜‰A…¹É•…ÍPPƒžn»–öW’â7–¶c–r èíÑ}‘¥Éôˆ¤((€€€€€€€Í•±˜¹¥µ…•Ì€ômt(€€€€€€€Í•±˜¹ÑÌ€ômt(€€€€€€€Í•±˜¹ÑÍ}Ý•…¬€ômt((€€€€€€€™½È˜¥¸Í½ÉÑ•¡½Ì¹±¥ÍÑ‘¥È¡¥µ…•Í}‘¥È¤¤è(€€€€€€€€€€€‰…Í”°•áÐ€ô½Ì¹Á…Ñ ¹ÍÁ±¥Ñ•áÐ¡˜¤(€€€€€€€€€€€¥˜•áÐ¹±½Ý•È ¤¹½Ð¥¸Í•±˜¹%5}aPè(€€€€€€€€€€€€€€€½¹Ñ¥¹Õ”(€€€€€€€€€€€¥µ}Á…Ñ €ô½Ì¹Á…Ñ ¹©½¥¸¡¥µ…•Í}‘¥È°˜¤((€€€€€€€€€€€Ñ}Á…Ñ €ôÍ•±˜¹}™¥¹‘}Á…¥È¡Ñ}‘¥È°‰…Í”¤(€€€€€€€€€€€¥˜Ñ}Á…Ñ ¥Ì9½¹”è(€€€€€€€€€€€€€€€½¹Ñ¥¹Õ”((€€€€€€€€€€€¥˜µ½‘”€ôô€ÑÉ…¥¸œè(€€€€€€€€€€€€€€€ÍÉ¥‰‰±•}Á…Ñ €ôÍ•±˜¹}™¥¹‘}Á…¥È¡ÍÉ¥‰‰±•}‘¥È°‰…Í”¤(€€€€€€€€€€€€€€€¥˜ÍÉ¥‰‰±•}Á…Ñ ¥Ì9½¹”è(€€€€€€€€€€€€€€€€€€€½¹Ñ¥¹Õ”(€€€€€€€€€€€€€€€Í•±˜¹ÑÍ}Ý•…¬¹…ÁÁ•¹¡ÍÉ¥‰‰±•}Á…Ñ ¤((€€€€€€€€€€€Í•±˜¹¥µ…•Ì¹…ÁÁ•¹¡¥µ}Á…Ñ ¤(€€€€€€€€€€€Í•±˜¹ÑÌ¹…ÁÁ•¹¡Ñ}Á…Ñ ¤((€€€‘•˜}™¥¹‘}Á…¥È¡Í•±˜°‘¥É}Á…Ñ °‰…Í•}¹…µ”¤è(€€€€€€€¥˜¹½Ð½Ì¹Á…Ñ ¹¥Í‘¥È¡‘¥É}Á…Ñ ¤è(€€€€€€€€€€€É•ÑÕÉ¸9½¹”(€€€€€€€™½È•áÐ¥¸Í•±˜¹Q}aPè(€€€€€€€€€€€Á…Ñ €ô½Ì¹Á…Ñ ¹©½¥¸¡‘¥É}Á…Ñ °‰…Í•}¹…µ”€¬•áÐ¤(€€€€€€€€€€€¥˜½Ì¹Á…Ñ ¹¥Í™¥±”¡Á…Ñ ¤è(€€€€€€€€€€€€€€€É•ÑÕÉ¸Á…Ñ (€€€€€€€É•ÑÕÉ¸9½¹”((€€€‘•˜}}±•¹}|¡Í•±˜¤è(€€€€€€€É•ÑÕÉ¸±•¸¡Í•±˜¹¥µ…•Ì¤((€€€‘•˜}}•Ñ¥Ñ•µ}|¡Í•±˜°¥¹‘•à¤è(€€€€€€€¥µ}Á…Ñ €ôÍ•±˜¹¥µ…•Ím¥¹‘•át(€€€€€€€¹…µ”€ô½Ì¹Á…Ñ ¹‰…Í•¹…µ”¡¥µ}Á…Ñ ¤(€€€€€€€Ñ}Á…Ñ €ôÍ•±˜¹ÑÍm¥¹‘•át((€€€€€€€¥µœ€ô%µ…”¹½Á•¸¡¥µ}Á…Ñ ¤¹½¹Ù•ÉÐ Iœ¤(€€€€€€€É…Ý}¥µœ€ô%µ…”¹½Á•¸¡¥µ}Á…Ñ ¤¹½¹Ù•ÉÐ Iœ¤(€€€€€€€É…Ý}¥µœ€ôÑ½É ¹…Í}Ñ•¹Í½È ¡Í•±˜¹ÑÉ…¹Í™½Éµ}µÍ¬¡É…Ý}¥µœ¤¤¹™±½…Ð ¤°‘ÑåÁ”õÑ½É ¹™±½…ÐÌÈ¤(€€€€€€€Ð€ô%µ…”¹½Á•¸¡Ñ}Á…Ñ ¤¹½¹Ù•ÉÐ 0œ¤((€€€€€€€¥˜Í•±˜¹µ½‘”€ôô€ÑÉ…¥¸œè(€€€€€€€€€€€Ñ}Ý•…­}Á…Ñ €ôÍ•±˜¹ÑÍ}Ý•…­m¥¹‘•át(€€€€€€€€€€€Ñ}Ý•…¬€ô%µ…”¹½Á•¸¡Ñ}Ý•…­}Á…Ñ ¤¹½¹Ù•ÉÐ 0œ¤((€€€€€€€¥˜Í•±˜¹ÑÉ…¹Í™½É´è(€€€€€€€€€€€™É½´™Õ¹|É¹…Õµ•¹Ñ…Ñ¥½¸¥µÁ½ÉÐMå¹¡É½¹¥é•‘•½µ•ÑÉ¥QÉ…¹Í™½É´(€€€€€€€€€€€Må¹¡É½¹¥é•‘•½µ•ÑÉ¥QÉ…¹Í™½É´¹É•Í•Ð ¤(€€€€€€€€€€€ÍÑ…Ñ”€ôÑ½É ¹•Ñ}É¹}ÍÑ…Ñ” ¤(€€€€€€€€€€€¥µœ€ôÍ•±˜¹ÑÉ…¹Í™½É´¡¥µœ¤(€€€€€€€€€€€Ð€ôÑ½É ¹…Í}Ñ•¹Í½È ¡Í•±˜¹ÑÉ…¹Í™½Éµ}µÍ¬¡Ð¤¤¹™±½…Ð ¤°‘ÑåÁ”õÑ½É ¹™±½…ÐÌÈ¤(€€€€€€€€€€€¥˜Í•±˜¹µ½‘”€ôô€ÑÉ…¥¸œè(€€€€€€€€€€€€€€€Ñ}Ý•…¬€ôÑ½É ¹…Í}Ñ•¹Í½È ¡Í•±˜¹ÑÉ…¹Í™½Éµ}µÍ¬¡Ñ}Ý•…¬¤¤¹™±½…Ð ¤°‘ÑåÁ”õÑ½É ¹™±½…ÐÌÈ¤(€€€€€€€€€€€Ñ½É ¹Í•Ñ}É¹}ÍÑ…Ñ”¡ÍÑ…Ñ”¤(€€€€€€€€€€€Må¹¡É½¹¥é•‘•½µ•ÑÉ¥QÉ…¹Í™½É´¹É•Í•Ð ¤((€€€€€€€¥µ…•}µ•Ñ…}‘¥Ð€ôì(€€€€€€€€€€€€™¥±•¹…µ•}½É}½‰¨œè¹…µ”°(€€€€€€€€€€€€™¥±•Á…Ñ œè¥µ}Á…Ñ °(€€€€€€€ô(€€€€€€€¥˜Í•±˜¹µ½‘”€ôô€ÑÉ…¥¸œè(€€€€€€€€€€€É•ÍÕ±Ð€ôì(€€€€€€€€€€€€€€€€¥µ…”œè¥µœ°(€€€€€€€€€€€€€€€€µ…Í¬œèÐ°(€€€€€€€€€€€€€€€€µ…Í­}Ý•…¬œèÑ}Ý•…¬°(€€€€€€€€€€€€€€€€¥µ…•}µ•Ñ…}‘¥Ðœè¥µ…•}µ•Ñ…}‘¥Ð°(€€€€€€€€€€€ô(€€€€€€€€€€€É•ÑÕÉ¸É•ÍÕ±Ð(€€€€€€€•±Í”è(€€€€€€€€€€€É•ÑÕÉ¸ì(€€€€€€€€€€€€€€€€¥µ…”œè¥µœ°(€€€€€€€€€€€€€€€€µ…Í¬œèÐ°(€€€€€€€€€€€€€€€€¥µ…•}µ•Ñ…}‘¥Ðœè¥µ…•}µ•Ñ…}‘¥Ð°(€€€€€€€€€€€€€€€€É…Ý}¥µ…”œèÉ…Ý}¥µœ°(€€€€€€€€€€€ô(