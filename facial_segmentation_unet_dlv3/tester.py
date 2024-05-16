
import os
import time
import torch
import datetime
import numpy as np
import pickle
from scipy.ndimage import label

import torch.nn as nn
from torch.autograd import Variable
from torchvision.utils import save_image
from torchvision import transforms
import matplotlib.pyplot as plt
import cv2
import PIL

import re
from unet import unet
from utils import *
from PIL import Image, ImageOps
from plotting import *

import pandas as pd
import csv

from maskExtraction import EyeFeatureExtractor
from measureAnatomy import EyeMetrics
from distance_plot import Plotter
from sam_model import get_bounding_boxes, SAM

# SAM_CHECKPOINT_PATH = os.path.join('..','..', 'SAM_WEIGHTS', 'sam_vit_h_4b8939.pth')
# SAM_ENCODER_VERSION = "vit_h"



def write_dice_scores_to_csv(storage, csv_file_path):
    with open(csv_file_path, mode='w', newline='') as file:
        writer = csv.writer(file)
        
        # Write the header row, assuming all dictionaries have the same keys
        if storage:
            headers = ['Batch Index'] + [f'Dice Score Class {cls}' for cls in storage[0].keys()]
            writer.writerow(headers)
            
            # Write each set of Dice scores to the CSV file
            for i, dice_scores in enumerate(storage):
                row = [i] + list(dice_scores.values())
                writer.writerow(row)


def apply_colormap(image, color_map):
    print("Unique pixel values in image:", np.unique(image))

    """Apply a colormap to a single-channel image."""
    colored_image = np.zeros((image.shape[0], image.shape[1], 3), dtype=np.uint8)
    for cls, color in color_map.items():
        colored_image[image == cls] = color
    return colored_image


# 2. 'l_eye', 'r_eye', 'l_brow', 'r_brow', = 1,2,3,4\
def relabel_classes(mask):
    mask[(mask == 3) | (mask == 2)] = 1
    mask[(mask == 5) | (mask == 6)] = 1

    mask[(mask == 3) | (mask == 4)] = 2    
    return mask
    
def visualize_and_dice(predictions, targets, batch_size, storage, corrected_masks, custom=False,dataset=None):
    """Visualize and save predicted and ground truth segmentation maps."""


    for idx in range(len(predictions)):
        pred_image = predictions[idx]
        if dataset != 'ted_long':
            gt_image = targets[idx]

            gt_image = np.squeeze(gt_image)
            gt_image = gt_image*255
            

            # New class mappings:
            # both eye and iris class should just be eye
            new_label_np = np.zeros_like(gt_image)
            new_label_np[(gt_image == 1) | (gt_image == 2)] = 1
            new_label_np[(gt_image == 3) | (gt_image == 4)] = 2
            new_label_np[(gt_image == 5) | (gt_image == 6)] = 1

            dice = dice_coefficient(pred_image, new_label_np, custom)

            # corrected_mask = mask_corrector(pred_image)
            storage.append(dice)
        corrected_masks.append(pred_image)
        
    return storage

def dice_coefficient(pred, target, custom=True, num_classes=3):
    """Compute the Dice score, combining left and right brow and sclera into single labels."""
    
    temp_pred = np.copy(pred)
    temp_target = np.copy(target) 
    

    dice_scores = {}
    for cls in range(num_classes):  # Now, only iterating over the combined classes
        pred_cls = (temp_pred == cls)
        target_cls = (temp_target == cls)
        intersection = np.logical_and(pred_cls, target_cls).sum()
        union = pred_cls.sum() + target_cls.sum()
        
        dice_score = 2 * intersection / (union + 1e-6)
        dice_scores[cls] = dice_score
        
    return dice_scores

class ResizeAndPad:
    def __init__(self, output_size=(512, 512), fill=0, padding_mode='constant'):
        self.output_size = output_size
        self.fill = fill
        self.padding_mode = padding_mode

    def __call__(self, img):
        # Calculate new height maintaining aspect ratio
        original_width, original_height = img.size
        new_height = int(original_height * (self.output_size[0] / original_width))
        img = img.resize((self.output_size[0], new_height), Image.NEAREST)

        # Calculate padding
        padding_top = (self.output_size[1] - new_height) // 2
        padding_bottom = self.output_size[1] - new_height - padding_top

        # Apply padding
        img = ImageOps.expand(img, (0, padding_top, 0, padding_bottom), fill=self.fill)
        
        return img
    

def transformer(dynamic_resize_and_pad, totensor, normalize, centercrop, imsize, is_mask=False):
    options = []
    if centercrop:
        options.append(transforms.CenterCrop(160))
    if dynamic_resize_and_pad:
        if is_mask:
            options.append(ResizeAndPad(output_size=(imsize, imsize), interpolation=Image.NEAREST))
        else:
            options.append(ResizeAndPad(output_size=(imsize, imsize)))
    if totensor:
        options.append(transforms.ToTensor())
    if normalize:
        options.append(transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)))
    return transforms.Compose(options)

def crop_and_resize(img):
    # Crop the image into left and right halves
    mid = img.width // 2
    left_half = img.crop((0, 0, mid, img.height))
    # Adjust the start of the right half if the width is not divisible by 2
    right_half_start = mid if img.width % 2 == 0 else mid + 1
    right_half = img.crop((right_half_start, 0, img.width, img.height))
    # Resize each half to 256x256
    left_resized = left_half.resize((256, 256))
    right_resized = right_half.resize((256, 256))
    return left_resized, right_resized


def transform_img_split(resize, totensor, normalize):
    options = []

    if resize:
        options.append(transforms.Lambda(crop_and_resize))

    if totensor:
        # Adjust to handle a pair of images (left and right halves)
        options.append(transforms.Lambda(lambda imgs: (transforms.ToTensor()(imgs[0]), transforms.ToTensor()(imgs[1]))))
        
    if normalize:
        # Normalize each image in the pair
        options.append(transforms.Lambda(lambda imgs: (transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))(imgs[0]), 
                                                                            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))(imgs[1]))))
        
    transform = transforms.Compose(options)
    return transform



def make_dataset(dir, gt=False, custom=True):
    images = []
    assert os.path.isdir(dir), '%s is not a valid directory' % dir

    f = dir.split('/')[-1].split('_')[-1]
    print (dir, len([name for name in os.listdir(dir) if os.path.isfile(os.path.join(dir, name))]))
    if custom:
        for root, dirs, files in os.walk(dir):
            for file in files:
                if 'checkpoint' not in file:
                    path = os.path.join(dir, file)
                    images.append(path)

    else:
        for i in range(len([name for name in os.listdir(dir) if os.path.isfile(os.path.join(dir, name))])):
            if not gt:
                i = i
                img = str(i) + '.jpg'
            else:
                img = str(i) + '.png'
            path = os.path.join(dir, img)
            images.append(path)
       
    return images

def extract_numeric_id(file_path, dataset):
    if dataset == 'ted':
    # Extracts numeric part from the filename, assuming it follows 't_' and precedes '_crop'
        match = re.search(r't_(\d+)_crop', file_path)
        return int(match.group(1)) if match else None
    elif dataset == 'md':
        match = re.search(r'md_(\d+)_crop', file_path)
        return int(match.group(1)) if match else None 
    elif dataset == 'cfd':
        # The ID may include uppercase/lowercase letters, digits, and hyphens
        match = re.search(r'CFD-([A-Za-z0-9-]+)-N?_crop', file_path)
        return match.group(1) if match else None 


def align_images_and_labels(test_image_paths, gt_test_paths,dataset):
    # Create dictionaries with the numeric ID as the key
    test_images_dict = {extract_numeric_id(path, dataset): path for path in test_image_paths}
    gt_images_dict = {extract_numeric_id(path, dataset): path for path in gt_test_paths}
    
    aligned_test_images = []
    aligned_gt_images = []
    
    # Align based on numeric IDs
    for numeric_id in test_images_dict.keys():
        if numeric_id in gt_images_dict:
            aligned_test_images.append(test_images_dict[numeric_id])
            aligned_gt_images.append(gt_images_dict[numeric_id])
    
    return aligned_test_images, aligned_gt_images

class Tester(object):
    def __init__(self, config, device):
        # exact model and loss
        self.model = config.model

        # Model hyper-parameters
        self.imsize = config.imsize
        self.parallel = config.parallel

        self.total_step = config.total_step
        self.batch_size = config.batch_size
        self.num_workers = config.num_workers
        self.g_lr = config.g_lr
        self.lr_decay = config.lr_decay
        self.beta1 = config.beta1
        self.beta2 = config.beta2
        self.pretrained_model = config.pretrained_model

        self.img_path = config.img_path
        self.label_path = config.label_path 
        self.log_path = config.log_path
        self.model_save_path = config.model_save_path
        self.sample_path = config.sample_path
        self.log_step = config.log_step
        self.sample_step = config.sample_step
        self.model_save_step = config.model_save_step
        self.version = config.version
        self.dataset = config.dataset
        self.device = device
        # Path
        self.log_path = os.path.join(config.log_path, self.version)
        self.sample_path = os.path.join(config.sample_path, self.version)
        self.model_save_path = os.path.join(config.model_save_path, self.version)
        
        if self.dataset != 'ted_long':
            self.test_label_path = config.test_label_path
            self.test_label_path_gt = config.test_label_path_gt
            self.test_color_label_path = config.test_color_label_path
            
        self.test_image_path = config.test_image_path

        self.get_sam_iris = config.get_sam_iris

        # Test size and model
        self.test_size = config.test_size
        self.model_name = config.model_name
        
        self.train_limit = config.train_limit
        
        self.csv_path = config.csv_path
        
        self.pickle_in = config.pickle_in
        self.split_face = config.split_face

        self.build_model()

    def test(self):
        sam = SAM()
        plotter = Plotter()
        

        if not self.pickle_in:
            if self.split_face:
                transform = transform_img_split(resize=True, totensor=True, normalize=True)
                transform_plotting = transform_img_split(resize=True, totensor=False, normalize=False)
                transform_plotting_sam = transformer(dynamic_resize_and_pad=True, totensor=False, normalize=False,centercrop=False, imsize=512)
                transform_gt = transformer(dynamic_resize_and_pad=True, totensor=True, normalize=False,centercrop=False, imsize=512)

            else:
                transform = transformer(dynamic_resize_and_pad=True, totensor=True, normalize=True, centercrop=False, imsize=512)
                transform_plotting = transformer(dynamic_resize_and_pad=True, totensor=False, normalize=False, centercrop=False, imsize=512)
                
            if self.dataset != 'celeb':
                custom = True
            else:
                custom = False
                
            # if custom, return list of paths, otherwise return list of paths for celeb that are numbers.jpg/png
            test_paths = make_dataset(self.test_image_path, custom=custom)
            if self.dataset != 'ted_long':
                gt_test_paths = make_dataset(self.test_label_path_gt, gt=True, custom=custom)
                print(f'length of test path is {len(test_paths)} and gt is {len(gt_test_paths)}')
            
            #align the paths so the indices match if custom dataset
            if custom and self.dataset != 'ted_long':
                test_paths, gt_test_paths = align_images_and_labels(test_paths, gt_test_paths, self.dataset)
                
            # make_folder(self.test_label_path, '')
            # make_folder(self.test_color_label_path, '') 
            
            # load model
            self.G.load_state_dict(torch.load(os.path.join(self.model_save_path, self.model_name)))
            self.G.eval() 
            
            batch_num = int(self.test_size / self.batch_size)
            storage = []
            corrected_masks = []
            iris_masks = []
            images_plotting = []
            names = []
            gt_for_storage = []

            for i in range(batch_num):
                imgs = []
                gt_labels = []
                l_imgs = []
                r_imgs = []
                original_sizes = []
                for j in range(self.batch_size):
                    current_idx = i * self.batch_size + j
                    if current_idx < len(test_paths):
                        path = test_paths[current_idx]
                        name = path.split('/')[-1][:-4]
                        names.append(name)
                        print(name)
                        if self.split_face:
                            original_sizes.append(Image.open(path).size)
                            l_img, r_img = transform(Image.open(path))
                            l_imgs.append(l_img)
                            r_imgs.append(r_img)
                            images_plotting.append(transform_plotting_sam(Image.open(path)))

                        else:
                            img = transform(Image.open(path))
                            imgs.append(img)
                            images_plotting.append(transform_plotting(Image.open(path)))
              
                        if self.get_sam_iris:
                            # 1. find bounding box for l and r iris from csv file based on name
                            l_iris, r_iris = get_bounding_boxes(self.csv_path, path.split('/')[-1], self.dataset)
                            # 2. submit bounding box to SAM
                            # 3. return mask of l and r iris
                            img = cv2.imread(path)    
                            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                    
                            boxes = np.array([r_iris,l_iris])
                            masks_dict = sam.segment_no_jitter(img, boxes)

                            # 4. make a temporary numpy array of 512 x 512
                            temp_iris_mask = np.zeros((512, 512), dtype=np.uint8)

                            # 5. resize the returned binary mask using the same logic for transform
                            # mask_transform = transformer(dynamic_resize_and_pad=True, totensor=False, normalize=False, centercrop=False, imsize=512)
                            l_iris_transform = transform_plotting_sam(Image.fromarray(masks_dict['left_iris']))
                            r_iris_transform = transform_plotting_sam(Image.fromarray(masks_dict['right_iris']))
                        
                       
                            # 6. Paste the iris mask into the temp array w labels 5 and 6 for l and r
                            # 7. Store in iria masks list
                            # Convert transformed PIL Images back to numpy arrays for further processing
                            l_iris_mask_np = np.array(l_iris_transform)
                            r_iris_mask_np = np.array(r_iris_transform)

                            temp_iris_mask[l_iris_mask_np > 0] = 3  
                            temp_iris_mask[r_iris_mask_np > 0] = 3
                
                            
                            iris_masks.append(temp_iris_mask)
    
                        if self.dataset != 'ted_long':
                            gt_path = gt_test_paths[current_idx]
                            print(path, gt_path)

                            gt_img = Image.open(gt_path)

                            
                            gt_label = transform_plotting_sam(gt_img)
                            
                            gt_labels.append(transform_gt(gt_img).numpy())
                            gt_for_storage.append(np.array(gt_label))

                    else:
                        break 
                    
                    
                if len(imgs) != 0 or len(l_imgs)!=0:
                    print('PREDICTING IMAGES NOW ')
                    if self.split_face:
                        l_imgs = torch.stack(l_imgs) 
                        r_imgs = torch.stack(r_imgs) 
                        print(self.device)
                        l_imgs = l_imgs.to(self.device)
                        r_imgs = r_imgs.to(self.device)
                        
                        l_labels_predict = self.G(l_imgs)
                        r_labels_predict = self.G(r_imgs)
                        
                        l_labels_predict_plain = generate_label_plain(l_labels_predict, self.imsize)
                        r_labels_predict_plain = generate_label_plain(r_labels_predict, self.imsize)
                        
                        labels_predict_plain = []

                        for idx, (left_pred, right_pred) in enumerate(zip(l_labels_predict_plain, r_labels_predict_plain)):
                            original_width, original_height = original_sizes[idx]
                            mid = original_width // 2
                            
                            # Calculate dimensions for left and right halves based on the original sizes
                            left_width = mid  # Since mid is the midpoint
                            right_width = original_width - mid  # Width from midpoint to right edge

                            # Resize predictions to match these dimensions
                            left_pred_resized = cv2.resize(left_pred, (left_width, original_height), interpolation=cv2.INTER_NEAREST)
                            right_pred_resized = cv2.resize(right_pred, (right_width, original_height), interpolation=cv2.INTER_NEAREST)

                            # Create a new empty array for the stitched prediction
                            stitched = np.zeros((original_height, original_width), dtype=np.uint8)

                            # Place the resized predictions onto the stitched canvas
                            stitched[:, :mid] = left_pred_resized
                            stitched[:, mid:] = right_pred_resized
                        
                            
                            # Resize stitched prediction to 512x512
                            resized_stitched = transform_plotting_sam(Image.fromarray(stitched))
                            
                            
                            labels_predict_plain.append(np.array(resized_stitched))
        
                        print('converting labels')
                        labels_predict_plain = np.array(labels_predict_plain)
                        print(len(labels_predict_plain))
                    else:
                        imgs = torch.stack(imgs) 
                        imgs = imgs.to(self.device)
                
                        # predict
                        labels_predict = self.G(imgs)
            
                        # # # After obtaining predictions
                        labels_predict_plain = generate_label_plain(labels_predict, self.imsize)
                    # if self.dataset != 'ted_long':
                    visualize_and_dice(labels_predict_plain, np.array(gt_labels), self.batch_size, storage, corrected_masks, custom=custom, dataset = self.dataset)

    
            if self.dataset != 'ted_long':
                if self.train_limit is None:
                    limit = 'all'
                else:
                    limit = self.train_limit

                title = f'{self.dataset}_{limit}_test'
                csv_file_path = f'{title}_dice_scores.csv'
                write_dice_scores_to_csv(storage, csv_file_path)
                plot_dice_boxplots(storage, title)
            # plot_dice_histograms(storage, title)
            
            if self.get_sam_iris:
                # 1. replace the pixels in corrected masks with the iris masks
                # 2. Make a transposed dict of all of the masks so that we can reuse the old shitty measurement path
                # 3. Once everything is in right format, send down measurements without VD / canthal tilt as they need the midline points
                # 4. Once verified that these are correct, send down measurement pathway 
                integrated_masks = []

                for corrected_mask, iris_mask in zip(corrected_masks, iris_masks):
                    if self.split_face:
                        corrected_mask[iris_mask == 3] = 3 
                    else:
                        # Replace the left iris region in the corrected mask
                        corrected_mask[iris_mask == 5] = 5
            
                        # Replace the right iris region in the corrected mask
                        corrected_mask[iris_mask == 6] = 6
                    integrated_masks.append(corrected_mask)
                
                features_list = [extract_features_from_mask(mask,idx, split_face=self.split_face) for idx, mask in enumerate(integrated_masks)]
                
                if self.dataset != 'ted_long':
                    gt_features_list = [extract_features_from_mask(mask,idx, gt=True, split_face=self.split_face) for idx,mask in enumerate(gt_for_storage)]
                
                    # Save all lists in a dictionary
                    data_to_save = {
                        'features_list': features_list,
                        'images_plotting': images_plotting,
                        'gt_features_list': gt_features_list,
                        'names': names
                    }
                else:
                    data_to_save = {
                        'features_list': features_list,
                        'images_plotting': images_plotting,
                        'names': names
                    }
                    
                # Save the dictionary
                with open('md_all_data.pkl', 'wb') as file:
                    pickle.dump(data_to_save, file)
                    
        if self.pickle_in:
            # Load the dictionary
            with open('md_all_data.pkl', 'rb') as file:
                loaded_data = pickle.load(file)

            # Unpack the lists
            features_list = loaded_data['features_list']
            images_plotting = loaded_data['images_plotting']
            names = loaded_data['names']
            if self.dataset != 'ted_long':
                gt_features_list = loaded_data['gt_features_list']

            
        pred_measurements = []
        pred_landmarks = []
        gt_measurements = []
        gt_landmarks = []

        bad_indices_pred = []
        bad_indices_gt = []
        print(len(features_list))
        # measurments for predictions
        print('ANALYZING AI PREDICTIONS NOW')

        for idx, features in enumerate(features_list):
            try:
                _, features_array = features  
                
                extractor = EyeFeatureExtractor(features_array, images_plotting[idx],idx)
                landmarks = extractor.extract_features()
                pred_landmarks.append(landmarks)

                # Create an instance of EyeMetrics with the landmarks
                eye_metrics = EyeMetrics(landmarks, features_array) 
                measurements = eye_metrics.run()
                pred_measurements.append(measurements)
                
                # store marked up images to see if this is working
                plotter.create_plots(images_plotting[idx], features_array, landmarks, names[idx], measurements)
            except (ValueError, KeyError):
                bad_indices_pred.append(idx)
        if self.train_limit == None:
            save_to_csv(self.dataset, names, pred_measurements, pred_landmarks)
                

        #measurements for gt 
        if self.dataset != 'ted_long':
            print('ANALYZING GT NOW')
            for idx, features in enumerate(gt_features_list):
                try:
                    _, features_array_gt = features           

                    extractor_gt = EyeFeatureExtractor(features_array_gt, images_plotting[idx],idx, gt=True)
                    landmarks_gt = extractor_gt.extract_features()
                    gt_landmarks.append(landmarks_gt)

                    # Create an instance of EyeMetrics with the landmarks
                    eye_metrics_gt = EyeMetrics(landmarks_gt, features_array_gt) 
                    measurements_gt = eye_metrics_gt.run()
                    gt_measurements.append(measurements_gt)
                    plotter.create_plots(images_plotting[idx], features_array_gt, landmarks_gt, names[idx], measurements_gt, gt=True)

                except (ValueError, KeyError):
                    bad_indices_gt.append(idx)
            if self.train_limit == None:
                save_to_csv(self.dataset, names, gt_measurements, gt_landmarks, gt=True)

                    
            print(f'PRINTING BAD INDICES PRED:{bad_indices_pred} and GT: {bad_indices_gt}. REMOVING PRED FROM GT ONLY')
            all_bad_indices = set(bad_indices_pred) | set(bad_indices_gt)

            # Remove bad indices from the names list
            names = [name for i, name in enumerate(names) if i not in all_bad_indices]

            # Remove bad indices from gt_measurements and gt_landmarks if they are in bad_indices_pred
            gt_measurements = [m for i, m in enumerate(gt_measurements) if i not in bad_indices_pred]
            gt_landmarks = [l for i, l in enumerate(gt_landmarks) if i not in bad_indices_pred]

            # Remove bad indices from pred_measurements and pred_landmarks if they are in bad_indices_gt
            pred_measurements = [m for i, m in enumerate(pred_measurements) if i not in bad_indices_gt]
            pred_landmarks = [l for i, l in enumerate(pred_landmarks) if i not in bad_indices_gt]
            
            title = f'{self.dataset}_{self.train_limit}_test'
            mae_df = calculate_mae_for_all_images(names, gt_measurements, gt_landmarks, pred_measurements, pred_landmarks)        
            mae_df.to_csv(f'{title}_mae.csv')



    def build_model(self):
        # device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Using device: {self.device}")
        self.G = unet().to(self.device)
        if self.parallel:
            self.G = nn.DataParallel(self.G)


def extract_features_from_mask(mask, idx, gt=False, split_face=False):
    if gt:
        features_array = {
            'right_iris': (mask == 6), 
            'left_iris': (mask == 5),   
            'right_sclera': np.logical_or(mask == 2, mask == 6),
            'left_sclera': np.logical_or(mask == 1, mask == 5),
            'right_brow': (mask == 4),
            'left_brow': (mask == 3)     
        }

    if split_face and not gt:
        # The new output mask initialized to zeros, same shape as input mask
        new_mask = np.zeros_like(mask)
        
        # Midpoint of the image to split left and right
        mid = mask.shape[1] // 2
        unique_3 = np.unique(new_mask)

        # Update left side
        new_mask[:, :mid][mask[:, :mid] == 1] = 2  # Left Sclera
        new_mask[:, :mid][mask[:, :mid] == 2] = 4  # Left Brow
        new_mask[:, :mid][mask[:, :mid] == 3] = 6  # Left Iris

        # Update right side
        new_mask[:, mid:][mask[:, mid:] == 1] = 1  # Right Sclera
        new_mask[:, mid:][mask[:, mid:] == 2] = 3  # Right Brow
        new_mask[:, mid:][mask[:, mid:] == 3] = 5  # Right Iris

        features_array = {
            'right_iris': (new_mask == 6), 
            'left_iris': (new_mask == 5),   
            'right_sclera': np.logical_or(new_mask == 2, new_mask == 6),
            'left_sclera': np.logical_or(new_mask == 1, new_mask == 5),
            'right_brow': (new_mask == 4),
            'left_brow': (new_mask == 3)     
        }

    features_transposed = {key: np.transpose(np.nonzero(value)) for key, value in features_array.items()}

    return features_transposed, features_array



def calculate_mae_for_all_images(names, gt_measurements_list, gt_landmarks_list, pred_measurements_list, pred_landmarks_list):
    # Initialize a list to store MAE results for each image
    image_mae_results = []

    # Iterate over all images
    for name, gt_measurements, gt_landmarks, measurements, landmarks in zip(names, gt_measurements_list, gt_landmarks_list, pred_measurements_list, pred_landmarks_list):
        # Initialize a dictionary for this image's MAE results
        image_mae = {'image_name': name}

        # Scaling factors based on iris diameters
        gt_cf = 11.71 / ((gt_landmarks['right_iris_diameter'] + gt_landmarks['left_iris_diameter']) / 2)
        pred_cf = 11.71 / ((landmarks['right_iris_diameter'] + landmarks['left_iris_diameter']) / 2)

        excluded_keys = ['left_vd_plot_point', 'right_vd_plot_point']
        special_keys = ['right_canthal_tilt', 'left_canthal_tilt', 'right_scleral_area', 'left_scleral_area']

        # Calculate MAE for each measurement and landmark, and store it in the dictionary
        for key in measurements.keys():
            if key not in excluded_keys:
                gt_val = gt_measurements.get(key, 0)  # Default to 0 if key not found in gt
                pred_val = measurements.get(key, 0)  # Default to 0 if key not found in predictions

                # Apply scaling if not a special key
                if key not in special_keys:
                    gt_val *= gt_cf
                    pred_val *= pred_cf
                
                # Calculate the MAE for this key
                image_mae[key] = abs(gt_val - pred_val)

        # Calculate MAE for the selected landmarks
        for landmark in ['right_iris_diameter', 'left_iris_diameter']:
            gt_land = gt_landmarks.get(landmark, 0)
            pred_land = landmarks.get(landmark, 0)
            image_mae[landmark] = abs(gt_land - pred_land)

        # Append this image's MAE results to the list
        image_mae_results.append(image_mae)

    # Create a DataFrame from the list of MAE results
    df_mae = pd.DataFrame(image_mae_results)
    print(df_mae.head())
    df_mae.set_index('image_name', inplace=True)

    return df_mae


def save_to_csv(dataset, names, measurements, landmarks, gt=False):
    # Save measurements to CSV
    if gt:
        with open(f'{dataset}_measurements_GROUND_TRUTH.csv', 'w', newline='') as file:
            writer = csv.writer(file)
            
            # Write header row
            header = ['Name'] + list(measurements[0].keys()) + ['right_iris_diameter', 'left_iris_diameter']
            writer.writerow(header)
            
            # Write data rows
            for name, measurement, landmark in zip(names, measurements, landmarks):
                row = [name] + list(measurement.values()) + [landmark['right_iris_diameter'], landmark['left_iris_diameter']]
                writer.writerow(row)
    else:
        with open(f'{dataset}_measurements.csv', 'w', newline='') as file:
            writer = csv.writer(file)
            
            # Write header row
            header = ['Name'] + list(measurements[0].keys()) + ['right_iris_diameter', 'left_iris_diameter']
            writer.writerow(header)
            
            # Write data rows
            for name, measurement, landmark in zip(names, measurements, landmarks):
                row = [name] + list(measurement.values()) + [landmark['right_iris_diameter'], landmark['left_iris_diameter']]
                writer.writerow(row)
        
 