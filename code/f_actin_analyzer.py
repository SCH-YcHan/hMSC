#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import re
import os
import cv2
import math
import copy
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from PIL import Image
from pathlib import Path
from scipy import ndimage
from skimage import io, measure
from skimage.color import rgb2gray
from skimage.filters import threshold_otsu
from skimage.measure import regionprops, label
from skimage.exposure import equalize_adapthist
from skimage.morphology import remove_small_objects, remove_small_holes


# In[ ]:

class StemCellAnalyzer:
    def __init__(self, data_path, save_path):
        self.data_path = data_path
        self.save_path = save_path

    def intden_analysis(self, type_folder, output_file, folders):
        for folder in folders:
            path = os.path.join(self.data_path, folder, type_folder)
            save_folder_path = os.path.join(self.save_path, folder, type_folder)
            os.makedirs(save_folder_path, exist_ok=True)
            
            results_list = []
            file_list = [f for f in os.listdir(path) if "_405" in f]
            for img_path in file_list:
                img_num = re.findall(r'\d+', img_path)[0]
                image_path = os.path.join(path, img_path)
                image = cv2.imread(image_path)[:,:,0]
                label_image = self._find_region(image)
                regions = regionprops(label_image)
                if len(regions) == 1:
                    intden_data = self._process_intden(image_path, img_num, folder, type_folder, save_folder_path)
                    if intden_data:
                        results_list.append(intden_data)
            
            # Save CSV file for each folder
            df = pd.DataFrame(results_list)
            df.to_csv(os.path.join(save_folder_path, output_file), index=False)

    def adhesion_analysis(self, type_folder, folders):
        for folder in folders:
            path = os.path.join(self.data_path, folder, type_folder)
            save_folder_path = os.path.join(self.save_path, folder, type_folder)
            os.makedirs(save_folder_path, exist_ok=True)
            
            file_list = [f for f in os.listdir(path) if "_405" in f]
            for img_path in file_list:
                img_num = re.findall(r'\d+', img_path)[0]
                image_path = os.path.join(path, img_path)
                image = cv2.imread(image_path)[:,:,0]
                label_image = self._find_region(image)
                regions = regionprops(label_image)
                if len(regions) == 1:
                    adhesion_data = self._process_adhesion(image_path, img_num, folder, type_folder, save_folder_path)
                    df = pd.DataFrame(adhesion_data)
                    df.to_csv(os.path.join(save_folder_path, f"{img_num}_{type_folder}_object_info.csv"), index=False)

    def yap_analysis(self, type_folder, output_file, folders):
        for folder in folders:
            path = os.path.join(self.data_path, folder, type_folder)
            save_folder_path = os.path.join(self.save_path, folder, type_folder)
            os.makedirs(save_folder_path, exist_ok=True)
            
            results_list = []
            file_list = [f for f in os.listdir(path) if "_405" in f]
            for img_path in file_list:
                img_num = re.findall(r'\d+', img_path)[0]
                image_path = os.path.join(path, img_path)
                image = cv2.imread(image_path)[:,:,0]
                label_image = self._find_region(image)
                regions = regionprops(label_image)
                if len(regions) == 1:
                    yap_data = self._process_yap(image_path, img_num, folder, type_folder, save_folder_path)
                    if yap_data:
                        results_list.append(yap_data)
            
            # Save CSV file for each folder
            df = pd.DataFrame(results_list)
            df.to_csv(os.path.join(save_folder_path, output_file), index=False)

    def _process_intden(self, image_path, img_num, scope, cell_type, save_folder_path):
        images = self._load_images(image_path)
        
        if images:
            image_m, image_488, image_546, image_647 = images

            label_image = self._find_region(image_m)
            regions = regionprops(label_image)
            largest_region = max(regions, key=lambda region: region.area)
            coords = largest_region.coords[:, [1, 0]]
            rect = cv2.minAreaRect(coords)
            box = np.intp(cv2.boxPoints(rect))
            angle = self._calculate_angle(box)
            
            if Path(image_path).parts[1] in ['3', '4', '5', '6']:
                if np.isnan(image_647).any():
                    return None
                else:
                    rotated_488, rotated_647, aspect_ratio, intensities, rotated_box = self._analyze_intensity(
                        box, angle, image_488, image_647
                    )
                    self._save_intensity_plot(img_num, intensities, cell_type, save_folder_path)
                    self._save_region_plot(img_num, box, rotated_box, image_488, image_647, rotated_488, rotated_647, cell_type, save_folder_path)

                    return {
                        'scope': scope,
                        'type': cell_type,
                        'path': image_path,
                        'area': largest_region.area,
                        'aspect_ratio': aspect_ratio,
                        'width': rect[1][0],
                        'height': rect[1][1],
                        'intden': np.sum(intensities['red_intensity_sum'])
                    }
            else:
                if np.isnan(image_546).any():
                    return None
                else:
                    rotated_488, rotated_546, aspect_ratio, intensities, rotated_box = self._analyze_intensity(
                        box, angle, image_488, image_546
                    )
                    self._save_intensity_plot(img_num, intensities, cell_type, save_folder_path)
                    self._save_region_plot(img_num, box, rotated_box, image_488, image_546, rotated_488, rotated_546, cell_type, save_folder_path)

                    return {
                        'scope': scope,
                        'type': cell_type,
                        'path': image_path,
                        'area': largest_region.area,
                        'aspect_ratio': aspect_ratio,
                        'width': rect[1][0],
                        'height': rect[1][1],
                        'intden': np.sum(intensities['red_intensity_sum'])
                    }
        
        return None

    def _process_adhesion(self, image_path, img_num, scope, cell_type, save_folder_path):
        images = self._load_images(image_path)

        if images:
            image_m, image_488, image_546, image_647 = images
            image_405 = Image.open(image_path)

            blue_channel = np.array(image_405)[:, :, 2]
            green_channel = np.array(image_488)[:, :, 1]

            if np.isnan(image_546).any():
                if np.isnan(image_647).any():
                    return None # error 발생시키는게 좋을듯
                else:
                    image_647 = cv2.cvtColor(image_647, cv2.COLOR_BGR2RGB)
                    red_channel = np.array(image_647)[:, :, 0]
            else:
                image_546 = cv2.cvtColor(image_546, cv2.COLOR_BGR2RGB)
                red_channel = np.array(image_546)[:, :, 0]

            image_m = np.stack((red_channel, green_channel, blue_channel), axis=-1)
            gray_image = rgb2gray(image_m)
            enhanced_image = equalize_adapthist(gray_image)

            label_image = self._find_region(enhanced_image)
            regions = regionprops(label_image)
            largest_region = max(regions, key=lambda region: region.area)
            
            largest_region_mask = label_image == largest_region.label
            masked_image = np.array(image_m) * np.expand_dims(largest_region_mask, axis=-1)
            
            image = np.array(masked_image)
            array_img = np.array(image)

            area_sizes, condition, labeled_array = self._analyze_adhesion(array_img)
            self._save_adhesion_plot(
                img_num,
                array_img,
                condition,
                labeled_array,
                image, area_sizes,
                cell_type,
                save_folder_path
            )

            object_data = self._object_info(area_sizes, labeled_array)
            
            return object_data
        
        return None

    def _process_yap(self, image_path, img_num, scope, cell_type, save_folder_path):
        images = self._load_images(image_path)

        if images:
            image_m, image_488, image_546, image_647 = images
            image_405 = cv2.imread(image_path)

            try:
                gray_image_405 = rgb2gray(image_405)
                label_image_405 = self._find_region(gray_image_405, enhanced=True)
                regions_405 = regionprops(label_image_405)
                largest_region_405 = max(regions_405, key=lambda region: region.area)
                coords_405 = largest_region_405.coords
                coords_405 = coords_405[:, [1, 0]]
                rect_405 = cv2.minAreaRect(coords_405)
                box_405 = cv2.boxPoints(rect_405)
                box_405 = np.intp(box_405)
                mask_405 = np.zeros_like(gray_image_405, dtype=bool)
                mask_405[coords_405[:, 1], coords_405[:, 0]] = True
                rot_matrix_405 = cv2.getRotationMatrix2D((image_405.shape[1] / 2, image_405.shape[0] / 2), self._calculate_angle(box_405), 1)
                rotated_647_405 = cv2.warpAffine(image_647, rot_matrix_405, (image_647.shape[1], image_647.shape[0]))
                ones_405 = np.ones(shape=(len(box_405), 1))
                points_ones_405 = np.hstack([box_405, ones_405])
                rotated_box_405 = np.int32(rot_matrix_405.dot(points_ones_405.T).T)
        
                gray_image_546 = rgb2gray(image_546)
                label_image_546 = self._find_region(gray_image_546, enhanced=True)
                regions_546 = regionprops(label_image_546)
                largest_region_546 = max(regions_546, key=lambda region: region.area)
                coords_546 = largest_region_546.coords
                coords_546 = coords_546[:, [1, 0]]
                rect_546 = cv2.minAreaRect(coords_546)
                box_546 = cv2.boxPoints(rect_546)
                box_546 = np.intp(box_546)
                mask_546 = np.zeros_like(gray_image_546, dtype=bool)
                mask_546[coords_546[:, 1], coords_546[:, 0]] = True
                rot_matrix_546 = cv2.getRotationMatrix2D((image_546.shape[1] / 2, image_546.shape[0] / 2), self._calculate_angle(box_546), 1)
                rotated_647_546 = cv2.warpAffine(image_647, rot_matrix_546, (image_647.shape[1], image_647.shape[0]))
                ones_546 = np.ones(shape=(len(box_546), 1))
                points_ones_546 = np.hstack([box_546, ones_546])
                rotated_box_546 = np.int32(rot_matrix_546.dot(points_ones_546.T).T)
        
                nuclear_intensity_sum, cell_intensity_sum = self._analyze_yap(rotated_box_405, rotated_647_405, rotated_box_546, rotated_647_546)
                self._save_yap_plot(
                    img_num,
                    image_405,
                    image_546,
                    box_405,
                    box_546,
                    rotated_647_405,
                    rotated_647_546,
                    rotated_box_405,
                    rotated_box_546,
                    cell_type,
                    save_folder_path
                )
        
                return {
                    'scope': scope,
                    'type': cell_type,
                    'path': image_path,
                    'nuclear_yap_intden': sum(nuclear_intensity_sum),
                    'cell_yap_intden': sum(cell_intensity_sum),
                    'calcul_yap': sum(nuclear_intensity_sum)/(sum(cell_intensity_sum)-sum(nuclear_intensity_sum))
                }
            except:
                return None
    
    def _calculate_angle(self, box):
        delta_y = box[1][1] - box[0][1]
        delta_x = box[1][0] - box[0][0]
        angle_radians = math.atan2(delta_y, delta_x)
        return math.degrees(angle_radians)
    
    def _find_region(self, image, enhanced=False):
        if enhanced:
            image = equalize_adapthist(image)
        thresh = threshold_otsu(image)
        binary_image = image > thresh
        cleaned_binary_image = remove_small_objects(binary_image, min_size=500)
        cleaned_binary_image = remove_small_holes(cleaned_binary_image, area_threshold=500)
        label_image = label(cleaned_binary_image)
        regions = regionprops(label_image)
        
        if len(regions) == 0:
            enhanced_image = equalize_adapthist(image, clip_limit=0.04)
            thresh = threshold_otsu(enhanced_image)
            binary_image = enhanced_image > thresh
            cleaned_binary_image = remove_small_objects(binary_image, min_size=500)
            cleaned_binary_image = remove_small_holes(cleaned_binary_image, area_threshold=500)
            label_image = label(cleaned_binary_image)
        
        return label_image
    
    def _load_images(self, image_path):
        image_path_m = image_path.replace('405', 'M')
        image_path_488 = image_path.replace('405', '488')
        image_path_546 = image_path.replace('405', '546')
        image_path_647 = image_path.replace('405', '647')

        if all(os.path.exists(p) for p in [image_path_m, image_path_488]): # merge 데이터가 없는 경우 처리 필요
            image_m = cv2.imread(image_path_m)
            image_488 = cv2.imread(image_path_488)
            
            if os.path.exists(image_path_546):
                image_546 = cv2.imread(image_path_546)
            else:
                image_546 = np.nan
                
            if os.path.exists(image_path_647):
                image_647 = cv2.imread(image_path_647)
            else:
                image_647 = np.nan
                
            return cv2.cvtColor(image_m, cv2.COLOR_BGR2RGB), image_488, image_546, image_647
            
        return None

    def _analyze_intensity(self, box, angle, image_488, image_546):
        rot_matrix = cv2.getRotationMatrix2D((image_488.shape[1] / 2, image_488.shape[0] / 2), angle, 1)
        rotated_488 = cv2.warpAffine(image_488, rot_matrix, (image_488.shape[1], image_488.shape[0]))
        rotated_546 = cv2.warpAffine(image_546, rot_matrix, (image_546.shape[1], image_546.shape[0]))

        x1, y1, x2, y2, rotated_box = self._bounding_box(box, rot_matrix, rotated_546)
        aspect_ratio = (y2 - y1) / (x2 - x1) if (x2 - x1) > (y2 - y1) else (x2 - x1) / (y2 - y1)

        red_intensity_sum, red_average_intensity, green_average_intensity = [], [], []
        for x in range(x1, x2+1) if (x2 - x1) > (y2 - y1) else range(y1, y2+1):
            red_intensity = rotated_546[y1:y2+1, x, 2] if (x2 - x1) > (y2 - y1) else rotated_546[x, x1:x2+1, 2]
            green_intensity = rotated_488[y1:y2+1, x, 1] if (x2 - x1) > (y2 - y1) else rotated_488[x, x1:x2+1, 1]
            red_intensity_sum.append(np.sum(red_intensity))
            red_average_intensity.append(np.mean(red_intensity))
            green_average_intensity.append(np.mean(green_intensity))

        intensities = {
            'red_intensity_sum': np.array(red_intensity_sum),
            'red_average_intensity': np.array(red_average_intensity),
            'green_average_intensity': np.array(green_average_intensity)
        }
        
        return rotated_488, rotated_546, aspect_ratio, intensities, rotated_box

    def _analyze_adhesion(self, array_img):
        R = array_img[:, :, 0]
        G = array_img[:, :, 1]
        B = array_img[:, :, 2]

        condition1 = (R > 80) & (G > 120) & (B < 30)
        condition2 = (R > 120) & (G > 80) & (B < 30)
        condition = condition1 | condition2
        
        binary_mask = np.zeros_like(array_img[:, :, 0], dtype=np.uint8)
        binary_mask[condition] = 1

        structure = np.ones((3, 3), dtype=int)
        binary_mask = ndimage.binary_dilation(binary_mask, structure=structure)
        labeled_array, num_features = ndimage.label(binary_mask)
        area_sizes = ndimage.sum(binary_mask, labeled_array, range(1, num_features + 1))
        
        return area_sizes, condition, labeled_array

    def _analyze_yap(self, rotated_box_405, rotated_647_405, rotated_box_546, rotated_647_546):

        x1_405 = np.min(rotated_box_405[:, 0])
        y1_405 = np.min(rotated_box_405[:, 1])
        x2_405 = np.max(rotated_box_405[:, 0])
        y2_405 = np.max(rotated_box_405[:, 1])

        height_405, width_405 = rotated_647_405.shape[:2]

        x1_405 = max(0, x1_405)
        y1_405 = max(0, y1_405)
        x2_405 = min(width_405 - 1, x2_405)
        y2_405 = min(height_405 - 1, y2_405)

        nuclear_intensity_sum = []

        if x2_405-x1_405 > y2_405-y1_405:
            for x in range(x1_405, x2_405+1):
                nuclear_row_intensity = rotated_647_405[y1_405:y2_405+1, x, 2]
                nuclear_intensity_sum.append(np.sum(nuclear_row_intensity))
                aspect_ratio = (y2_405-y1_405)/(x2_405-x1_405)
        else:
            for y in range(y1_405, y2_405+1):
                nuclear_row_intensity = rotated_647_405[y, x1_405:x2_405+1, 2]
                nuclear_intensity_sum.append(np.sum(nuclear_row_intensity))
                aspect_ratio = (x2_405-x1_405)/(y2_405-y1_405)

        nuclear_intensity_sum = np.array(nuclear_intensity_sum)

        x1_546 = np.min(rotated_box_546[:, 0])
        y1_546 = np.min(rotated_box_546[:, 1])
        x2_546 = np.max(rotated_box_546[:, 0])
        y2_546 = np.max(rotated_box_546[:, 1])

        height_546, width_546 = rotated_647_546.shape[:2]

        x1_546 = max(0, x1_546)
        y1_546 = max(0, y1_546)
        x2_546 = min(width_546 - 1, x2_546)
        y2_546 = min(height_546 - 1, y2_546)

        cell_intensity_sum = []

        if x2_546-x1_546 > y2_546-y1_546:
            for x in range(x1_546, x2_546+1):
                cell_row_intensity = rotated_647_546[y1_546:y2_546+1, x, 2]
                cell_intensity_sum.append(np.sum(cell_row_intensity))
                aspect_ratio = (y2_546-y1_546)/(x2_546-x1_546)
        else:
            for y in range(y1_546, y2_546+1):
                cell_row_intensity = rotated_647_546[y, x1_546:x2_546+1, 2]
                cell_intensity_sum.append(np.sum(cell_row_intensity))
                aspect_ratio = (x2_546-x1_546)/(y2_546-y1_546)

        cell_intensity_sum = np.array(cell_intensity_sum)

        return nuclear_intensity_sum, cell_intensity_sum

    def _create_mask(self, array_img, labeled_array, label, color):
        mask = np.zeros_like(array_img)
        mask[np.isin(labeled_array, label)] = color
        return mask
    
    def _apply_original_filter(self, array_img, labeled_array, label):
        filtered_img = np.zeros_like(array_img)
        mask = np.isin(labeled_array, label)
        filtered_img[mask] = array_img[mask]
        return filtered_img

    def _object_info(self, area_sizes, labeled_array):
        object_data_list = []
        
        label_R = np.where((area_sizes >= 10) & (area_sizes <= 40))[0] + 1
        label_G = np.where((area_sizes > 40) & (area_sizes <= 70))[0] + 1
        label_B = np.where((area_sizes > 70) & (area_sizes <= 100))[0] + 1

        for l_R in label_R:
            object_size = np.sum(labeled_array == l_R)
            object_data_list.append({
                'Label': 'R',
                'Object_ID': l_R,
                'Size': object_size
            })

        for l_G in label_G:
            object_size = np.sum(labeled_array == l_G)
            object_data_list.append({
                'Label': 'G',
                'Object_ID': l_G,
                'Size': object_size
            })

        for l_B in label_B:
            object_size = np.sum(labeled_array == l_B)
            object_data_list.append({
                'Label': 'B',
                'Object_ID': l_B,
                'Size': object_size
            })

        return object_data_list
    
    def _bounding_box(self, box, rot_matrix, rotated_546):
        ones = np.ones(shape=(len(box), 1))
        rotated_box = np.int32(rot_matrix.dot(np.hstack([box, ones]).T).T)
        x1, y1, x2, y2 = np.min(rotated_box[:, 0]), np.min(rotated_box[:, 1]), np.max(rotated_box[:, 0]), np.max(rotated_box[:, 1])
        height, width = rotated_546.shape[:2]
        return max(0, x1), max(0, y1), min(width-1, x2), min(height-1, y2), rotated_box

    def _save_intensity_plot(self, img_num, intensities, cell_type, save_folder_path):
        plt.figure(figsize=(10, 6))
        plt.plot(intensities['green_average_intensity'], label='Green Channel (488 nm)', color='green')
        plt.plot(intensities['red_average_intensity'], label='Red Channel (546 nm)', color='red')
        plt.title('Intensity Profile Along the Major Axis')
        plt.xlabel('Position Along Major Axis')
        plt.ylabel('Average Intensity')
        plt.legend()
        plt.savefig(os.path.join(save_folder_path, f"{img_num}_{cell_type}_intensity_plot.png"), dpi=300, bbox_inches='tight')
        plt.close()

        intensity_df = pd.DataFrame({
            'red_average_intensity': intensities['red_average_intensity'],
            'green_average_intensity': intensities['green_average_intensity']
        })
        intensity_df.to_csv(os.path.join(save_folder_path, f"{img_num}_{cell_type}_intden_data.csv"), index=False)
        

    def _save_region_plot(self, img_num, box, rotated_box, image_488, image_546, rotated_488, rotated_546, cell_type, save_folder_path):
        # Draw the minimum area rectangle on the original and rotated images
        original_with_box_488 = image_488.copy()
        original_with_box_546 = image_546.copy()
        cv2.drawContours(original_with_box_488, [box], 0, (0, 255, 0), 2)  # Draw green box on original 488 image
        cv2.drawContours(original_with_box_546, [box], 0, (0, 0, 255), 2)  # Draw red box on original 546 image

        rotated_with_box_488 = rotated_488.copy()
        rotated_with_box_546 = rotated_546.copy()
        cv2.drawContours(rotated_with_box_488, [rotated_box], 0, (0, 255, 0), 2)  # Draw green box on rotated 488 image
        cv2.drawContours(rotated_with_box_546, [rotated_box], 0, (0, 0, 255), 2)  # Draw red box on rotated 546 image

        original_with_box_546 = cv2.cvtColor(original_with_box_546, cv2.COLOR_BGR2RGB)
        rotated_with_box_546 = cv2.cvtColor(rotated_with_box_546, cv2.COLOR_BGR2RGB)
        
        # Plot the original and rotated images with the detected region
        plt.figure(figsize=(12, 12))
        
        # Plot the original 488 image with the detected region
        plt.subplot(2, 2, 1)
        plt.imshow(original_with_box_488, cmap='gray')
        plt.title('Original Green Channel (488 nm) with Detected Region')
        
        # Plot the rotated 488 image with the detected region
        plt.subplot(2, 2, 2)
        plt.imshow(rotated_with_box_488, cmap='gray')
        plt.title('Rotated Green Channel (488 nm) with Detected Region')
        
        # Plot the original 546 image with the detected region
        plt.subplot(2, 2, 3)
        plt.imshow(original_with_box_546, cmap='gray')
        plt.title('Original Red Channel (546 nm) with Detected Region')
        
        # Plot the rotated 546 image with the detected region
        plt.subplot(2, 2, 4)
        plt.imshow(rotated_with_box_546, cmap='gray')
        plt.title('Rotated Red Channel (546 nm) with Detected Region')
        
        plt.savefig(os.path.join(save_folder_path, f"{img_num}_{cell_type}_rotate_plot.png"), dpi=300, bbox_inches='tight')
        plt.close()

    def _save_adhesion_plot(self, img_num, array_img, condition, labeled_array, image, area_sizes, cell_type, save_folder_path):
        label_R = np.where((area_sizes >= 10) & (area_sizes <= 40))[0] + 1
        label_G = np.where((area_sizes > 40) & (area_sizes <= 70))[0] + 1
        label_B = np.where((area_sizes > 70) & (area_sizes <= 100))[0] + 1
        
        mask_R = self._create_mask(array_img, labeled_array, label_R, [255, 0, 0])
        mask_G = self._create_mask(array_img, labeled_array, label_G, [0, 255, 0])
        mask_B = self._create_mask(array_img, labeled_array, label_B, [0, 0, 255])
        
        filtered_O = np.zeros_like(array_img)
        filtered_O[condition] = array_img[condition]
        filtered_R = self._apply_original_filter(array_img, labeled_array, label_R)
        filtered_G = self._apply_original_filter(array_img, labeled_array, label_G)
        filtered_B = self._apply_original_filter(array_img, labeled_array, label_B)

        fig, axes = plt.subplots(4, 2, figsize=(10,20))

        axes[0, 0].imshow(image)
        axes[0, 0].set_title('Original Image')
        axes[0, 0].axis('off')
        
        axes[0, 1].imshow(filtered_O)
        axes[0, 1].set_title('Filtered Image')
        axes[0, 1].axis('off')
        
        # R 범위
        axes[1, 0].imshow(filtered_R)
        axes[1, 0].set_title('Original Image Filtered (R: 10~40)')
        axes[1, 0].axis('off')
        
        axes[1, 1].imshow(mask_R)
        axes[1, 1].set_title('Filtered Image (R: 10~40)')
        axes[1, 1].axis('off')
        
        # G 범위
        axes[2, 0].imshow(filtered_G)
        axes[2, 0].set_title('Original Image Filtered (G: 41~70)')
        axes[2, 0].axis('off')
        
        axes[2, 1].imshow(mask_G)
        axes[2, 1].set_title('Filtered Image (G: 41~70)')
        axes[2, 1].axis('off')
        
        # B 범위
        axes[3, 0].imshow(filtered_B)
        axes[3, 0].set_title('Original Image Filtered (B: 71~100)')
        axes[3, 0].axis('off')
        
        axes[3, 1].imshow(mask_B)
        axes[3, 1].set_title('Filtered Image (B: 71~100)')
        axes[3, 1].axis('off')

        plt.tight_layout()
        plt.savefig(os.path.join(save_folder_path, f"{img_num}_{cell_type}_adhesion_plot.png"), dpi=300, bbox_inches='tight')
        plt.close()

    def _save_yap_plot(
        self,
        img_num,
        image_405,
        image_546,
        box_405,
        box_546,
        rotated_647_405,
        rotated_647_546,
        rotated_box_405,
        rotated_box_546,
        cell_type,
        save_folder_path
    ):

        original_with_box_405 = copy.deepcopy(image_405)
        original_with_box_546 = copy.deepcopy(image_546)
        cv2.drawContours(original_with_box_405, [box_405], 0, (255, 0, 0), 2)
        cv2.drawContours(original_with_box_546, [box_546], 0, (0, 0, 255), 2)

        rotated_with_box_647_405 = copy.deepcopy(rotated_647_405)
        rotated_with_box_647_546 = copy.deepcopy(rotated_647_546)
        cv2.drawContours(rotated_with_box_647_405, [rotated_box_405], 0, (255, 0, 255), 2) 
        cv2.drawContours(rotated_with_box_647_546, [rotated_box_546], 0, (255, 0, 255), 2)

        original_with_box_405 = cv2.cvtColor(original_with_box_405, cv2.COLOR_BGR2RGB)
        original_with_box_546 = cv2.cvtColor(original_with_box_546, cv2.COLOR_BGR2RGB)
        rotated_with_box_647_405 = cv2.cvtColor(rotated_with_box_647_405, cv2.COLOR_BGR2RGB)
        rotated_with_box_647_546 = cv2.cvtColor(rotated_with_box_647_546, cv2.COLOR_BGR2RGB)

        plt.figure(figsize=(12, 12))
        
        plt.subplot(2, 2, 1)
        plt.imshow(original_with_box_405, cmap='gray')
        plt.title('Original 405nm Detected Region')
        
        plt.subplot(2, 2, 2)
        plt.imshow(rotated_with_box_647_405, cmap='gray')
        plt.title('Rotated 647 nm Detected Region')

        plt.subplot(2, 2, 3)
        plt.imshow(original_with_box_546, cmap='gray')
        plt.title('Original 546nm Detected Region')
        
        plt.subplot(2, 2, 4)
        plt.imshow(rotated_with_box_647_546, cmap='gray')
        plt.title('Rotated 647 nm Detected Region')
        
        plt.savefig(os.path.join(save_folder_path, f'{img_num}_{cell_type}_yap_plot.png'), dpi=300, bbox_inches='tight')
        plt.close()
        