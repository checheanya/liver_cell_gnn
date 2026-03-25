
import os

import numpy as np
import pandas as pd
from PIL import Image
import cv2
import openslide
from matplotlib import pyplot as plt
from tqdm import tqdm


# def find_boxes(wsi_path, data_path, save_dir, patient_label):
#     patient_label = patient_label.astype(str)
#     for patient_id in tqdm(os.listdir(data_path)):
#         wsi_data = os.path.join(wsi_path, patient_id + '.ndpi')
#         patch_size = 256 // 2
#         slide = openslide.open_slide(wsi_data)
#         print("Finished Reading WSI file: ", wsi_data)
#         img = slide.read_region((0, 0), 3, slide.level_dimensions[3]).convert('RGB')
#         print("Finished region extraction.")
#         Ratio = float(slide.properties['openslide.mpp-x'])
#         patches = os.listdir(os.path.join(data_path, patient_id))
#         boxes = [(float(patch.split('_')[0]), float(patch.split('_')[1])) for patch in patches]
#         print("Finished Reading patches.")
#         label = patient_label[patient_label['标本号'] == patient_id]['risk_group'].values[0]
#         print("label: ", label)
#         for box in tqdm(boxes):
#             x, y = box
#             x = x / Ratio / 4 / 2
#             y = y / Ratio / 4 / 2
#             x = int(x)
#             y = int(y)
#             img = np.array(img)
#             print("Finished Reading Image.")
#             # Draw a square with top-left at (x,y) onto img
#             img = cv2.rectangle(img, (y, x), (y + patch_size, x + patch_size), (255, 0, 0), 5)
#             print("Finished Drawing Box.")
#
#         # Visualize img
#         img = Image.fromarray(img)
#         save_path = os.path.join(save_dir, patient_id + '_' + str(len(boxes)) + '_' + label + '.png')
#         # img.show()
#         img.save(save_path)

def find_boxes(wsi_path, data_path, save_dir, patient_label):
    patient_label = patient_label.astype(str)
    patients = os.listdir(data_path)
    patients = ['201714953']
    for patient_id in tqdm(patients):
        wsi_data = os.path.join(wsi_path, patient_id + '.ndpi')
        patch_size = 256 // 2
        slide = openslide.open_slide(wsi_data)
        print("Finished Reading WSI file: ", wsi_data)
        img = slide.read_region((0, 0), 3, slide.level_dimensions[3]).convert('RGB')
        print("Finished region extraction.")
        Ratio = float(slide.properties['openslide.mpp-x'])
        patches = os.listdir(os.path.join(data_path, patient_id))
        boxes = [(float(patch.split('_')[0]), float(patch.split('_')[1])) for patch in patches]
        print("Finished Reading patches.")
        label = patient_label[patient_label['标本号'] == patient_id]['risk_group'].values[0]
        print("label: ", label)
        img = np.array(img)
        for box in tqdm(boxes):
            x, y = box
            x = x / Ratio / 4 / 2
            y = y / Ratio / 4 / 2
            x = int(x)
            y = int(y)
            print("Finished Reading Image.")

            # Center of the rectangular patch
            center_x = y + patch_size // 2
            center_y = x + patch_size // 2

            # Fill patch rectangle with orange
            img[x:x + patch_size, y:y + patch_size] = [255, 165, 0]  # Orange fill
            print("Finished Filling Rectangle with Orange-Red.")

            # Draw a circle centered on the patch
            radius = patch_size // 2  # Radius is half the patch size
            cv2.circle(img, (center_x, center_y), radius, (255, 165, 0), -1)  # Filled orange circle
            print("Finished Drawing Orange-Red Circle.")

            # Heatmap gradient from orange to light blue
            start_color = np.array([255, 165, 0])  # Orange
            end_color = np.array([173, 216, 230])  # Light blue
            max_radius = radius + 50  # Max radius for gradient

            for r in range(radius, max_radius):
                alpha = (r - radius) / (max_radius - radius)  # Interpolation weight
                color = (1 - alpha) * start_color + alpha * end_color  # Interpolated color
                color = tuple(map(int, color))  # Cast to int
                cv2.circle(img, (center_x, center_y), r, color, 2)  # Draw circle
            print("Finished Creating Gradient Effect.")


        # Add circles to img

        # Save result image
        img = Image.fromarray(img)
        save_path = os.path.join(save_dir, patient_id + '_' + str(len(boxes)) + '_' + label + '.png')
        img.save(save_path)

        plt.imshow(img)
        plt.show()


if __name__ == '__main__':
    wsi_path = '/data0/pathology/all_patients'
    data_path = '/data0/yuanyz/NewGraph/datasets/patientminimum_spanning_tree256412/test'
    save_path = '/data0/yuanyz/NewGraph/tools/result/img_new1'
    high_patient = pd.read_csv('/data0/yuanyz/NewGraph/tools/data/high_risk_group.csv')
    low_patient = pd.read_csv('/data0/yuanyz/NewGraph/tools/data/low_risk_group.csv')
    patient_label = pd.concat([high_patient, low_patient], axis=0)
    find_boxes(wsi_path, data_path, save_path, patient_label)
