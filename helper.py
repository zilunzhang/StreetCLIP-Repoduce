import numpy as np
import pandas as pd
from PIL import Image
import open_clip
import os
import scipy.io as sio
import math
from torch.utils.data import Dataset, DataLoader
from transformers import CLIPProcessor, CLIPModel
import torch


class Image2GPSDataset(Dataset):
    def __init__(self, img_dir, transform):
        self.img_dir = img_dir
        self.img_names = [img_name for img_name in os.listdir(self.img_dir) if img_name.endswith("jpg")]
        self.transform = transform

    def __len__(self):
        return len(self.img_names)

    def __getitem__(self, idx):
        img_name = self.img_names[idx]
        img_path = os.path.join(self.img_dir, img_name)
        image = Image.open(img_path)
        image = self.transform(image)
        return img_name, image


def read_img_comment(
        img_path="/media/zilun/mx500/2-year-work/MLLM_geoguesser/revisiting-im2gps/query/gps_query_imgs/venice_00016_265350847_0252df858c_118_44972214@N00.jpg"):
    im = Image.open(img_path)
    img_comment = im.app["COM"]
    return img_comment


def sind(degree):
    return math.sin(math.radians(degree))


def cosd(degree):
    return math.cos(math.radians(degree))


def gps_distance(lat1, long1, lat2, long2):
    delta_long = -(long1 - long2)
    delta_lat = -(lat1 - lat2)
    a = sind(delta_lat / 2) ** 2 + cosd(lat1) * cosd(lat2) * (sind(delta_long / 2) ** 2)
    c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))
    R = 6371
    d = R * c
    return d


def haversine(lat1, lon1, lat2, lon2):
    """
    Calculate the great circle distance in kilometers between two points
    on the earth (specified in decimal degrees)
    """
    # Convert decimal degrees to radians
    lat1, lon1, lat2, lon2 = map(math.radians, [lat1, lon1, lat2, lon2])

    # Haversine formula
    dlon = lon2 - lon1
    dlat = lat2 - lat1
    a = math.sin(dlat / 2) ** 2 + math.cos(lat1) * math.cos(lat2) * math.sin(dlon / 2) ** 2
    c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))

    # Radius of earth in kilometers. Use 3956 for miles
    r = 6371

    # Calculate the result
    return c * r


def convert_mat2csv(args):
    fname_list, latitude_list, longitude_list = [], [], []
    imgname_mat_contents = sio.loadmat(args.imgname_mat_path)["image_names"].tolist()[0]
    gps_mat_contents = sio.loadmat(args.gps_mat_path)["query_file_gps"].tolist()[0]

    for i, content in enumerate(zip(imgname_mat_contents, gps_mat_contents)):
        img_name, gps_coord = content[0][0], content[1][0]
        lat, long = gps_coord
        img_path = os.path.join(args.test_dataset_dir, img_name)
        assert os.path.exists(img_path)
        fname_list.append(img_name)
        latitude_list.append(lat)
        longitude_list.append(long)

    img2gps_df = pd.DataFrame(
        {
            "filename": fname_list,
            "latitude": latitude_list,
            "longitude": longitude_list
        }
    )
    print(img2gps_df)
    img2gps_df.to_csv("img2gps3k_dataset_2997.csv", index=False)


def _convert_to_rgb(image):
    return image.convert('RGB')


def build_model(model_name, ckpt_path, device, use_streetclip=True):
    if use_streetclip:
        model = CLIPModel.from_pretrained(ckpt_path)
        preprocess_val = CLIPProcessor.from_pretrained(ckpt_path)
        model = model.to(device)

    else:
        if model_name == "ViT-B-32":
            image_resolution = 224
            model, _, preprocess_val = open_clip.create_model_and_transforms(model_name, pretrained="openai")
            # checkpoint = torch.load(ckpt_path, map_location="cpu")
            # model.load_state_dict(checkpoint)

        if model_name == "ViT-L-14-336":
            image_resolution = 336
            model, _, preprocess_val = open_clip.create_model_and_transforms("ViT-L-14-336", pretrained="openai")
            # checkpoint = torch.load(ckpt_path, map_location="cpu")
            # model.load_state_dict(checkpoint)

        elif model_name == "ViT-H-14":
            image_resolution = 224
            model, _, preprocess_val = open_clip.create_model_and_transforms(model_name, pretrained="laion2b_s32b_b79k")
            # checkpoint = torch.load(ckpt_path, map_location="cpu")
            # model.load_state_dict(checkpoint)

        model = model.to(device)

    return model, preprocess_val


def get_name_predict_gt(result_dict, gt_coord_df):
    img_name_list, predict_gps_list, gt_gps_list = [], [], []
    for index, row in gt_coord_df.iterrows():
        img_name = str(row["filename"])
        predict_lat = result_dict[img_name][2]
        predict_lng = result_dict[img_name][3]
        gt_lat = row["latitude"]
        gt_lng = row["longitude"]
        img_name_list.append(img_name)
        predict_gps_list.append([predict_lat, predict_lng])
        gt_gps_list.append([gt_lat, gt_lng])

    return img_name_list, predict_gps_list, gt_gps_list


# https://github.com/TIBHannover/GeoEstimation/blob/master/classification/utils_global.py
def vectorized_gc_distance(latitudes, longitudes, latitudes_gt, longitudes_gt):
    R = 6371
    factor_rad = 0.01745329252
    longitudes = factor_rad * longitudes
    longitudes_gt = factor_rad * longitudes_gt
    latitudes = factor_rad * latitudes
    latitudes_gt = factor_rad * latitudes_gt
    delta_long = longitudes_gt - longitudes
    delta_lat = latitudes_gt - latitudes
    subterm0 = torch.sin(delta_lat / 2) ** 2
    subterm1 = torch.cos(latitudes) * torch.cos(latitudes_gt)
    subterm2 = torch.sin(delta_long / 2) ** 2
    subterm1 = subterm1 * subterm2
    a = subterm0 + subterm1
    c = 2 * torch.asin(torch.sqrt(a))
    gcd = R * c
    return gcd


# https://github.com/TIBHannover/GeoEstimation/blob/master/classification/utils_global.py
def gcd_threshold_eval(gc_dists, thresholds):
    # calculate accuracy for given gcd thresolds
    results = {}
    for thres in thresholds:
        results[thres] = torch.true_divide(
            torch.sum(gc_dists <= thres), len(gc_dists)
        ).item()
    return results


def calculate_metric(img_name_list, predict_gps_list, gt_gps_list):
    d_error_list = [1, 25, 200, 750, 2500]
    assert len(img_name_list) == len(predict_gps_list) == len(gt_gps_list)

    errors = []
    img_names = []
    for i in range(len(img_name_list)):
        img_name = img_name_list[i]
        predict_gps = predict_gps_list[i]
        gt_gps = gt_gps_list[i]
        d = gps_distance(gt_gps[0], gt_gps[1], predict_gps[0], predict_gps[1])
        errors.append(d)
        img_names.append(img_name)

    for d_error in d_error_list:
        correct_count = 0
        for error in errors:
            if error <= d_error:
                correct_count += 1
        acc = (correct_count / len(errors))
        print('Accuracy at {} km is {}'.format(d_error, acc))

    return errors


def calculate_metric_new(img_name_list, predict_gps_list, gt_gps_list):
    predict_gps_list = torch.from_numpy(np.array(predict_gps_list))
    gt_gps_list = torch.from_numpy(np.array(gt_gps_list))
    gc_dists = vectorized_gc_distance(predict_gps_list[:, 0], predict_gps_list[:, 1], gt_gps_list[:, 0], gt_gps_list[:, 1])
    result = gcd_threshold_eval(gc_dists, thresholds=[1, 25, 200, 750, 2500])
    print(result)
    return gc_dists