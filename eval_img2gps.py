import pandas as pd
from PIL import Image
import open_clip
import torch
import os
import numpy as np
import argparse
from torch.utils.data import DataLoader
from tqdm import tqdm
from helper import build_model, Image2GPSDataset, calculate_metric, get_name_predict_gt, calculate_metric_new


# Country list from https://arxiv.org/pdf/2302.00275.pdf, Appendix B.3.1
country_candidates_list_from_streetclip = [
    "Afghanistan", "Albania", "Algeria", "Andorra", "Angola", "Antigua and Barbuda", "Argentina", "Armenia", "Aruba",
    "Australia", "Austria", "Azerbaijan", "Bahamas", "Bahrain", "Bangladesh", "Barbados", "Belarus", "Belgium",
    "Belize", "Benin", "Bhutan", "Bolivia", "Bosnia and Herzegovina", "Botswana", "Brazil", "Brunei", "Bulgaria",
    "Burkina Faso", "Burundi", "Cabo Verde", "Cambodia", "Cameroon", "Canada", "Central African Republic", "Chad",
    "Chile", "China", "Colombia", "Comoros", "Costa Rica", "Croatia", "Cuba", "Cyprus", "Czech Republic",
    "Côte d'Ivoire", "Democratic Republic of the Congo", "Denmark", "Djibouti", "Dominica", "Dominican Republic",
    "East Timor", "Ecuador", "Egypt", "El Salvador", "Equatorial Guinea", "Eritrea", "Estonia", "Ethiopia", "Fiji",
    "Finland", "France", "Gabon", "Gambia", "Georgia", "Germany", "Ghana", "Greece", "Greenland", "Grenada",
    "Guatemala", "Guinea", "Guinea-Bissau", "Guyana", "Haiti", "Honduras", "Hungary", "Iceland", "India",
    "Indonesia", "Iran", "Iraq", "Ireland", "Israel", "Italy", "Jamaica", "Japan", "Jordan", "Kazakhstan",
    "Kenya", "Kuwait", "Kyrgyzstan", "Laos", "Latvia", "Lebanon", "Lesotho", "Liberia", "Libya", "Liechtenstein",
    "Lithuania", "Luxembourg", "Madagascar", "Malawi", "Malaysia", "Mali", "Malta", "Marshall Islands", "Mauritania",
    "Mauritius", "Mexico", "Micronesia", "Moldova", "Mongolia", "Montenegro", "Morocco", "Mozambique", "Myanmar",
    "Namibia", "Nauru", "Nepal", "Netherlands", "New Zealand", "Nicaragua", "Niger", "Nigeria", "North Korea",
    "North Macedonia", "Norway", "Oman", "Pakistan", "Palau", "Palestine", "Panama", "Papua New Guinea", "Paraguay",
    "Peru", "Philippines", "Poland", "Portugal", "Qatar", "Republic of the Congo", "Romania", "Russia", "Rwanda",
    "Saint Kitts and Nevis", "Saint Lucia", "Saint Vincent and the Grenadines", "Samoa", "San Marino", "Saudi Arabia",
    "Senegal", "Serbia", "Seychelles", "Sierra Leone", "Singapore", "Slovakia", "Slovenia", "Solomon Islands",
    "Somalia", "South Africa", "South Korea", "South Sudan", "Spain", "Sri Lanka", "Sudan", "Suriname", "Swaziland",
    "Sweden", "Switzerland", "Syria", "Taiwan", "Tajikistan", "Tanzania", "Thailand", "Togo", "Tonga",
    "Trinidad and Tobago", "Tunisia", "Turkey", "Turkmenistan", "Tuvalu", "Uganda", "Ukraine", "United Arab Emirates",
    "United Kingdom", "Uruguay", "Uzbekistan", "Vanuatu", "Venezuela", "Vietnam", "Yemen", "Zambia", "Zimbabwe",
    # "United States"
]

# mismatch between names in country list from the paper and names in SimpleMaps-Basic database
# (https://simplemaps.com/data/world-cities)
streetclip2simplemap = {
    "Bahamas": "The Bahamas",
    "Czech Republic": "Czechia",
    "Democratic Republic of the Congo": "Congo (Kinshasa)",
    "East Timor": "Timor-Leste",
    "Gambia": "The Gambia",
    "Micronesia": "Federated States of Micronesia",
    "North Macedonia": "Macedonia",
    "Republic of the Congo": "Congo (Brazzaville)"
}

US_states = [
    "Alabama", "Alaska", "Arizona", "Arkansas", "California", "Colorado", "Connecticut", "Delaware", "Florida",
    "Georgia", "Hawaii", "Idaho", "Illinois", "Indiana", "Iowa", "Kansas", "Kentucky", "Louisiana", "Maine", "Maryland",
    "Massachusetts", "Michigan", "Minnesota", "Mississippi", "Missouri", "Montana", "Nebraska", "Nevada", "New Hampshire",
    "New Jersey", "New Mexico", "New York", "North Carolina", "North Dakota", "Ohio", "Oklahoma", "Oregon", "Pennsylvania",
    "Rhode Island", "South Carolina", "South Dakota", "Tennessee", "Texas", "Utah", "Vermont", "Virginia", "Washington",
    "West Virginia", "Wisconsin", "Wyoming"
]

country_candidates_list_from_streetclip += US_states


def eval_geolocation_openclip(img_dir, model_name, model_path, country_list, simple_map_df, num_worker, batch_size, topk_city_per_country=30):

    def country2city(country):
        if country != "Palestine":
            cities_df = simple_map_df[simple_map_df["country"] == country]
        else:
            # "Palestine" does not exist in the database from SimpleMaps.
            # For the purpose of creating text context for Palestine in order to make CLIP-like models work,
            # I combine two regions ("Gaza Strip" and "West Bank") from the SimpleMaps to represent the Palestine.
            # There is no political meaning behind the data processing. I stand with UN's resolution.
            cities_1 = simple_map_df[simple_map_df["country"] == "Gaza Strip"]
            cities_2 = simple_map_df[simple_map_df["country"] == "West Bank"]
            cities_df = pd.concat([cities_1, cities_2])
        if country in US_states:
            cities_df = simple_map_df[simple_map_df["country"] == "United States"]
        cities_df_sorted = cities_df.sort_values("population", ascending=False)
        city_names = cities_df_sorted["city_ascii"].tolist()[:topk_city_per_country]
        return city_names

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, img_preprocess = build_model(
            model_name,
            model_path,
            device,
            use_streetclip=False
        )
    tokenizer = open_clip.tokenize
    img2gps_dataset = Image2GPSDataset(img_dir, img_preprocess)

    img2gps_dataloader = DataLoader(
        img2gps_dataset,
        num_workers=num_worker,
        batch_size=batch_size,
        pin_memory=True,
        shuffle=False
    )

    # Stage 1: Linear prob a country
    # A Street View photo in {state}, United States.
    country_text_prompt = ["A Street View photo in {}.".format(country) if country not in US_states else "A Street View photo in {}, United States.".format(country) for country in country_list]
    img2country_sim_list = []
    img_name_list = []
    image_normed_features_list = []
    for index, (img_names, imgs) in tqdm(enumerate(img2gps_dataloader)):
        text = tokenizer(country_text_prompt)
        imgs = imgs.to(device)
        text = text.to(device)
        with torch.no_grad(), torch.cuda.amp.autocast():
            image_features = model.encode_image(imgs)
            text_features = model.encode_text(text)
            image_features /= image_features.norm(dim=-1, keepdim=True)
            text_features /= text_features.norm(dim=-1, keepdim=True)
            image_normed_features_list.append(image_features)
            text_probs = (100.0 * image_features @ text_features.T).softmax(dim=-1).cpu()
            img2country_sim_list.append(text_probs)
            img_name_list.extend(img_names)
    img2country_sim = torch.cat(img2country_sim_list).numpy()
    image_normed_features = torch.cat(image_normed_features_list)
    img2country_argmax = np.argmax(img2country_sim, 1)
    selected_country = [country_list[index] for index in img2country_argmax]
    selected_country = [streetclip2simplemap[country_name] if country_name in streetclip2simplemap else country_name for country_name in selected_country]

    # Stage 2: Linear prob a city
    result_dict = dict()
    selected_city = [country2city(country) for country in selected_country]

    for i, city_list in tqdm(enumerate(selected_city)):
        city_text_prompt = ["A Street View photo from {}.".format(city) for city in city_list]
        text = tokenizer(city_text_prompt)
        text = text.to(device)
        with torch.no_grad(), torch.cuda.amp.autocast():
            text_features = model.encode_text(text)
            text_features /= text_features.norm(dim=-1, keepdim=True)
            img_name = img_name_list[i]
            img_normed_feat = image_normed_features[i].unsqueeze(0)
            country_name = selected_country[i]
            text_probs = (100.0 * img_normed_feat @ text_features.T).softmax(dim=-1).cpu().numpy()
        img2city_argmax = np.argmax(text_probs, 1)[0]
        select_city_name = city_list[img2city_argmax]
        lat = simple_map_df[simple_map_df["city_ascii"] == select_city_name]["lat"].tolist()[0]
        lng = simple_map_df[simple_map_df["city_ascii"] == select_city_name]["lng"].tolist()[0]
        result_dict[img_name] = [country_name, select_city_name, lat, lng]

    return result_dict


def eval_geolocation_streetclip(img_dir, model_name, model_path, country_list, simple_map_df, num_worker, batch_size, topk_city_per_country=30):

    def country2city(country):
        if country != "Palestine":
            cities_df = simple_map_df[simple_map_df["country"] == country]
        else:
            # "Palestine" does not exist in the database from SimpleMaps.
            # For the purpose of creating text context for Palestine in order to make CLIP-like models work,
            # I combine two regions ("Gaza Strip" and "West Bank") from the SimpleMaps to represent the Palestine.
            # There is no political meaning behind the data processing. I stand with UN's resolution.
            cities_1 = simple_map_df[simple_map_df["country"] == "Gaza Strip"]
            cities_2 = simple_map_df[simple_map_df["country"] == "West Bank"]
            cities_df = pd.concat([cities_1, cities_2])
        if country in US_states:
            cities_df = simple_map_df[simple_map_df["country"] == "United States"]
        cities_df_sorted = cities_df.sort_values("population", ascending=False)
        city_names = cities_df_sorted["city_ascii"].tolist()[:topk_city_per_country]
        return city_names

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, img_preprocess = build_model(
            model_name,
            model_path,
            device,
            use_streetclip=True
        )
    img_names = os.listdir(img_dir)
    img_pils = [Image.open(os.path.join(img_dir, img_name)) for img_name in img_names]
    img_pils = [img_pils[i:i + batch_size] for i in range(0, len(img_pils), batch_size)]
    # Stage 1: Linear prob a country
    country_text_prompt = ["A Street View photo in {}.".format(country) for country in country_list]
    img2country_sim_list = []
    for i, img_pil in tqdm(enumerate(img_pils)):
        inputs = img_preprocess(text=country_text_prompt, images=img_pil, return_tensors="pt", padding=True)
        inputs['pixel_values'] = inputs['pixel_values'].to(device)
        inputs['input_ids'] = inputs['input_ids'].to(device)
        inputs['attention_mask'] = inputs['attention_mask'].to(device)
        with torch.no_grad(), torch.cuda.amp.autocast():
            outputs = model(**inputs)
            logits_per_image = outputs.logits_per_image.softmax(dim=1).cpu()  # this is the image-text similarity score
            img2country_sim_list.append(logits_per_image)
    img2country_sim = torch.cat(img2country_sim_list).numpy()
    img2country_argmax = np.argmax(img2country_sim, 1)
    selected_country = [country_list[index] for index in img2country_argmax]
    selected_country = [streetclip2simplemap[country_name] if country_name in streetclip2simplemap else country_name for country_name in selected_country]

    # Stage 2: Linear prob a city
    img_pils = [item for img_pil in img_pils for item in img_pil]
    result_dict = dict()
    selected_city = [country2city(country) for country in selected_country]
    for i, city_list in tqdm(enumerate(selected_city)):
        city_text_prompt = ["A Street View photo from {}.".format(city) for city in city_list]
        img_pil = img_pils[i]
        inputs = img_preprocess(text=city_text_prompt, images=img_pil, return_tensors="pt", padding=True)
        inputs['pixel_values'] = inputs['pixel_values'].to(device)
        inputs['input_ids'] = inputs['input_ids'].to(device)
        inputs['attention_mask'] = inputs['attention_mask'].to(device)
        with torch.no_grad(), torch.cuda.amp.autocast():
            outputs = model(**inputs)
            text_probs = outputs.logits_per_image.softmax(dim=1).cpu().numpy()  # this is the image-text similarity score
        img_name = img_names[i]
        country_name = selected_country[i]
        img2city_argmax = np.argmax(text_probs, 1)[0]
        select_city_name = city_list[img2city_argmax]
        lat = simple_map_df[simple_map_df["city_ascii"] == select_city_name]["lat"].tolist()[0]
        lng = simple_map_df[simple_map_df["city_ascii"] == select_city_name]["lng"].tolist()[0]
        result_dict[img_name] = [country_name, select_city_name, lat, lng]

    return result_dict


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model-name", default="ViT-B-32", type=str,
        help="ViT-B-32 or ViT-L-14-336 or ViT-H-14",
    )
    parser.add_argument(
        "--ckpt-path", default="/media/zilun/mx500/2-year-work/MLLM_geoguesser/StreetCLIP", type=str,
        help="Path to ViT-B-32.pt",
    )
    parser.add_argument(
        "--random-seed", default=3407, type=int,
        help="random seed",
    )
    parser.add_argument(
        "--test-dataset-dir",
        default="./data/img2gps3k_dataset/image",
        # default="./data/img2gps_dataset/image",
        type=str,
        help="test dataset dir",
    )
    parser.add_argument(
        "--gps-mat-path",
        default="./data/img2gps3k_dataset/img2gps3k_gps.mat",
        # default="./data/img2gps_dataset/img2gps_gps.mat",
        type=str,
        help="gps path",
    )
    parser.add_argument(
        "--imgname-mat-path",
        default="./data/img2gps3k_dataset/img2gps3k_image_names.mat",
        # default="./data/img2gps_dataset/img2gps_image_names.mat",
        type=str,
        help="imgname path",
    )
    parser.add_argument(
        "--imgname-coord-path",
        default="./data/img2gps3k_dataset/img2gps3k_dataset_2997.csv",
        # default="./data/img2gps_dataset/img2gps_dataset_237.csv",
        type=str,
        help="path of img2gps_dataset_237.csv/img2gps3k_dataset_2997.csv",
    )
    parser.add_argument(
        "--workers", default=8, type=int,
        help="number of workers",
    )
    parser.add_argument(
        "--batch-size", default=200, type=int,
        help="batch size",
    )

    parser.add_argument(
        "--city-coord-csv", default="./data/simplemaps_worldcities_basicv1.76/worldcities.csv", type=str,
        help="batch size",
    )
    args = parser.parse_args()
    args.device = "cuda" if torch.cuda.is_available() else "cpu"
    print(args)

    # convert_mat2csv(args)
    # d = gps_distance(32.845269, -117.273988, 45.433703, 12.339298)
    gt_coord_df = pd.read_csv(args.imgname_coord_path)

    city_coord_df = pd.read_csv(args.city_coord_csv)
    print(len(country_candidates_list_from_streetclip), city_coord_df.shape)

    countries = city_coord_df["country"].tolist()
    print(len(set(countries)))
    print(list(set(countries)))
    for country in country_candidates_list_from_streetclip:
        if country not in countries:
            print("{} not in the country list from df".format(country))

    result_dict_georsclip = eval_geolocation_openclip(
        args.test_dataset_dir,
        args.model_name,
        args.ckpt_path,
        country_candidates_list_from_streetclip,
        city_coord_df,
        args.workers,
        args.batch_size,
        topk_city_per_country=30
    )
    georsclip_img_name_list, georsclip_predict_gps_list, georsclip_gt_gps_list = get_name_predict_gt(result_dict_georsclip, gt_coord_df)
    calculate_metric(georsclip_img_name_list, georsclip_predict_gps_list, georsclip_gt_gps_list)

    result_dict_streetclip = eval_geolocation_streetclip(
        args.test_dataset_dir,
        args.model_name,
        args.ckpt_path,
        country_candidates_list_from_streetclip,
        city_coord_df,
        args.workers,
        args.batch_size,
        topk_city_per_country=30
    )
    streetclip_img_name_list, streetclip_predict_gps_list, streetclip_gt_gps_list = get_name_predict_gt(result_dict_streetclip, gt_coord_df)
    calculate_metric(streetclip_img_name_list, streetclip_predict_gps_list, streetclip_gt_gps_list)


if __name__ == "__main__":

    main()
