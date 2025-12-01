import argparse
import os
import sys
from pathlib import Path
import numpy as np
from tqdm import tqdm
from sentence_transformers import SentenceTransformer
import csv

# allow importing utils from project root when running from preprocess/
ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from utils.cache_utils import get_hf_cache_dir
def encode_feature(caption_dir, feature_name, segmentation_dir):
    num_frames = len(os.listdir(segmentation_dir))
    print(f"num_frames:{num_frames}")
    max_id = 0
    model = SentenceTransformer(
        "intfloat/e5-mistral-7b-instruct",
        cache_folder=get_hf_cache_dir(),
    )
    model.max_seq_length = 4096
    output_text_list = [file for file in os.listdir(caption_dir) if "output_text_id" in file]
    os.makedirs(os.path.join(caption_dir,feature_name),exist_ok=True)
    for file in os.listdir(segmentation_dir):
        data = np.load(os.path.join(args.segmentation_dir,file))
        file_max_id = np.max(data)
        if file_max_id > max_id:
            max_id = file_max_id
    print(f"max_id:{max_id}")
    features_np_list = [np.zeros((max_id+1,4096)) for _ in range(num_frames)]


    for caption_file_name in tqdm(output_text_list):
        obj_id = int(caption_file_name.split('id')[1].split('.')[0])
        file_path = os.path.join(caption_dir, caption_file_name)
            
        with open(file_path, mode='r', encoding='utf-8') as input_file:
            reader = csv.reader(input_file)
            # 获取表头
            header = next(reader)
            for row in tqdm(reader):
                frame_id = int(row[0].split('/')[-1].split('.')[0])
                document_embeddings = model.encode(row[-1])
                features_np_list[frame_id-1][obj_id] = document_embeddings

    for _id, features in enumerate(features_np_list):
        np.save(os.path.join(caption_dir,feature_name,f"{_id+1:06}"),features)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--feature_name",type=str,default='features')
    parser.add_argument("--segmentation_dir",type=str)
    parser.add_argument("--output_name",type=str,default='final_features')
    parser.add_argument("--caption_dir",type=str)
    args = parser.parse_args()

    encode_feature(args.caption_dir,args.feature_name,args.segmentation_dir)
    features_dir = os.path.join(args.caption_dir,args.feature_name)
    output_dir = os.path.join(args.caption_dir,args.output_name)
    assert(len(os.listdir(features_dir)) ==  len(os.listdir(args.segmentation_dir)))
    os.makedirs(output_dir,exist_ok=True)
    feature_len = len(os.listdir(features_dir))
    for i in tqdm(range(1, feature_len+1)):
        org_seg_map = np.load(os.path.join(args.segmentation_dir,f"{i:06}.npy"))
        org_feature_map = np.load(os.path.join(features_dir,f"{i:06}.npy"))
        new_seg_map = org_seg_map - 1
        new_feature_map = org_feature_map[1:]
        new_seg_map = new_seg_map[np.newaxis, :, :] 
        np.save(os.path.join(output_dir,f"{i:06}_f.npy"), new_feature_map)
        np.save(os.path.join(output_dir,f"{i:06}_s.npy"), new_seg_map)
    print(f"Final features are saved in {output_dir} ")
    print("Done.")