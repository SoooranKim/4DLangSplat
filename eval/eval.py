import cv2
import numpy as np
import torch
import time
from tqdm import tqdm
import colormaps
import json
import os
import glob
import random
import sys
sys.path.append("..")
import logging
from openclip_encoder import OpenCLIPNetwork
from autoencoder.model import Autoencoder, VanillaVAE
from pathlib import Path
from eval_utils import smooth, smooth_cuda, colormap_saving, vis_mask_save, polygon_to_mask, stack_mask, show_result
import argparse
from collections import defaultdict
from typing import Dict, Union
import matplotlib.pyplot as plt
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer
from sklearn.metrics import confusion_matrix
import seaborn as sns
import csv

def get_logger(name, log_file=None, log_level=logging.INFO, file_mode='w'):
    logger = logging.getLogger(name)
    stream_handler = logging.StreamHandler()
    handlers = [stream_handler]

    if log_file is not None:
        file_handler = logging.FileHandler(log_file, file_mode)
        handlers.append(file_handler)

    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    for handler in handlers:
        handler.setFormatter(formatter)
        handler.setLevel(log_level)
        logger.addHandler(handler)
    logger.setLevel(log_level)
    return logger



def eval_gt_lerfdata(json_folder: Union[str, Path] = None, output_path: Path = None, prompts=None,replace_prompts=None, dataset_type=None) -> Dict:
    """
    prompts: if pompts is None, check all words else check only words in prompts
    replace_prompts: dict() replace the prompt for query
    Organize lerf's gt annotations
    gt format:
        file name: frame_xxxxx.json
        file content: labelme format
    return:
        gt_ann: dict()
            keys: str(int(idx))
            values: dict()
                keys: str(label)
                values: dict() which contain 'bboxes' and 'mask'
    """
    # Load the COCO format json file
    with open(os.path.join(json_folder, '_annotations.coco.json'), 'r') as f:
        data = json.load(f)

    gt_ann = {}
    img_paths = []
    id2name = {}
    name2id = {}
    im_id2imidx = {}
    # test_img_ids = data['test_img_ids']
    for item in data['categories']:
        idx = item['id']
        id2name[int(idx)] = item['name']
        name2id[item['name']] = int(idx)
    for img_data in data['images']:
        img_ann = defaultdict(dict)
        idx = img_data['id']
        # if idx=='0':
        #     import ipdb; ipdb.set_trace()
        img_name = img_data['file_name']
        img_paths.append(os.path.join(json_folder, img_name))
        h, w = img_data['height'], img_data['width']
        
        # import pdb; pdb.set_trace()
        for annotation in data['annotations']:
            if annotation['image_id'] == idx:
                label = id2name[annotation['category_id']]
                if prompts is not None and label not in prompts:
                    # print(f"skip {label}")
                    continue

                box = np.asarray(annotation['bbox']).reshape(-1)  # x1, y1, width, height
                box[2] += box[0]    
                box[3] += box[1]
                segmentation = annotation['segmentation'][0]
                assert len(segmentation)%2 == 0
                point_segmentation = []
                for i in range(0,len(segmentation),2):
                    point_segmentation.append([segmentation[i], segmentation[i+1]])
                
                mask = polygon_to_mask((h, w), point_segmentation)
                if replace_prompts is not None and label in replace_prompts.keys():
                    label_list = replace_prompts[label]
                    label_list.append(label)
                else:
                    label_list = [label]
                for label in label_list:
                    if img_ann[label].get('mask', None) is not None:
                        mask = stack_mask(img_ann[label]['mask'], mask)
                        img_ann[label]['bboxes'] = np.concatenate(
                            [img_ann[label]['bboxes'].reshape(-1, 4), box.reshape(-1, 4)], axis=0)
                    else:
                        img_ann[label]['bboxes'] = box
                    img_ann[label]['mask'] = mask
                
                    # Save for visualization
                    save_path = output_path / 'gt' / img_name.split('.')[0] / f'{label}.jpg'
                    if args.visualize_results:
                        save_path.parent.mkdir(exist_ok=True, parents=True)
                        vis_mask_save(mask, save_path)
        
        gt_ann[f'{idx}'] = img_ann
    

    for item in data['images']:
        idx = item['id']
        filename = item['file_name']
        
        if args.dataset_type == 'hypernerf':
            im_id2imidx[idx] = int(filename.split('_')[0])-1 # 减一是为了对齐npy文件序号和image idx
        else:
            im_id2imidx[idx] = int(filename.split('_')[0]) # 减一是为了对齐npy文件序号和image idx

        # print(idx)
        # print(int(filename.split('_')[0])-1)
    

    return gt_ann, (h, w), img_paths, id2name, name2id, im_id2imidx


def activate_stream(sem_map, 
                    image, 
                    clip_model, 
                    image_name: Path = None,
                    img_ann: Dict = None, 
                    thresh : float = 0.5, 
                    colormap_options = None,
                    name2id = None,
                    scale = 30, 
                    chose_mask_strategy='point', 
                    imageid=None,
                    visualize_results=False):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    valid_map = clip_model.get_max_across(sem_map)                 # 3xkx832x1264
    n_head, n_prompt, h, w = valid_map.shape

    # positive prompts
    chosen_iou_list, chosen_lvl_list = [], []
    prompt_iou_lvl_dict ={}
    mask_dict = {}
    mask_for_video_dict={}
    for k in range(n_prompt):
        iou_lvl = torch.zeros(n_head).to(device)
        mask_lvl = torch.zeros((n_head, h, w)).to(device)
        mask_for_video = torch.zeros((n_head, h, w)).to(device)
        output_list = []
        thresh_list = []
        for i in range(n_head):

            avg_pool = torch.nn.AvgPool2d(kernel_size=scale, stride=1, padding=14, count_include_pad=False).to(device)

            avg_filtered = avg_pool(valid_map[i][k].unsqueeze(0).unsqueeze(0))
            valid_map[i][k] = 0.5 * (avg_filtered.squeeze(0).squeeze(0) + valid_map[i][k])
            

            if visualize_results:
                output_path_relev = image_name / 'heatmap' / f'{clip_model.positives[k]}_{i}'
                output_path_relev.parent.mkdir(exist_ok=True, parents=True)
                colormap_saving(valid_map[i][k].unsqueeze(-1), colormap_options,
                                output_path_relev)
            
            # truncate the heatmap into mask
            output = valid_map[i][k]
            output = output - torch.min(output)
            output = output / (torch.max(output) + 1e-9)
            output = output * (1.0 - (-1.0)) + (-1.0)
            output = torch.clip(output, 0, 1)
            output_list.append(output)

            
            thresh_list.append(thresh)
            if os.getenv("DEBUG",'f') == 't':
                print(f"use uniform thresh:{thresh}")

                
            if visualize_results:
                p_i = torch.clip(valid_map[i][k] - 0.5, 0, 1).unsqueeze(-1)
                valid_composited = colormaps.apply_colormap(p_i / (p_i.max() + 1e-6), colormaps.ColormapOptions("turbo"))
                mask = (valid_map[i][k] < 0.5).squeeze()
                valid_composited[mask, :] = image[mask, :] * 0.6
                output_path_compo = image_name / 'composited' / f'{clip_model.positives[k]}_{i}'
                output_path_compo.parent.mkdir(exist_ok=True, parents=True)
                colormap_saving(valid_composited, colormap_options, output_path_compo)
            
            if i==0 and visualize_results and img_ann is not None:
                background_only = image.clone()
                background_only = background_only * 0.6
                output_path_background = image_name / 'background' / f'{clip_model.positives[k]}_{i}'
                output_path_background.parent.mkdir(exist_ok=True, parents=True)
                colormap_saving(background_only, colormap_options, output_path_background)
                
                overlay_color = torch.tensor([128 / 255, 0.0, 128 / 255]).cuda()  
                promtp_name = clip_model.positives[k]
                if promtp_name in img_ann:
                    mask_gt = img_ann[promtp_name]['mask'].astype(np.uint8)
                    overlay_layer = overlay_color * 0.5  
                    annotated_image = image.clone()
                    annotated_image[mask_gt.squeeze() > 0] = annotated_image[mask_gt.squeeze() > 0] * 0.5 + overlay_layer * 255

                    output_path_annotation = image_name / 'annotation' / f'{clip_model.positives[k]}_{i}'
                    output_path_annotation.parent.mkdir(exist_ok=True, parents=True)
                    colormap_saving(annotated_image, colormap_options, output_path_annotation)
            

            if os.getenv("adaptive_thresh",'f') == 't':
                low_filter = os.getenv("low_thresh_filter",0)
                if low_filter>0:
                    mask_area = (output.cpu().numpy() > thresh).astype(np.uint8)
                    mask_area = smooth(mask_area)
                    mean_ = valid_map[i][k][mask_area].mean().item()
                else:
                    mean_ = valid_map[i][k].mean().item()

                alpha = os.getenv("adaptive_alpha",None)
                alpha = float(alpha)
                assert alpha>0 and alpha < 1
                thresh = (mean_  -1 ) * alpha + 1
                logger.info(f'mean_:{mean_}, adaptive thresh: {thresh}')

            mask_pred = (output > thresh).type(torch.uint8)
            mask_for_video[i] = mask_pred
            mask_pred = smooth_cuda(mask_pred)
            mask_lvl[i] = mask_pred
            
            promtp_name = clip_model.positives[k]
            if img_ann is not None and promtp_name in img_ann:
                mask_gt = torch.from_numpy(img_ann[promtp_name]['mask'].astype(np.uint8)).to(device)
                intersection = torch.sum(torch.logical_and(mask_gt, mask_pred))
                union = torch.sum(torch.logical_or(mask_gt, mask_pred))
                iou = torch.sum(intersection) / torch.sum(union) if torch.sum(union) > 0 else 0.0
            else:
                iou = 0.0
            iou_lvl[i] = iou


        score_lvl = torch.zeros((n_head,), device=valid_map.device)
        

        for i in range(n_head):
            if chose_mask_strategy == "point":
                score = valid_map[i, k].max()
                score_lvl[i] = score
            elif chose_mask_strategy == "mean":
                # Calculate the average score within the thresholded mask
                
                thresh = thresh_list[i]
                chose_mask_area = (output_list[i].cpu().numpy() > thresh).astype(np.uint8)
 
                chose_mask_area_after_smooth = chose_mask_area
                
                if np.sum(chose_mask_area_after_smooth) > 0:
                    score = valid_map[i, k][chose_mask_area_after_smooth].mean().item()
                    print("score:",score)
                else:
                    score = 0

                score_lvl[i] = score
            elif chose_mask_strategy == 'adaptive':
                low_filter = os.getenv("low_thresh_filter",0)
                if low_filter>0:
                    chose_mask_area = (output_list[i].cpu().numpy() > thresh).astype(np.uint8)
                    chose_mask_area = smooth(chose_mask_area)
                    mean_ = valid_map[i, k][chose_mask_area].mean().item()
                else:
                    mean_ = valid_map[i, k].mean().item()
                alpha = os.getenv("adaptive_alpha",None)
                assert alpha>0 and alpha < 1
                adaptive_thresh = (mean_  -1 ) * alpha + 1
                chose_mask_area = (output_list[i].cpu().numpy() > adaptive_thresh).astype(np.uint8)
                chose_mask_area_after_smooth = smooth(chose_mask_area)
            
            else:
                raise NotImplementedError

        chosen_lvl = torch.argmax(score_lvl)
        if os.getenv("DEBUG",'f') == 't':
            print(f"chosen_lvl:{chosen_lvl}")
        chosen_iou_list.append(iou_lvl[chosen_lvl])
        chosen_lvl_list.append(chosen_lvl.cpu().numpy())
        
        if visualize_results:
            # save for visulsization
            save_path = image_name / f'chosen_{clip_model.positives[k]}.png'
            vis_mask = mask_lvl[chosen_lvl].cpu().numpy()
            vis_mask_save(vis_mask, save_path)
            save_path = image_name / f'chosen_for_video_{clip_model.positives[k]}.png'
            vis_mask = mask_for_video[chosen_lvl].cpu().numpy()
            vis_mask_save(vis_mask, save_path)

        prompt_iou_lvl_dict[clip_model.positives[k]] = (iou_lvl[chosen_lvl], chosen_lvl.cpu().numpy(), score_lvl.cpu().numpy(), thresh_list)
        mask_dict[clip_model.positives[k]]=mask_lvl[chosen_lvl]
        mask_for_video_dict[clip_model.positives[k]]=[mask_for_video[chosen_lvl]]
            
                    
            
    return chosen_iou_list, chosen_lvl_list, prompt_iou_lvl_dict, mask_dict, mask_for_video_dict

def cal_avg_video_feature(video_model, mask, video_features_dim3, query_embeddings):
    
    h, w = mask.shape
    video_features_dim3_chosen =video_features_dim3[mask==1]
    mask = (mask==1).cpu().numpy()

    video_features = video_model.decode(video_features_dim3_chosen).cpu().detach().numpy()
    flatten_features = video_features.reshape(-1, video_features.shape[-1])
    cos_similarities = cosine_similarity(flatten_features, query_embeddings.reshape(1, -1))
    mean_cos_similarity = cos_similarities.mean()
    return mean_cos_similarity

def drawn_similarity_images(similarity_list,save_path,thresh_hold=0., gt_intervals=None):
    indices = [x[0] for x in similarity_list]
    values = [x[1] for x in similarity_list]
    plt.figure(figsize=(10, 6))  
    ax = plt.gca()
    
    # Calculate y-axis range with padding
    y_min, y_max = min(values) if values else 0, max(values) if values else 1
    y_range = y_max - y_min
    y_min = y_min - 0.05 * y_range if y_range > 0 else y_min - 0.05
    y_max = y_max + 0.05 * y_range if y_range > 0 else y_max + 0.05
    
    # Draw GT intervals as colored background regions
    gt_interval_drawn = False
    if gt_intervals is not None and len(gt_intervals) > 0:
        for interval in gt_intervals:
            start_idx, end_idx = interval[0], interval[1]
            # Clamp intervals to valid range
            start_idx = max(start_idx, min(indices)) if indices else start_idx
            end_idx = min(end_idx, max(indices)) if indices else end_idx
            if start_idx <= end_idx:
                ax.axvspan(start_idx, end_idx, alpha=0.3, color='green', 
                          label='GT Interval' if not gt_interval_drawn else '')
                gt_interval_drawn = True
    
    plt.plot(indices, values, marker='o', linestyle='-', color='b', label='Similarity', markersize=4)
    plt.axhline(y=thresh_hold, color='r', linestyle='--', linewidth=1.5, label=f'Threshold: {thresh_hold:.3f}')
    
    plt.title('Similarity across Different Indices')
    plt.xlabel('Frame Index')
    plt.ylabel('Cosine Similarity')
    plt.ylim(y_min, y_max)
    plt.legend()

    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(save_path), dpi=150, bbox_inches='tight')
    plt.close()


def plot_confusion_matrix(y_true, y_pred, labels=None,title='default',output_path='.'):
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    
    plt.figure(figsize=(6, 4))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=labels, yticklabels=labels)
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title(f'{title} Confusion Matrix')
    plt.savefig(os.path.join(output_path,f"{title}_Confusion_matrix.png"))

def evaluate_video_feature(similarity_list, ground_truth_list, threshhold):
    '''
        similarity_list: [(idx, similarity, iou), ...]
        ground_truth_list: [(l1,r1), (l2,r2), ...]
    '''
    label_list = []
    predict_list = []
    for fm in similarity_list:
        if assert_idx_in_list(fm[0],ground_truth_list):
            label_list.append(True)
        else:
            label_list.append(False)
    # Generate predictions based on the threshold
    for fm in similarity_list:
        if fm[1] >= threshhold:
            predict_list.append(True)
        else:
            predict_list.append(False)

    # Calculate metrics
    true_positive = sum([1 for i, pred in enumerate(predict_list) if pred and label_list[i]])
    false_positive = sum([1 for i, pred in enumerate(predict_list) if pred and not label_list[i]])
    false_negative = sum([1 for i, pred in enumerate(predict_list) if not pred and label_list[i]])
    
    accuracy = sum([1 for i, pred in enumerate(predict_list) if pred == label_list[i]]) / len(predict_list)
    precision = true_positive / (true_positive + false_positive) if (true_positive + false_positive) > 0 else 0
    recall = true_positive / (true_positive + false_negative) if (true_positive + false_negative) > 0 else 0
    
    # Calculate IoU vIou=(sum iou of S_intersection)/|S_union|
    iou_values = []
    for i, (pred, fm) in enumerate(zip(predict_list, similarity_list)):
        if label_list[i] or pred:
            if label_list[i] and pred:
                iou = similarity_list[i][2]
            else:
                iou = 0
            iou_values.append(iou)
    
    avg_iou = sum(iou_values) / len(iou_values) if iou_values else 0

    return {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'average_iou': avg_iou,
        'label_list': label_list,
        'predict_list': predict_list
    }
    

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Evaluation script parameters")
    parser.add_argument("--exp_name", type=str,required=True)
    parser.add_argument("--iterations", type=int,required=True)
    parser.add_argument("--prompts",nargs='+',type=str,default=None)
    parser.add_argument("--output_path",type=str, default="eval_result")
    parser.add_argument("--annotation_folder",type=str, required=True)
    parser.add_argument("--langfeat_mode",choices=['sam','video'],default='sam')
    parser.add_argument("--dataset_type",choices=['hypernerf','neu3d'],default="hypernerf")
    parser.add_argument("--ae_ckpt_path",type=str,required=True)
    parser.add_argument("--video_ae_ckpt_path",type=str,default=None)
    parser.add_argument("--use_gt_feat",type=int,default=0)
    parser.add_argument("--gt_clip_feat_path",type=str,default=None)
    parser.add_argument("--use_gt_clip_feat",type=int,default=0)
    parser.add_argument('--mask_tresh', type=float,default=0.4)
    parser.add_argument('--scale', type=int,default=29)
    parser.add_argument('--chose_mask_strategy', choices=['point','mean'],default="point")
    parser.add_argument('--tag', type=str, default="default")
    parser.add_argument('--feat_dim', type=int, default=3)
    parser.add_argument('--video_feat_dim',type=int,default=6)
    parser.add_argument('--video_feat_dir', type=str,default=None)
    parser.add_argument('--use_gt_video_feat_dim3',type=int,default=0)
    parser.add_argument('--apply_video_search',action='store_true')
    parser.add_argument('--video_encoder_hidden_dims',nargs="+",type=int,default=[2048,1024,512,256,128,64,32,6])
    parser.add_argument('--video_decoder_hidden_dims',nargs="+",type=int,default=[32,64,128,256,512,1024,2048,4096])
    parser.add_argument('--encoder_hidden_dims',nargs="+",type=int,default=[256,128,64,32,3])
    parser.add_argument('--decoder_hidden_dims',nargs="+",type=int,default=[16,32,64,128,256,512])
    parser.add_argument('--video_frame_gt_path',type=str,default=None)
    parser.add_argument('--video_eval_iterations',default=20000,type=int)
    parser.add_argument('--smooth_feature_post',action='store_true')
    parser.add_argument('--smooth_feature_post_frames',default=1,type=int)
    parser.add_argument('--smooth_feature_post_coff',nargs="+",type=float,default=None)
    parser.add_argument('--visualize_results',action='store_true', help='Whether to save visualization results')
    parser.add_argument('--detail_results', action='store_true', help='Whether to save detailed results')
    args = parser.parse_args()
    # import pdb; pdb.set_trace()
    mask_thresh = args.mask_tresh                 
    prompts =args.prompts
    if args.apply_video_search and args.smooth_feature_post_coff == None:
        if args.smooth_feature_post_frames == 1:
            smooth_feature_post_coff = [0.1,0.8,0.1]
        elif args.smooth_feature_post_frames == 2:
            smooth_feature_post_coff = [0.1, 0.2, 0.4, 0.2, 0.1]
    else:
        smooth_feature_post_coff = args.smooth_feature_post_coff
    if args.video_frame_gt_path == None and args.apply_video_search:
        args.video_frame_gt_path = os.path.join(args.annotation_folder,"video_annotations.json")
        assert os.path.exists(args.video_frame_gt_path), f"video_frame_gt_path:{args.video_frame_gt_path} not exists"
    # NOTE logger
    timestamp = time.strftime('%Y%m%d_%H%M%S', time.localtime())
    output_path = os.path.join(args.output_path,args.exp_name,f"{timestamp}-{args.tag}")
    os.makedirs(output_path, exist_ok=True)
    log_file = os.path.join(output_path, f'{timestamp}.log')
    logger = get_logger(args.exp_name, log_file=log_file, log_level=logging.INFO)
    logger.info(args)
    logger.info(f"prompts:{prompts} if prompt is None means check every prompt in datasets")                           

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device:{device}")
    timstamps = []
    if args.langfeat_mode == 'sam':
        logger.info('sam model, level 1 2 3')
        
        if args.use_gt_feat:
            npy_file_name = 'gt_npy'
        else:
            npy_file_name = 'renders_npy'
        logger.info(f"npy_file_name: {npy_file_name}")
        if args.dataset_type == 'hypernerf':
            feat_dir = [os.path.join(os.getenv("ExpsDir","../output"),args.dataset_type,f"{args.exp_name}_{level}",f"video_lang/ours_{args.iterations}/{npy_file_name}") for level in range(1,4)]
        elif args.dataset_type == 'neu3d':
            feat_dir = [os.path.join(os.getenv("ExpsDir","../output"),args.dataset_type,f"{args.exp_name}_{level}",f"test_lang/ours_{args.iterations}/{npy_file_name}") for level in range(1,4)]
        else:
            raise NotImplementedError
    else:
        logger.info('video model, level 0s')
        level=0
        
        if args.use_gt_feat:
            npy_file_name = 'gt_npy'
        else:
            npy_file_name = 'renders_npy'
        logger.info(f"npy_file_name: {npy_file_name}")
        if args.dataset_type == 'hypernerf':
            feat_dir = [os.path.join(os.getenv("ExpsDir","../output"),args.dataset_type,f"{args.exp_name}_{level}",f"video_lang/ours_{args.iterations}/{npy_file_name}") ]
        elif args.dataset_type == 'neu3d':
            feat_dir = [os.path.join(os.getenv("ExpsDir","../output"),args.dataset_type,f"{args.exp_name}_{level}",f"test_lang/ours_{args.iterations}/{npy_file_name}") ]

        else:
            raise NotImplementedError

    if args.apply_video_search:
        assert args.video_feat_dir is not None
        # Load video features
        level = 0
        if args.use_gt_video_feat_dim3 == 1:
            video_npy_file_name = 'gt_npy'
        else:
            video_npy_file_name = 'renders_npy'
        video_feature_dir = os.path.join(os.getenv("ExpsDir","../output"),args.dataset_type,f"{args.video_feat_dir}_{level}",f"video_lang/ours_{args.video_eval_iterations}/{video_npy_file_name}")

        video_feature_list = os.listdir(video_feature_dir)
        video_feature_list.sort()
        video_features = []
        logger.info(f"Loading video features from {video_feature_dir} ({len(video_feature_list)} files)")
        if len(video_feature_list) == 0:
            logger.warning(f"No video feature files found under {video_feature_dir}")
        log_interval = max(1, len(video_feature_list) // 10) if len(video_feature_list) else 1
        for idx, name in enumerate(video_feature_list):
            video_features.append(np.load(os.path.join(video_feature_dir,name)))
            if len(video_feature_list) and (((idx + 1) % log_interval == 0) or (idx + 1 == len(video_feature_list))):
                logger.info(f"Loaded {idx + 1}/{len(video_feature_list)} video feature files")

    else:
        video_features = []

    colormap_options = colormaps.ColormapOptions(
        colormap="turbo",
        normalize=True,
        colormap_min=-1.0,
        colormap_max=1.0,
    )
  

    json_folder = os.path.join(args.annotation_folder,'train')
    
    replace_prompts = {}
    prompts_for_video = []
    video_prompts_to_intervals = {}  # Store prompt -> intervals mapping
    if args.apply_video_search:
        logger.info(f"Loading video frame annotations from {args.video_frame_gt_path}")
        with open(args.video_frame_gt_path) as f:
            gt_frame_dict = json.load(f)
            # Support both old nested structure and new flat structure
            if isinstance(list(gt_frame_dict.values())[0], dict):
                # Old nested structure: {"chicken container": {"closed chicken container": [...], ...}}
                for key in gt_frame_dict.keys():
                    replace_prompts[key] = []
                    for target_prompt in gt_frame_dict[key].keys():
                        replace_prompts[key].append(target_prompt)
                        prompts_for_video.append(target_prompt)
                        video_prompts_to_intervals[target_prompt] = gt_frame_dict[key][target_prompt]
            else:
                # New flat structure: {"closed chicken container": [...], "opened chicken container": [...]}
                for prompt, intervals in gt_frame_dict.items():
                    prompts_for_video.append(prompt)
                    video_prompts_to_intervals[prompt] = intervals
        logger.info(f"Loaded {len(prompts_for_video)} prompts from video annotations")

    logger.info(f"Loading ground-truth annotations from {json_folder}")
    gt_ann, image_shape, image_paths, id2name, name2id, im_id2imidx = eval_gt_lerfdata(Path(json_folder), Path(output_path),prompts,replace_prompts,args.dataset_type)
    
    # Add video prompts directly to gt_ann if they don't exist in COCO annotations
    if args.apply_video_search:
        h, w = image_shape
        for idx in gt_ann.keys():
            img_ann = gt_ann[idx]
            for prompt in prompts_for_video:
                if prompt not in img_ann:
                    # Add empty mask and bbox for prompts not in COCO annotations
                    img_ann[prompt] = {
                        'mask': np.zeros((h, w), dtype=np.uint8),
                        'bboxes': np.array([]).reshape(0, 4)
                    }

    eval_index_list = [int(idx) for idx in list(gt_ann.keys())] # range(1, frame_num+1)
    logger.info(f"Loaded GT for {len(eval_index_list)} frames with image shape {image_shape}")

    if os.getenv("smooth_video_feature_pre",'f')=='t':
        video_smooth_frames = int(os.getenv("video_smooth_frames",2))
    else:
        video_smooth_frames = 0
    
    video_smooth_near_list = [[None for _ in range(2*video_smooth_frames)] for __  in range(len(eval_index_list))]
    logger.info(f"Loading semantic features for {len(eval_index_list)} frames across {len(feat_dir)} level(s)")
    compressed_sem_feats = np.zeros((len(feat_dir), len(eval_index_list), *image_shape, args.feat_dim), dtype=np.float32)
    for i in range(len(feat_dir)):
        feat_paths_lvl = sorted(glob.glob(os.path.join(feat_dir[i], '*.npy')),
                                key=lambda file_name: int(os.path.basename(file_name).split(".npy")[0]))
        logger.info(f"Level {i + 1}/{len(feat_dir)}: found {len(feat_paths_lvl)} npy files under {feat_dir[i]}")
        level_log_interval = max(1, len(eval_index_list) // 10) if len(eval_index_list) else 1
        for j, idx in enumerate(eval_index_list):
            compressed_sem_feats[i][j] = np.load(feat_paths_lvl[im_id2imidx[idx]])  
            if len(eval_index_list) and (((j + 1) % level_log_interval == 0) or (j + 1 == len(eval_index_list))):
                logger.info(f"Level {i + 1}: loaded semantic features for {j + 1}/{len(eval_index_list)} frames")
            
                
                    

    # load e5_model
    if args.apply_video_search:
        logger.info("Initializing e5-mistral-7b-instruct model (SentenceTransformer)")
        cache_roots = [
            "/media/ssd1/users/sooran/.cache",
            "/media/ssd1/users/sooran/.cache/huggingface",
            "/media/ssd1/users/sooran/.cache/huggingface/hub",
        ]
        resolved_model_path = None
        for cache_root in cache_roots:
            local_snapshot_dir = os.path.join(
                cache_root, "models--intfloat--e5-mistral-7b-instruct", "snapshots"
            )
            if not os.path.isdir(local_snapshot_dir):
                continue
            snapshot_candidates = [
                os.path.join(local_snapshot_dir, d)
                for d in os.listdir(local_snapshot_dir)
                if os.path.isdir(os.path.join(local_snapshot_dir, d))
            ]
            if not snapshot_candidates:
                continue
            snapshot_candidates.sort(key=os.path.getmtime, reverse=True)
            for candidate in snapshot_candidates:
                shard_files = glob.glob(os.path.join(candidate, "model*.safetensors"))
                if shard_files:
                    resolved_model_path = candidate
                    logger.info(f"Found local e5 snapshot with weights at {resolved_model_path}")
                    break
                else:
                    logger.warning(
                        f"Snapshot {candidate} missing model*.safetensors shards; skipping."
                    )
            if resolved_model_path is not None:
                break
        if resolved_model_path is not None:
            logger.info("Loading e5 model strictly from local cache (offline mode)")
            e5_model = SentenceTransformer(
                resolved_model_path,
                local_files_only=True,
            )
        else:
            logger.warning(
                "No local snapshot found under any known cache root; falling back to Hugging Face download."
            )
            e5_model = SentenceTransformer(
                "intfloat/e5-mistral-7b-instruct",
                cache_folder="/media/ssd1/users/sooran/.cache",
            )
        e5_model.max_seq_length = 4096
        name2name_e5_embeddings = {}
        # Use prompts_for_video directly (works for both old nested and new flat structure)
        unique_prompt_names = sorted(set(prompts_for_video))
        logger.info(f"Encoding {len(unique_prompt_names)} prompts with e5 model")
        if len(unique_prompt_names) == 0:
            logger.warning("No prompts found for e5 encoding; check video annotations.")
        prompt_log_interval = max(1, len(unique_prompt_names) // 10) if len(unique_prompt_names) else 1
        for idx, prompt_name in enumerate(unique_prompt_names):
            name2name_e5_embeddings[prompt_name] = e5_model.encode(prompt_name, prompt_name="summarization_query")
            if len(unique_prompt_names) and (((idx + 1) % prompt_log_interval == 0) or (idx + 1 == len(unique_prompt_names))):
                logger.info(f"Encoded {idx + 1}/{len(unique_prompt_names)} prompts")
        import gc
        del e5_model
        gc.collect()
    else:
        name2name_e5_embeddings = None

    clip_model = OpenCLIPNetwork(device)
    checkpoint = torch.load(args.ae_ckpt_path, map_location=device)
    
    # Infer architecture from checkpoint if mismatch detected
    def infer_architecture_from_checkpoint(checkpoint, feature_dim=512):
        """Infer encoder and decoder dimensions from checkpoint keys"""
        encoder_dims = []
        decoder_dims = []
        
        # Extract all Linear layer weights from encoder
        encoder_linear_weights = {}
        for key in checkpoint.keys():
            if key.startswith('encoder.') and key.endswith('.weight'):
                idx = int(key.split('.')[1])
                encoder_linear_weights[idx] = checkpoint[key]
        
        # Sort by index and extract dimensions
        sorted_indices = sorted(encoder_linear_weights.keys())
        for idx in sorted_indices:
            weight = encoder_linear_weights[idx]
            # Skip if not 2D (should be [out_dim, in_dim] for Linear layer)
            if len(weight.shape) != 2:
                continue
            out_dim, in_dim = weight.shape
            if idx == 0:
                # First layer: feature_dim -> out_dim
                encoder_dims.append(out_dim)
            else:
                # Subsequent layers: should match previous output
                if in_dim == encoder_dims[-1]:
                    encoder_dims.append(out_dim)
                else:
                    # Mismatch, but continue (might be due to BatchNorm in between)
                    if len(encoder_dims) > 0:
                        encoder_dims.append(out_dim)
        
        # Extract all Linear layer weights from decoder
        decoder_linear_weights = {}
        for key in checkpoint.keys():
            if key.startswith('decoder.') and key.endswith('.weight'):
                idx = int(key.split('.')[1])
                decoder_linear_weights[idx] = checkpoint[key]
        
        # Sort by index and extract dimensions
        sorted_indices = sorted(decoder_linear_weights.keys())
        for idx in sorted_indices:
            weight = decoder_linear_weights[idx]
            # Skip if not 2D (should be [out_dim, in_dim] for Linear layer)
            if len(weight.shape) != 2:
                continue
            out_dim, in_dim = weight.shape
            if idx == 0:
                # First layer: encoder_dims[-1] -> out_dim
                decoder_dims.append(out_dim)
            else:
                # Subsequent layers: should match previous output
                if in_dim == decoder_dims[-1]:
                    decoder_dims.append(out_dim)
        
        return encoder_dims, decoder_dims
    
    # Try to load with provided architecture first
    try:
        model = Autoencoder(args.encoder_hidden_dims, args.decoder_hidden_dims).to(device)
        model.load_state_dict(checkpoint, strict=True)
        logger.info(f"Successfully loaded checkpoint with provided architecture")
    except RuntimeError as e:
        logger.warning(f"Architecture mismatch detected. Inferring architecture from checkpoint...")
        # Infer architecture from checkpoint
        inferred_encoder_dims, inferred_decoder_dims = infer_architecture_from_checkpoint(checkpoint, feature_dim=512)
        logger.info(f"Inferred encoder_dims: {inferred_encoder_dims}")
        logger.info(f"Inferred decoder_dims: {inferred_decoder_dims}")
        
        # Create model with inferred architecture
        model = Autoencoder(inferred_encoder_dims, inferred_decoder_dims).to(device)
        model.load_state_dict(checkpoint, strict=True)
        logger.info(f"Successfully loaded checkpoint with inferred architecture")
    
    model.eval()
    # load video autoencoder
    if args.apply_video_search:
        assert args.video_ae_ckpt_path is not None
        # Try to load video autoencoder with provided architecture first
        try:
            video_model = Autoencoder(args.video_encoder_hidden_dims,args.video_decoder_hidden_dims,feature_dim=4096).to(device)
            video_checkpoint = torch.load(args.video_ae_ckpt_path,map_location=device)
            video_model.load_state_dict(video_checkpoint, strict=True)
            logger.info(f"Successfully loaded video checkpoint with provided architecture")
        except RuntimeError as e:
            logger.warning(f"Video architecture mismatch detected. Inferring architecture from checkpoint...")
            video_checkpoint = torch.load(args.video_ae_ckpt_path,map_location=device)
            # Infer architecture from checkpoint
            inferred_encoder_dims, inferred_decoder_dims = infer_architecture_from_checkpoint(video_checkpoint, feature_dim=4096)
            logger.info(f"Inferred video encoder_dims: {inferred_encoder_dims}")
            logger.info(f"Inferred video decoder_dims: {inferred_decoder_dims}")
            
            # Create model with inferred architecture
            video_model = Autoencoder(inferred_encoder_dims, inferred_decoder_dims, feature_dim=4096).to(device)
            video_model.load_state_dict(video_checkpoint, strict=True)
            logger.info(f"Successfully loaded video checkpoint with inferred architecture")
        
        video_model.eval()
    else:
        video_model = None

    chosen_iou_all, chosen_lvl_list = [], []
    prompt_iou_all_dict = {}

    for j, idx in enumerate(tqdm(eval_index_list)):
        interval_eval = os.getenv('interval',None)
        if interval_eval is not None:
            print('interval for evaluation:', interval_eval)
            if j % int(interval_eval) !=0:
                continue
        image_name = Path(output_path) / f'{idx+1:0>5}'
        if args.visualize_results:
            image_name.mkdir(exist_ok=True, parents=True)
        
        sem_feat = compressed_sem_feats[:, j, ...]
        
        sem_feat = torch.from_numpy(sem_feat).float().to(device)

                
        if sem_feat.min()>0:
            sem_feat = sem_feat*2.0-1 #! scale back to (-1,1)
        
        rgb_img = cv2.imread(image_paths[j])[..., ::-1]
        rgb_img = (rgb_img / 255.0).astype(np.float32)
        rgb_img = torch.from_numpy(rgb_img).to(device)

        with torch.no_grad():
            lvl, h, w, _ = sem_feat.shape
            if os.getenv("use_vae",'f')=='t':
                restored_feat = model.decode(sem_feat.flatten(0, 2))
            else:
                restored_feat = model.decode(sem_feat.flatten(0, 2))

            if args.use_gt_clip_feat:
                assert args.gt_clip_feat_path is not None
                logger.info(f"Extract gt features from gt_clip_feat_path:{args.gt_clip_feat_path}")
                
                seg_map = torch.from_numpy(np.load(os.path.join(args.gt_clip_feat_path,f"cam00-{image_paths[j].split('/')[-1].split('_')[0]}_s.npy")))
                feature_map = torch.from_numpy(np.load(os.path.join(args.gt_clip_feat_path,f"cam00-{image_paths[j].split('/')[-1].split('_')[0]}_f.npy")))
                y, x = torch.meshgrid(torch.arange(0, image_shape[0]), torch.arange(0, image_shape[1]), indexing='ij')
                x = x.reshape(-1, 1)
                y = y.reshape(-1, 1)
                seg = seg_map[:, y, x].squeeze(-1).long()
                mask = seg != -1

                point_feature1 = feature_map[seg[1:2]].squeeze(0).reshape(image_shape[0], image_shape[1], -1)
                point_feature2 = feature_map[seg[2:3]].squeeze(0).reshape(image_shape[0], image_shape[1], -1)
                point_feature3 = feature_map[seg[3:4]].squeeze(0).reshape(image_shape[0], image_shape[1], -1)

                point_feature = torch.stack([point_feature1, point_feature2, point_feature3], dim=0)  # (3, h, w, c)

                # point_feature = point_feature.permute(0, 3, 1, 2)  # (3, c, h, w)
                restored_feat = point_feature.to(device)
            else:
                restored_feat = restored_feat.view(lvl, h, w, -1)           # 3x832x1264x512

        img_ann = gt_ann[f'{idx}']

        clip_model.set_positives(list(img_ann.keys()))
        c_iou_list, c_lvl, prompt_iou_lvl_dict, chosen_mask_dict, chosen_mask_for_video_dict = activate_stream(restored_feat, rgb_img, clip_model, image_name,
                                            img_ann=img_ann, thresh=mask_thresh, colormap_options=colormap_options, name2id=name2id,scale=args.scale,chose_mask_strategy=args.chose_mask_strategy, imageid=j, visualize_results=args.visualize_results)


        for key, (iou, lvl, lvl_all, tresh_all) in prompt_iou_lvl_dict.items():
            if key not in prompt_iou_all_dict:
                prompt_iou_all_dict[key] = []
            if args.apply_video_search and key in prompts_for_video:
                video_feature_dim3 = torch.from_numpy(video_features[im_id2imidx[idx]]).float().to(device)
                video_features_sim = cal_avg_video_feature(video_model,chosen_mask_for_video_dict[key][0],video_feature_dim3,name2name_e5_embeddings[key])
                video_features_sim_list = []
                smoothed_video_features_sim = video_features_sim
            else:
                video_features_sim = 0
                smoothed_video_features_sim = 0
            prompt_iou_all_dict[key].append((idx,iou,lvl,lvl_all, tresh_all, smoothed_video_features_sim))

        chosen_iou_all.extend(c_iou_list)
        chosen_lvl_list.extend(c_lvl)

        
    result_data = []

    for key in prompt_iou_all_dict.keys():
        if key in prompts_for_video:
            continue
        format_data =[key]
        mean_iou_key = sum([fm[1].item() for fm in prompt_iou_all_dict[key]])/ len(prompt_iou_all_dict[key])
        format_data.append(mean_iou_key)
        format_data.append([fm[2] for fm in prompt_iou_all_dict[key]])  # Lvls
        format_data.append([fm[3] for fm in prompt_iou_all_dict[key]])  # Similarity
        format_data.append([fm[4] for fm in prompt_iou_all_dict[key]])  # Thresh
        format_data.append([fm[5] for fm in prompt_iou_all_dict[key]])  # Video features Similarity

        for idx in  eval_index_list:
            exist_prompt_this_frame = False
            for fm in  prompt_iou_all_dict[key]:
                if fm[0] == idx:
                    format_data.append(fm[1])
                    exist_prompt_this_frame = True
            if exist_prompt_this_frame == False:
                format_data.append("NA")

        result_data.append(format_data)
        
        logger.info(f"key:{key}, mean_iou:{mean_iou_key}")
    logger.info(f"Mean IoU: {sum([fm[1] for fm in result_data])/len(result_data)}")

    if args.detail_results:
        with open(os.path.join(output_path,'time-agnostic_results.csv'), mode='w', newline='', encoding='utf-8') as file:
            writer = csv.writer(file)
            header_list = ['Prompt', 'Mean IoU', 'Lvls', 'Similarity','Tresh',"Video feature Similarity"]
            for i in eval_index_list:
                header_list.append(f'frame_{i+1}_iou')
            writer.writerow(header_list)
            for data in result_data:
                writer.writerow(data)


    #### draw video feature similarity pic ####
    def assert_idx_in_list(idx, gt_list):
        for interval in gt_list:
            if idx>=interval[0] and idx<= interval[1]:
                return True
        return False


   
    if args.apply_video_search:
        with open(args.video_frame_gt_path) as f:
            gt_frame_dict = json.load(f)
        video_res_list = []
        clip_res_list = []
        for key in prompts_for_video:
            
            video_similarity = [(im_id2imidx[fm[0]], fm[5], fm[1]) for fm in prompt_iou_all_dict[key]]
            sorted_video_similarity = sorted(video_similarity, key=lambda x: x[0])
            if args.smooth_feature_post:
                assert args.smooth_feature_post_frames*2 + 1 == len(smooth_feature_post_coff)
                assert sum(smooth_feature_post_coff) - 1 < 1e-9
                video_smooth_frames = args.smooth_feature_post_frames
                smoothed_video_similarity = []
                for i in range(len(sorted_video_similarity)):
                    smooth_res = 0
                    
                    for j in range(-video_smooth_frames,video_smooth_frames+1):
                        if i+j >=0 and i+j < len(sorted_video_similarity):
                            smooth_res+= sorted_video_similarity[i+j][1] * smooth_feature_post_coff[j+video_smooth_frames]
                        else:
                            smooth_res = sorted_video_similarity[i][1]
                            break
                    smoothed_video_similarity.append(smooth_res)
                
                for i in range(len(sorted_video_similarity)):
                    sorted_video_similarity[i] = (sorted_video_similarity[i][0],smoothed_video_similarity[i], sorted_video_similarity[i][2])
                
            video_thresh = sum(tup[1] for tup in video_similarity) / len(video_similarity)
            
            clip_similarity = [(im_id2imidx[fm[0]], fm[3][fm[2]], fm[1]) for fm in prompt_iou_all_dict[key]]
            sorted_clip_similarity = sorted(clip_similarity, key=lambda x: x[0])
            if args.smooth_feature_post:
                video_smooth_frames = args.smooth_feature_post_frames
                smoothed_clip_similarity = []
                for i in range(len(sorted_clip_similarity)):
                    smooth_res = 0
                    for j in range(-video_smooth_frames,video_smooth_frames+1):
                        if i+j >=0 and i+j < len(sorted_clip_similarity):
                            smooth_res+= sorted_clip_similarity[i+j][1] * smooth_feature_post_coff[j+video_smooth_frames]
                        else:
                            smooth_res = sorted_clip_similarity[i][1]
                            break
                    smoothed_clip_similarity.append(smooth_res)
                
                for i in range(len(sorted_clip_similarity)):
                    sorted_clip_similarity[i] = (sorted_clip_similarity[i][0],smoothed_clip_similarity[i], sorted_clip_similarity[i][2])
                
            clip_thresh = sum(tup[1] for tup in sorted_clip_similarity) / len(sorted_clip_similarity)
            
            # Get GT intervals for this key from video_annotations.json
            # GT intervals should be in npy index format (same as similarity_list indices)
            gt_intervals_for_key = None
            if key in video_prompts_to_intervals:
                gt_intervals_raw = video_prompts_to_intervals[key]
                if gt_intervals_raw and len(gt_intervals_raw) > 0:
                    # GT intervals from video_annotations.json are already in npy index format
                    # (same format as used in evaluate_video_feature)
                    gt_intervals_for_key = [(interval[0], interval[1]) for interval in gt_intervals_raw]
            
            drawn_similarity_images(sorted_video_similarity, os.path.join(output_path,f"{key}_video_feat_sim.png"),thresh_hold=video_thresh, gt_intervals=gt_intervals_for_key)
            drawn_similarity_images(sorted_clip_similarity, os.path.join(output_path,f"{key}_clip_feat_sim.png"),thresh_hold=clip_thresh, gt_intervals=gt_intervals_for_key)

            # Evaluate video feature if GT intervals exist for this prompt
            if key in video_prompts_to_intervals:
                gt_intervals = video_prompts_to_intervals[key]
                if gt_intervals and len(gt_intervals) > 0:
                    # Check if this prompt has valid GT mask (from COCO annotations)
                    # If not, vIoU cannot be calculated meaningfully
                    has_gt_mask = False
                    for idx in eval_index_list:
                        if f'{idx}' in gt_ann and key in gt_ann[f'{idx}']:
                            mask = gt_ann[f'{idx}'][key]['mask']
                            if mask is not None and np.sum(mask) > 0:
                                has_gt_mask = True
                                break
                    
                    video_res = evaluate_video_feature(sorted_video_similarity, gt_intervals, threshhold=video_thresh)
                    clip_res = evaluate_video_feature(sorted_clip_similarity, gt_intervals, threshhold=clip_thresh)
                    
                    if has_gt_mask:
                        logger.info(f"Key: {key}. Video Feature: vIoU:{video_res['average_iou']}, Accuracy:{video_res['accuracy']}. Clip Feature: vIoU:{clip_res['average_iou']}, Accuracy:{clip_res['accuracy']}")
                        video_res_list.append((video_res['average_iou'],video_res['accuracy']))
                        clip_res_list.append((clip_res['average_iou'],clip_res['accuracy']))
                    else:
                        logger.info(f"Key: {key}. Video Feature: vIoU:N/A (no GT mask), Accuracy:{video_res['accuracy']}. Clip Feature: vIoU:N/A (no GT mask), Accuracy:{clip_res['accuracy']}")
                        video_res_list.append((None, video_res['accuracy']))
                        clip_res_list.append((None, clip_res['accuracy']))
                    
                    if args.detail_results:
                        plot_confusion_matrix(video_res['label_list'],video_res['predict_list'],[False,True], f'video-{key}',output_path)
                        plot_confusion_matrix(clip_res['label_list'],clip_res['predict_list'],[False,True], f'clip-{key}',output_path)
            if args.detail_results:
                with open(os.path.join(output_path,f'video-query-results-{key}.csv'), mode='w', newline='', encoding='utf-8') as file:
                    writer = csv.writer(file)
                    header_list = ['Type']
                    for fm in sorted_clip_similarity:
                        header_list.append(f'frame_{fm[0]}_iou')
                    writer.writerow(header_list)
                    writer.writerow(["clip similarity"]+[fm[1] for fm in sorted_clip_similarity])
                    writer.writerow(["clip meaniou"]+[fm[2] for fm in sorted_clip_similarity])
                    writer.writerow(["video similarity"]+[fm[1] for fm in sorted_video_similarity])
                    writer.writerow(["video meaniou"]+[fm[2] for fm in sorted_video_similarity])

        # Calculate average vIoU and accuracy, handling None values for vIoU
        video_vious = [fm[0] for fm in video_res_list if fm[0] is not None]
        clip_vious = [fm[0] for fm in clip_res_list if fm[0] is not None]
        avg_video_viou = sum(video_vious) / len(video_vious) if video_vious else None
        avg_clip_viou = sum(clip_vious) / len(clip_vious) if clip_vious else None
        
        logger.info(f"Video: Average vIoU: {avg_video_viou if avg_video_viou is not None else 'N/A (no GT masks)'}, Average Accuracy: {sum([fm[1] for fm in video_res_list])/len(video_res_list)}")
        logger.info(f"Clip: Average vIoU: {avg_clip_viou if avg_clip_viou is not None else 'N/A (no GT masks)'}, Average Accuracy: {sum([fm[1] for fm in clip_res_list])/len(clip_res_list)}")
