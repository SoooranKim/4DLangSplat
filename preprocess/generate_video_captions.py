from transformers import Qwen2VLForConditionalGeneration, AutoTokenizer, AutoProcessor
from qwen_vl_utils import process_vision_info
import os
import sys
from pathlib import Path
import csv
from tqdm import tqdm
import argparse
from loguru import logger
from sentence_transformers import SentenceTransformer
import torch
import numpy as np

# allow importing utils from project root when running from preprocess/
ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from utils.cache_utils import get_hf_cache_dir
# from vllm import LLM, SamplingParams

# default: Load the model on the available device(s), reusing local cache if present
_HF_CACHE_DIR = get_hf_cache_dir()

model = Qwen2VLForConditionalGeneration.from_pretrained(
    "Qwen/Qwen2-VL-7B-Instruct",
    torch_dtype="auto",
    device_map="auto",
    cache_dir=_HF_CACHE_DIR,
)

processor = AutoProcessor.from_pretrained(
    "Qwen/Qwen2-VL-7B-Instruct",
    cache_dir=_HF_CACHE_DIR,
)


def video_caption_generate(video_path, prompt=None, nframes=8):
    
    messages = [
        {
            "role": "user",
            "content": [
                {
                    "type": "video",
                    "video": video_path,
                    "nframes":nframes
                },
                {
                    "type": "text", 
                    "text": prompt
                }
            ]
        }
    ]
    text = processor.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    image_inputs, video_inputs = process_vision_info(messages)
    inputs = processor(
        text=[text],
        images=image_inputs,
        videos=video_inputs,
        padding=True,
        return_tensors="pt",
    )
    inputs = inputs.to("cuda")

    generated_ids = model.generate(**inputs, max_new_tokens=128)
    
    generated_ids_trimmed = [
        out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
    ]
    output_text = processor.batch_decode(
        generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
    )
    return output_text

def image_caption_generate(image_base_path,idx,video_prompt, num_frames):
    if idx - 3 < 1 or idx + 3 > num_frames:
        frame_list = [idx]
    else:
        frame_list = [idx-3, idx, idx + 3]
    

    
    context_image_list = []
    for id in frame_list:
        if os.path.exists(os.path.join(image_base_path,f"{id:06}.png")):
            context_image_list.append(
                {
                    "type": "image",
                    "image": os.path.join(image_base_path,f"{id:06}.png")
                }
            )

    messages = [
        {
            'role':'user',
            'content':
                context_image_list + [
                    {
                        'type': 'text',
                        'text': f"You have an understanding of the overall transformation process of the object: '{video_prompt}'. \
                            Now, I have provided you with images extracted from this process. Please describe the specific state of the object(s) in the given image, without referring to the entire video process. \
                                Avoid describing states that you can't infer directly from the picture. Avoid repeating descriptions in context. For example, if the context suggests the object is moving up and down but the image shows it is just moving down, explicitly only state that the object is in a moving down state. If the context suggests the object is breaking but the image shows it is complete right now, explicitly only state that the object appears to be complete and intact. If context tells you something changes from green to blue, but it's blue in this image, just state that the object is blue. "
                        
                    }
                ]
        },
    ]
    text = processor.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True, add_vision_id=True
    )

    image_inputs, video_inputs = process_vision_info(messages)
    inputs = processor(
        text=[text],
        images=image_inputs,
        videos=video_inputs,
        padding=True,
        return_tensors="pt",
    )
    inputs = inputs.to("cuda")

    # Inference
    generated_ids = model.generate(**inputs, max_new_tokens=128)
    generated_ids_trimmed = [
        out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
    ]
    output_text = processor.batch_decode(
        generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
    )
    return output_text

def chose_best_captions(video_caption_list):
    

    # from transformers import BertTokenizer, BertModel
    import numpy as np
    from sklearn.metrics.pairwise import cosine_similarity

    
    model = SentenceTransformer(
        "intfloat/e5-mistral-7b-instruct",
        cache_folder=_HF_CACHE_DIR,
    )
    model.max_seq_length = 4096
    def get_embedding(text):
        
        document_embeddings = model.encode(text)
        return document_embeddings

   
    caption_embeddings = [get_embedding(caption[1]) for caption in video_caption_list]
    similarities = cosine_similarity(caption_embeddings)
    average_similarities = similarities.mean(axis=1)
    best_caption_idx = np.argmax(average_similarities)

    del model
    torch.cuda.empty_cache()

    return video_caption_list[best_caption_idx]


VIDEO_PROMPT = "I highlighted the objects I want you to describe in red outline and blurred the objects that don't need you to describe. First please determine the object highlighted in red line in the video. Then briefly summarize the transformation process of this object."
if __name__ =="__main__":
    parser = argparse.ArgumentParser(description='A simple argument parser example.')
    parser.add_argument("--output_base",type=str, required=True)
    parser.add_argument('--video_file', type=str, default="/datadrive/zhourenping/output", help='video input path')
    parser.add_argument('--video_prompt', type=str, default=VIDEO_PROMPT, help='video prompt input')
    parser.add_argument('--segmentation_dir', type=str, default="/datadrive/zhourenping/origin_mask_large", help='segmentation_dir')
    parser.add_argument('--start_frame', type=int, default=10, help='start_frame')
    parser.add_argument('--frame_interval', type=int, default=1, help='frame_interval')
    parser.add_argument('--end_frame', type=int, default=22, help='end_frame')
    parser.add_argument("--mode", choices=['video', 'image', 'feature'])
    parser.add_argument("--specific_id", type=int,nargs="+")
    parser.add_argument('--output_features_dir',type=str,default='features')
    parser.add_argument('--caption_dir',type=str,default=None)
    parser.add_argument('--fps',type=int,default=38)
    # Parse the arguments
    args = parser.parse_args()

    logger.add(os.path.join(args.output_base,"logger.log"), rotation="1 MB")  # Automatically rotate log files after reaching 1 MB
    logger.info(args)
    output_file = os.path.join(args.output_base,"output")
    os.makedirs(output_file,exist_ok=True)
    num_frames = len(os.listdir(args.segmentation_dir))
    logger.info(f"num_frames:{num_frames}")
    if args.mode == 'video':
        print("Generate the video caption.")
        max_obj_id = 0
        for file in os.listdir(args.video_file):
            if ".mp4" in file:
                max_obj_id = max(max_obj_id,int(file.split(".")[0]))
        print(f"max_obj_id:{max_obj_id}")
        video_caption_list = []
        for obj_id in range(1,max_obj_id+1):
            if args.specific_id is not None and obj_id not in args.specific_id:
                continue
            if len(os.listdir(f"{args.video_file}/{obj_id:02}"))<20:
                continue
            video_captions = []
            n_frame = min(int(round(num_frames/args.fps)),18)
            logger.info(f"n_frame:{n_frame}")

            video_caption = video_caption_generate(f"{args.video_file}/{obj_id:02}.mp4",prompt=args.video_prompt, nframes=n_frame)
            logger.info(f"obj_id:{obj_id}, n_frame={n_frame}, caption={video_caption}")
            video_captions.append((n_frame, video_caption[0]))
            video_caption_list.append((obj_id,video_captions))



        with open(os.path.join(output_file,f"output_video_description.csv"), mode='w', newline='', encoding='utf-8') as file:
            writer = csv.writer(file)
            writer.writerow(['obj_id', "n_frames", 'video_description'])
            for obj_id,video_captions in video_caption_list:
                for n_frames, video_caption in video_captions:
                    processed_caption = video_caption.replace('\r', '').replace('\n', '\\n')
                    writer.writerow([obj_id,n_frames,processed_caption])

                    logger.info(f"obj_id:{obj_id}, video_caption:{video_caption}, n_frames:{n_frames}")

    # exit()
    elif args.mode == 'image':
        print("Generate the image caption.")

        with open(os.path.join(output_file,f"output_video_description.csv"), mode='r', encoding='utf-8') as input_file:
            reader = csv.reader(input_file)
            header = next(reader)
            # print(f"Header: {header}")
            
            description_dict = {}
            for row in reader:

                print(f"Object ID: {row[0]}, n_frames: {row[1]}, Video Description: {row[2]}")
                if row[0] not in description_dict.keys():
                    description_dict[row[0]] = []
                
                description_dict[row[0]].append((row[1],row[2]))
        for obj_id, description_list in description_dict.items():
            print(description_list)
            obj_id = int(obj_id)
            if args.specific_id is not None and obj_id not in args.specific_id:
                continue
            output_csv_path = os.path.join(output_file,f"output_text_id{obj_id}.csv")
            with open(output_csv_path, mode='w', newline='', encoding='utf-8') as file:
                writer = csv.writer(file)
                writer.writerow(['frame',"n_frame", 'output_text'])
                output_text_dict = {}
                for (n_frames, description) in description_list:
                    image_base_path = f"{args.video_file}/{obj_id:02}"
                    for now_id in tqdm(range(1, num_frames+1,1)):
                        if not os.path.exists(os.path.join(image_base_path,f"{now_id:06}.png")):
                            continue
                        description = description.replace('\\n', '\n')
                        output_text = image_caption_generate(image_base_path, now_id, description, num_frames=num_frames)
                        # print(output_text)
                        if now_id not in output_text_dict:
                            output_text_dict[now_id] = []
                        output_text_dict[now_id].append((n_frames,output_text))
                for now_id, nframes_description_list in output_text_dict.items():
                    for (nframes,description )in nframes_description_list:
                        writer.writerow([os.path.join(image_base_path,f"{now_id:06}.png"), nframes, description])

    else:
        raise ValueError("mode should be video or image or feature")