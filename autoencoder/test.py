import os
import numpy as np
import torch
import argparse
import shutil
from torch.utils.data import DataLoader
from tqdm import tqdm
from dataset import Autoencoder_dataset
from model import Autoencoder,VanillaVAE

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset_path', type=str, required=True)
    parser.add_argument('--model_name', type=str, required=True)
    parser.add_argument('--encoder_dims',
                    nargs = '+',
                    type=int,
                    default=[256, 128, 64, 32, 3],
                    )
    parser.add_argument('--decoder_dims',
                    nargs = '+',
                    type=int,
                    default=[16, 32, 64, 128, 256, 256, 512],
                    )
    parser.add_argument('--hidden_dims',
                    type=int,
                    default=3
                    )
    parser.add_argument('--feature_dims',
                    type=int,
                    default=512
                    )
    # parser.add_argument("--output_dim", type=int, default=3)
    parser.add_argument('--num_workers',type=int,default=0)
    parser.add_argument('--language_name', type = str, default = None)
    parser.add_argument('--output_name',type=str,default=None)
    parser.add_argument('--ckpt_path', type=str, default=None)
    parser.add_argument('--device', type=str, default='cuda:0')
    args = parser.parse_args()
    
    model_name = args.model_name
    encoder_hidden_dims = args.encoder_dims
    # encoder_hidden_dims[-1] = args.output_dim
    decoder_hidden_dims = args.decoder_dims
    dataset_path = args.dataset_path
    if args.ckpt_path is not None:
        ckpt_path = args.ckpt_path
    else:
        ckpt_path = f"ckpt/{model_name}/best_ckpt.pth"

    # data_dir = f"{dataset_path}/language_features"
    if args.language_name is None:
        data_dir = f"{dataset_path}/language_features"  
        output_dir = f"{dataset_path}/language_features_dim{encoder_hidden_dims[-1]}"
    else:
        data_dir = os.path.join(dataset_path, args.language_name)
        if args.output_name is not None:
            output_dir = os.path.join(dataset_path,f"{args.language_name}-{args.output_name}_dim{encoder_hidden_dims[-1]}")
        else:    
            output_dir = os.path.join(dataset_path,f"{args.language_name}-language_features_dim{encoder_hidden_dims[-1]}")
    os.makedirs(output_dir, exist_ok=True)
    
    # copy the segmentation map
    from tqdm import tqdm
    for filename in tqdm(os.listdir(data_dir)):
        if filename.endswith("_s.npy"):
            source_path = os.path.join(data_dir, filename)
            target_path = os.path.join(output_dir, filename)
            shutil.copy(source_path, target_path)


    checkpoint = torch.load(ckpt_path)
    train_dataset = Autoencoder_dataset(data_dir)

    print(f"Creating DataLoader with num_workers={args.num_workers}")
    test_loader = DataLoader(
        dataset=train_dataset, 
        batch_size=256,
        shuffle=False, 
        num_workers=args.num_workers, 
        drop_last=False,
        pin_memory=False,
        persistent_workers=False  # num_workers=0일 때는 항상 False
    )
    print(f"DataLoader created successfully with num_workers={args.num_workers}")

    device = args.device
    if os.getenv("use_vae",'f') == 't':
        model = VanillaVAE(encoder_hidden_dims, decoder_hidden_dims, latent_dim=9).to(device)
    else:
        model = Autoencoder(encoder_hidden_dims, decoder_hidden_dims, feature_dim=args.feature_dims).to(device)

    model.load_state_dict(checkpoint)
    model.eval()

    for idx, feature in tqdm(enumerate(test_loader)):
        data = feature.to(device)
        data = data.to(torch.float32)
        
        with torch.no_grad():
            if os.getenv("use_vae",'f') == 't':
                mu, log_var  = model.encode(data)
                outputs = model.reparameterize(mu, log_var).to('cpu').numpy()  

            else:
                outputs = model.encode(data).to("cpu").numpy()  
        if idx == 0:
            features = outputs
        else:
            features = np.concatenate([features, outputs], axis=0)

    os.makedirs(output_dir, exist_ok=True)
    start = 0
    
    for k,v in train_dataset.data_dic.items():
        path = os.path.join(output_dir, k)
        np.save(path, features[start:start+v])
        start += v
