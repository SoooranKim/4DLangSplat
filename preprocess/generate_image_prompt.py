import numpy as np
# from PIL import Image
from tqdm import tqdm
from PIL import Image, ImageFilter, ImageDraw
import os
import imageio
import argparse


def process_frames(unique_ids, num_frames, mask_dir, image_dir, output_dir, begin_idx=1, end_str="png"):
    """
    According to the maximum cropsize, process each frame and save the cropped image.
    """
    for i in tqdm(range(begin_idx, num_frames + begin_idx)):
        image_path = f'{image_dir}/{i:06}.{end_str}'
        mask_path = f'{mask_dir}/{i:06}.npy'
        image = Image.open(image_path).convert('RGBA')
        mask = np.load(mask_path)

        for mask_id in unique_ids:
            clear_image = image.copy()

            blurred_image = image.filter(ImageFilter.GaussianBlur(radius=10))
            image_bw = blurred_image.convert("L").convert("RGBA")

            mask_for_id = (mask == mask_id)

            if not np.any(mask_for_id):
                continue

            final_image = Image.composite(clear_image, image_bw, Image.fromarray((mask_for_id * 255).astype(np.uint8)))

            boundary = np.argwhere(mask_for_id)
            if boundary.size > 0:
                y_coords, x_coords = np.where(mask_for_id)
                min_x, max_x = x_coords.min(), x_coords.max()
                min_y, max_y = y_coords.min(), y_coords.max()

                draw = ImageDraw.Draw(final_image)

                point_radius = 2

                for y in range(min_y, max_y + 1):
                    for x in range(min_x, max_x + 1):
                        # print(mask.shape)
                        if mask_for_id[y, x] and (x == 0 or y == 0 or x == mask.shape[1] - 1 or y == mask.shape[0] - 1 or not mask_for_id[y - 1, x] or not mask_for_id[y + 1, x] or
                                                  not mask_for_id[y, x - 1] or not mask_for_id[y, x + 1]):
                            draw.ellipse((x - point_radius, y - point_radius, x + point_radius, y + point_radius), outline="red", width=2)

            os.makedirs(f"{output_dir}/{mask_id:02}", exist_ok=True)
            output_path = f'{output_dir}/{mask_id:02}/{i:06}.png'
            final_image.save(output_path)


def pic2video(input_dir, output_path):
    image_list = os.listdir(input_dir)
    image_list.sort()
    images = [Image.open(os.path.join(input_dir, name)) for name in image_list]
    imageio.mimwrite(output_path, images, fps=30)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--mask_dir", type=str)
    parser.add_argument("--image_dir", type=str)
    parser.add_argument("--output_dir", type=str, default='./prompt_images')
    parser.add_argument("--begin_idx", type=int, default=1)
    parser.add_argument("--end_str", type=str, default="png")
    args = parser.parse_args()
    mask_dir = args.mask_dir
    image_dir = args.image_dir
    output_dir = args.output_dir
    num_frames = len(os.listdir(image_dir))

    unique_ids = set()

    for i in range(args.begin_idx, num_frames + args.begin_idx):
        mask_path = f'{mask_dir}/{i:06}.npy'
        mask = np.load(mask_path)
        unique_ids.update(np.unique(mask))

    process_frames(unique_ids, num_frames, mask_dir, image_dir, output_dir, args.begin_idx, args.end_str)

    for i in range(max(unique_ids)):
        pic2video(f"{output_dir}/{i:02}", f"{output_dir}/{i:02}.mp4")


