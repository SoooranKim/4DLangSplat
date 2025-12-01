dataset_path=../data/hypernerf/
image_dir=${dataset_path}/rgb/2x
output_dir=preprocess_result/keyboard
video_features_name=video_features
segmentation_dir=../submodules/4d-langsplat-tracking-anything-with-deva/output/keyboard/large/origin_mask_large
cd preprocess
python generate_image_prompt.py \
    --mask_dir ${segmentation_dir} \
    --image_dir ${image_dir} \
    --output_dir ${output_dir} \
    --end_str png 

python generate_video_captions.py \
    --output_base ${output_dir} \
    --video_file ${output_dir} \
    --segmentation_dir ${segmentation_dir} \
    --mode video

python generate_video_captions.py \
    --output_base ${output_dir} \
    --video_file ${output_dir} \
    --segmentation_dir ${segmentation_dir}  \
    --mode image 

python generate_video_features.py \
    --caption_dir ${output_dir}/output \
    --segmentation_dir ${segmentation_dir} 

rm -rf ${dataset_path}/${video_features_name}
cp -r ${output_dir}/output/final_features ${dataset_path}/${video_features_name}