import os
import shutil
import numpy as np
import torch
import matplotlib.pyplot as plt
from PIL import Image
import json
from sam2.build_sam import build_sam2_video_predictor


debug = True   # Set to True to enable debug mode, which will display the frames and annotations
display_frames = False   # Set to True during debug mode, which will display the frames and annotations

# Initialize device and model
def initialize_sam():
    # select the device for computation
    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")
    # device = torch.device("cpu")
    print(f"using device: {device}")

    if device.type == "cuda":
        torch.autocast("cuda", dtype=torch.bfloat16).__enter__()
        if torch.cuda.get_device_properties(0).major >= 8:
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True
    elif device.type == "mps":
        print("\nSupport for MPS devices is preliminary. SAM 2 is trained with CUDA and might "
              "give numerically different outputs and sometimes degraded performance on MPS. "
              "See e.g. https://github.com/pytorch/pytorch/issues/84936 for a discussion.")

    # sam2_checkpoint = "../checkpoints/sam2.1_hiera_large.pt"  # /Users/kjr/Python/SegmentAnything/sam2/checkpoints
    sam2_checkpoint = "python-server/checkpoints/sam2.1_hiera_large.pt"  # For deployment in docker
    model_cfg = "configs/sam2.1/sam2.1_hiera_l.yaml"    # /Users/kjr/Python/SegmentAnything/sam2
    # sam2_checkpoint = "../checkpoints/sam2.1_hiera_small.pt"
    # model_cfg = "configs/sam2.1/sam2.1_hiera_s.yaml"
    
    predictor = build_sam2_video_predictor(model_cfg, sam2_checkpoint, device=device)
    return predictor




def show_mask(mask, ax, obj_id=None, random_color=False):
    if random_color:
        color = np.concatenate([np.random.random(3), np.array([0.6])], axis=0)
    else:
        cmap = plt.get_cmap("tab10")
        cmap_idx = 0 if obj_id is None else obj_id
        color = np.array([*cmap(cmap_idx)[:3], 0.6])
    h, w = mask.shape[-2:]
    mask_image = mask.reshape(h, w, 1) * color.reshape(1, 1, -1)
    ax.imshow(mask_image)


def show_points(coords, labels, ax, marker_size=200):
    pos_points = coords[labels==1]
    neg_points = coords[labels==0]
    ax.scatter(pos_points[:, 0], pos_points[:, 1], color='green', marker='*', s=marker_size, edgecolor='white', linewidth=1.25)
    ax.scatter(neg_points[:, 0], neg_points[:, 1], color='red', marker='*', s=marker_size, edgecolor='white', linewidth=1.25)


def show_box(box, ax):
    x0, y0 = box[0], box[1]
    w, h = box[2] - box[0], box[3] - box[1]
    ax.add_patch(plt.Rectangle((x0, y0), w, h, edgecolor='orange', facecolor=(0, 0, 0, 0), lw=1))


def find_bounding_box(mask):
    # Get the row, col indices of all True (non-zero) pixels
    coords = np.argwhere(mask[0])
    
    # If the mask is empty, coords will be empty.
    # Otherwise, compute bounding box:
    if coords.size > 0:
        # coords[:, 0] are the y indices (rows), coords[:, 1] are the x indices (columns)
        y_min, x_min = coords.min(axis=0)
        y_max, x_max = coords.max(axis=0)
    
        box = np.array([int(x_min), int(y_min), int(x_max), int(y_max)], dtype=np.float32)
        # print("Bounding Box:", box)
        # box = np.array([1494, 349, 1857, 548], dtype=np.float32)
        # show_box(box, ax)
        return box
    else:
        print("Mask is empty, no bounding box found.")
        return None

def show_bounding_box(mask, ax):
    box = find_bounding_box(mask)
    if box is not None:
        show_box(box, ax)

def get_annotation_frame_box(frame_annotation, frame_width, frame_height):
    x_min = int(frame_annotation['x1'] / frame_annotation['width'] * frame_width)
    y_min = int(frame_annotation['y1'] / frame_annotation['height'] * frame_height)
    x_max = int(frame_annotation['x2'] / frame_annotation['width'] * frame_width)
    y_max = int(frame_annotation['y2'] / frame_annotation['height'] * frame_height)
    return [x_min, y_min, x_max, y_max] 




def delete_image_file(frame_path, number):
    # Format the number with leading zeros to make it 5 digits
    filename = f"{number:05d}.jpg"
    # Construct the full file path
    file_path = os.path.join(frame_path, filename)
    
    # Delete the file
    os.remove(file_path)
    print(f"Deleted file: {file_path}")




def bounding_box_to_annotation(box, width, height, frame_annotation):
    [x1, y1, x2, y2] = box

    # Will provide a new annotation based on initial annotation, but with new location of bounding box
    new_annotation = {
        "x1": int(x1 / width * frame_annotation["width"]),
        "y1": int(y1 / height * frame_annotation["height"]),
        "x2": int(x2 / width * frame_annotation["width"]),
        "y2": int(y2 / height * frame_annotation["height"]),
        "width": frame_annotation["width"],
        "height": frame_annotation["height"],
        "objectid": frame_annotation["objectid"],
        "type": frame_annotation["type"],
        "tags": frame_annotation["tags"],
        "sam2": True,                                    # Indicates generated by SAM2
        "interpolated": False
    }
    
    return new_annotation



# Generate the frames from the video using ffmpeg
def generate_video_frames_using_ffmeg(src_video_file, video_fps, frame_start, frame_end, dst_frame_path):
    # Determine source video
    # src_video_path = os.path.join(gva_src, data_src, video_folder)
    # src_video_file = os.path.join(src_video_path, f"{video_file}.{video_extn}")

    # Check if it's a file
    if os.path.isfile(src_video_file):
        print(f"File '{src_video_file}' exists and is a file.")
    else:        
        # Raise an exception if the file does not exist
        raise FileNotFoundError(f"File '{src_video_file}' does not exist or is not a file.")

    # Determine target folder for frame generation
    # dst_frame_path = os.path.join(tmp_folder, video_folder, f"{video_file}_{frame_start}_{frame_end}")
    # Create the directory if it doesn't exist
    os.makedirs(dst_frame_path, exist_ok=True)

    # Setup ffmpeg command parameters
    start_time = f"{(frame_start) / video_fps}"
    num_frames = f"{frame_end - frame_start + 1}"      # We add second +1 to account for ffmpeg -vframe repeats first frame
    frame_rate = '"fps={}"'.format(video_fps)
    start_number = f"{frame_start}"                    # We -1 to account for ffmpeg -vframe repeats first frame
    # select_pattern = f"\"select=\'between(n,{frame_start},{frame_end})\'\""
    output_pattern = f"{dst_frame_path}/%05d.jpg"

    # ffmpeg -vstart option has a side effect that first frame is duplicated, and frames are then one behind
    # We patch this by labelling output one frame earlier, then will delete the first frame
    
    # cmd_arr = ['ffmpeg', '-i', src_video_file, '-vf', select_pattern, '-fps_mode', 'passthrough', '-q:v', '2', '-start_number', f"{frame_start}", output_pattern]
    cmd_arr = ['ffmpeg', '-ss', start_time, '-i', src_video_file, '-q:v', '2', '-start_number', start_number, '-vf', frame_rate, '-vframes', num_frames, output_pattern]
    ffmpeg_cmd = ' '.join(cmd_arr)
    print('Execute ffmpeg: ', ffmpeg_cmd)

    # Run ffmpeg
    ffmpeg_exit_code = os.system(ffmpeg_cmd)
    print('Exit Code ffmpeg:', ffmpeg_exit_code)
    if ffmpeg_exit_code != 0:
        raise RuntimeError(f"FFmpeg command failed with exit code {ffmpeg_exit_code}")

    # Delete the first frame as duplicate from -vframe option
    delete_image_file(dst_frame_path, int(start_number))
    




# Will apply SAM2 to annotate all frames given first frame annotation
def get_annotations_from_video_frames(video_dir, direction, frame_annotation, vis_frame_stride = 5):
    # `video_dir` a directory of JPEG frames with filenames like `<frame_index>.jpg`
    print(f"Use SAM2 to annotate the video frames in {video_dir}")

        
    # Initialize the predictor globally
    predictor = initialize_sam() 

    # scan all the JPEG frame names in this directory
    frame_names = [
        p for p in os.listdir(video_dir)
        if os.path.splitext(p)[-1] in [".jpg", ".jpeg", ".JPG", ".JPEG"]
    ]
    frame_names.sort(key=lambda p: int(os.path.splitext(p)[0]))

    # take a look the first video frame
    frame_idx = 0
    
    # Get the height and width of first frame
    with Image.open(os.path.join(video_dir, frame_names[frame_idx])) as img:
        width, height = img.size  # img.size returns a tuple (width, height)
        print(f"Frame {frame_idx} - Width: {width}px, Height: {height}px")

    # Transform frame_annotation to initial bounding box
    init_box = get_annotation_frame_box(frame_annotation, width, height)
    
    # Initialise the state of the predictor
    inference_state = predictor.init_state(video_path=video_dir)

    # Now annotate first frame
    ann_frame_idx = 0 if (direction == 'forward') else len(frame_names) - 1  # the frame index we interact with
    ann_obj_id = 0  # give a unique id to each object we interact with (it can be any integers)
    
    # Let's add a box at (x_min, y_min, x_max, y_max) e.g. (300, 0, 500, 400) to get started
    box = np.array(init_box, dtype=np.float32)
    
    # Generate the following frames from the first frame using SAM2
    print(f"Generating mask from the index frame ({ann_frame_idx}) using SAM2")
    _, out_obj_ids, out_mask_logits = predictor.add_new_points_or_box(
        inference_state=inference_state,
        frame_idx=ann_frame_idx,
        obj_id=ann_obj_id,
        box=box,
    )

    # Register the first frame value
    first_frame_num = int(frame_names[0].split('.')[0])

    # show the results on the current (interacted) frame
    if debug & display_frames:
        plt.figure(figsize=(12, 9))
        plt.title(f"frame {ann_frame_idx}")
        plt.imshow(Image.open(os.path.join(video_dir, frame_names[ann_frame_idx])))
        show_mask((out_mask_logits[0] > 0.0).cpu().numpy(), plt.gca(), obj_id=out_obj_ids[0])
        show_bounding_box((out_mask_logits[0] > 0.0).cpu().numpy(), plt.gca())

    # run propagation throughout the video and collect the results in a dict
    annotations = {}

    reverse = False if (direction == 'forward') else True

    for out_frame_idx, out_obj_ids, out_mask_logits in predictor.propagate_in_video(
        inference_state,
        reverse=reverse,
    ):

        out_mask = (out_mask_logits[0] > 0.0).cpu().numpy()
        
        frame_num = int(frame_names[out_frame_idx].split('.')[0])
        frame_count = frame_num - first_frame_num
        box = find_bounding_box(out_mask)

        # Display mask over base image
        if debug & display_frames & (frame_count % vis_frame_stride == 0):
            plt.figure(figsize=(12, 9))
            plt.title(f"frame {frame_num} ({frame_count})")
            plt.imshow(Image.open(os.path.join(video_dir, frame_names[out_frame_idx])))
            show_mask(out_mask, plt.gca(), obj_id=out_obj_ids[0])
            show_bounding_box(out_mask, plt.gca())

        if box is not None:
            int_box = box.astype(np.int32).tolist()
            annotation = bounding_box_to_annotation(int_box, width, height, frame_annotation)
            if debug & (frame_count % vis_frame_stride == 0):
                print('Frame: ', frame_num)
                print('Box: ', int_box)
                print('Annotation: ', annotation)
            annotations[frame_num] = annotation

    # Garbage collect the predictor
    predictor = None

    # Write the annotations dictionary to a JSON file
    if debug:
        output_json_path = os.path.join(video_dir, 'annotations.json')
        try:
            with open(output_json_path, 'w') as json_file:
                json.dump(annotations, json_file, indent=4)
            print(f"Annotations successfully written to {output_json_path}")

        except IOError as e:
            raise RuntimeError(f"Failed to write annotations to {output_json_path}: {e}")
        
    return annotations

    




# Main processing functions
def process_video(data_src, video_folder, video_file, video_extn, video_fps, 
                 frame_start, frame_end, direction, frame_annotation, gva_src, tmp_folder):
    """Main function to process video and generate annotations"""

    # Generate the frames from the video using ffmpeg
    src_video_file = os.path.join(gva_src, data_src, video_folder, f"{video_file}.{video_extn}")
    dst_frame_path = os.path.join(tmp_folder, video_folder, f"{video_file}_{frame_start}_{frame_end}")
    generate_video_frames_using_ffmeg(src_video_file, video_fps, frame_start, frame_end, dst_frame_path)

    # Get the annotations from the video frames
    annotations = get_annotations_from_video_frames(dst_frame_path, direction, frame_annotation)
    # annotations = {}

    # Cleanup by removing the dst_frame_path folder
    if not debug:
        shutil.rmtree(dst_frame_path)

    return annotations 

