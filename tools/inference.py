# reference from poseidon
import torch
import cv2
import numpy as np
from torchvision import transforms
from posetimation.zoo.DSTA.dsta_std_resnet50 import DSTA_STD_ResNet50 
from engine.defaults import default_parse_args
from posetimation import get_cfg, update_config 

from ultralytics import YOLO
import yaml

# Indices of keypoints used in PoseTrack (excluding 'left_eye' and 'right_eye')
used_keypoint_indices = [0,2,5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16]

used_keypoint_colors = [
    (255, 0, 0),      # Crimson Red
    (0, 255, 0),      # Lime Green
    (0, 0, 255),      # Royal Blue
    (255, 255, 0),    # Sunny Yellow
    (0, 255, 255),    # Aqua Cyan
    (255, 0, 255),    # Magenta Pink
    (192, 192, 192),  # Silver Grey
    (128, 0, 128),    # Plum Purple
    (255, 165, 0),    # Tangerine Orange
    (128, 128, 0),    # Olive Green
    (0, 128, 128),    # Teal
    (75, 0, 130),     # Indigo
    (255, 105, 180),  # Hot Pink
    (0, 191, 255),    # Deep Sky Blue
    (255, 223, 0),    # Golden Yellow
    (165, 42, 42),    # Chocolate Brown
    (34, 139, 34)     # Forest Green
]

def preprocess_frame(frame, cfg):
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    frame = cv2.resize(frame, tuple(cfg.MODEL.IMAGE_SIZE))
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    frame = transform(frame)
    return frame

def run_inference(model, frames, device, cfg):
    with torch.no_grad():
        input_tensor = frames.to(device, non_blocking=True)
        output = model(input_tensor)
    return output  # Output is the heatmaps

def load_model(cfg, checkpoint_path, device):
    model = DSTA_STD_ResNet50(cfg, device=device)
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device)  # Move the model to the device
    model.eval()
    return model

def process_keypoints(predicts, used_keypoint_indices):
    keypoints = predicts['pred_jts']

    # **Filter the keypoints**
    keypoints = keypoints[:, used_keypoint_indices, :, :]

    return keypoints

def process_video(video_path, model, detector, device, cfg, window_size=5, step_frame=1):
    cap = cv2.VideoCapture(video_path)
    frame_buffer = []
    results = []

    # Initialize video writer
    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    out = cv2.VideoWriter('output_video.mp4', cv2.VideoWriter_fourcc(*'mp4v'), fps, (frame_width, frame_height))

    total_frames = cap.get(cv2.CAP_PROP_FRAME_COUNT)
    print("Number of frames: ", total_frames)

    i = 0
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame_buffer.append(frame)

        print("Frame: {}/{}".format(i, total_frames))
        i += 1
        
        if len(frame_buffer) == window_size * step_frame:
            # Select frames from buffer based on step_frame
            sampled_frames = frame_buffer[::step_frame]

            # Determine the index of the central frame
            central_frame_index = len(sampled_frames) // 2
            central_frame = sampled_frames[central_frame_index]
            
            # Run detector on the central frame
            detections = detector.predict(central_frame, verbose=False)
            
            detection = detections[0]
            
            # Extract bounding boxes
            boxes = detection.boxes  # Boxes object
            
            for box in boxes:
                x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                
                # Enlarge bbox by 25%
                w = x2 - x1
                h = y2 - y1
                x_center = x1 + w / 2
                y_center = y1 + h / 2
                w *= 1.25
                h *= 1.25
                x1 = x_center - w / 2
                y1 = y_center - h / 2
                x2 = x_center + w / 2
                y2 = y_center + h / 2
                
                # Ensure coordinates are within image boundaries
                x1_crop = int(max(x1, 0))
                y1_crop = int(max(y1, 0))
                x2_crop = int(min(x2, central_frame.shape[1]))
                y2_crop = int(min(y2, central_frame.shape[0]))

                # Apply the same bbox to all sampled frames
                cropped_frames = []
                for frame in sampled_frames:
                    cropped_frame = frame[y1_crop:y2_crop, x1_crop:x2_crop]
                    processed_frame = preprocess_frame(cropped_frame, cfg)
                    cropped_frames.append(processed_frame)
                
                # Stack the sampled frames
                cropped_frames = torch.stack(cropped_frames, dim=0)
                cropped_frames = cropped_frames.unsqueeze(0)

                # Run inference on the preprocessed frames
                output = run_inference(model, cropped_frames, device, cfg)

                # Extract keypoints
                keypoints = process_keypoints(output, used_keypoint_indices)

                keypoints = keypoints[0]  # shape: [num_keypoints, 2]

                # Map keypoints to original image coordinate space
                keypoints[:, 0] += x1_crop
                keypoints[:, 1] += y1_crop

                # Convert keypoints to numpy
                keypoints = keypoints.cpu().numpy()

                # Draw keypoints on central_frame
                for j, (x, y) in enumerate(keypoints):
                    # set color based on keypoint index
                    color = used_keypoint_colors[used_keypoint_indices[j]]
                    cv2.circle(central_frame, (int(x), int(y)), 3, color, -1)
            
            # Write central_frame to video
            out.write(central_frame)
            
            # Remove the first `step_frame` frames from the buffer
            frame_buffer = frame_buffer[step_frame:]
    
    cap.release()
    out.release()
    return results

# Load your configuration
def load_config(config_path):
    """ Load the YAML configuration file.

    Args:
        config_path (str): Path to the YAML configuration file.

    Returns:
        dict: Configuration dictionary.
    """
    with open(config_path, 'r') as file:
        config = yaml.safe_load(file)
    return config

def setup(args):
    cfg = get_cfg(args)
    update_config(cfg, args)
    return cfg

args = default_parse_args()
cfg = setup(args)
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

model_path = "./best_model.pt"

# Load the model
model = load_model(cfg, model_path, device)

# define detector
detector = YOLO('./models/yolo/yolov8s-pose.pt')

# Run inference and visualization on the video
video_path = './sample.mp4'

results = process_video(video_path, model, detector, device, cfg, window_size=cfg.WINDOWS_SIZE)