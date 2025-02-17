import sys
import os
import time
import argparse
from pathlib import Path

import cv2
import lmdb
import numpy as np
from tqdm import tqdm

import torch
import torchvision

sys.path.insert(0, "../")
from lightning import ModelModule
from datamodule.transforms import VideoTransform


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def get_video_frame_count_and_fps(video_file):
    # Open the video file
    cap = cv2.VideoCapture(video_file)
    
    # Check if the video was opened successfully
    if not cap.isOpened():
        print("Error: Could not open the video file.")
        return None, None
    else:
        # Get the total number of frames and FPS
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = cap.get(cv2.CAP_PROP_FPS)
    
    # Release the video capture object
    cap.release()
    
    return total_frames, fps


def get_base_filename(data_filename):
    return os.path.splitext(os.path.basename(data_filename))[0]


def lmdb_key_list(episode_name, begin_frame, end_frame):
    return [f"{Path(episode_name.split('.')[0])}/{frame_idx + 1:07d}.jpg".encode('ascii') \
            for frame_idx in range(begin_frame, end_frame)]


def get_rgb_frames(lmdb_env, lmdb_keys):
    frames = []
    # for key in tqdm(lmdb_keys):
    for key in lmdb_keys:
        with lmdb_env.begin() as txn:
            frame = txn.get(key)
        try:
            frame = cv2.imdecode(
                np.frombuffer(frame, dtype=np.uint8),
                cv2.IMREAD_COLOR,
            )
        except:
            break
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frames.append(frame)
    return frames


class InferencePipeline(torch.nn.Module):
    def __init__(self, args, ckpt_path, detector="retinaface"):
        super(InferencePipeline, self).__init__()
        if detector == "mediapipe":
            print('using mediapipe detector')
            from preparation.detectors.mediapipe.detector import LandmarksDetector
            from preparation.detectors.mediapipe.video_process import VideoProcess
            self.landmarks_detector = LandmarksDetector()
            self.video_process = VideoProcess(convert_gray=False)
        elif detector == "retinaface":
            print('using retinaface detector')
            from preparation.detectors.retinaface.detector import LandmarksDetector
            from preparation.detectors.retinaface.video_process import VideoProcess
            self.landmarks_detector = LandmarksDetector(device="cuda:0")
            self.video_process = VideoProcess(convert_gray=False)
        self.video_transform = VideoTransform(subset="test")

        ckpt = torch.load(ckpt_path, map_location=device)
        self.modelmodule = ModelModule(args).to(device)
        self.modelmodule.model.load_state_dict(ckpt)
        self.modelmodule.eval()

        self.args = args

        self.rgb_lmdb_env = None
        if os.path.exists(args.rgb_lmdb_file):
            self.rgb_lmdb_env = lmdb.open(args.rgb_lmdb_file, readonly=True, lock=False, max_readers=512)


    def load_video(self, data_filename, start_frame=0, end_frame=None, fps=25):
        start_time = start_frame / fps
        end_time = None if end_frame is None else end_frame / fps

        video, _, _ = torchvision.io.read_video(data_filename, start_pts=start_time, end_pts=end_time, pts_unit="sec")
        
        # Ensure exact frame count by slicing
        expected_frame_count = end_frame - start_frame if end_frame else video.shape[0]
        video = video[:expected_frame_count]  # Trim extra frames if necessary

        return video.numpy()

    
    def load_video_lmdb(self, data_filename, start_frame=0, end_frame=None, fps=25):
        lmdb_keys = lmdb_key_list(get_base_filename(data_filename), start_frame, end_frame)
        frames = get_rgb_frames(self.rgb_lmdb_env, lmdb_keys)

        return np.stack(frames, axis=0)


    def forward(self, data_filename, window_size=200, batch_size=32):
        data_filename = os.path.abspath(data_filename)
        assert os.path.isfile(data_filename), f"data_filename: {data_filename} does not exist."

        zero_frame = 0
        num_frames, fps = get_video_frame_count_and_fps(data_filename)
        num_frames_per_batch = window_size * batch_size
        feats = []
        texts = []

        # zero_frame = 57000
        # num_frames = 57500

        for i in tqdm(range(zero_frame, num_frames, num_frames_per_batch)):
            start_frame = i
            end_frame = min(i + num_frames_per_batch, num_frames)

            # # Benchmark load_video
            # start_time = time.time()
            # video = self.load_video(data_filename, start_frame, end_frame, fps)
            # load_video_time = time.time() - start_time
            # print(f"load_video time: {load_video_time:.4f} seconds")
            # print(video.shape)
            
            # # Benchmark load_video_lmdb
            # start_time = time.time()
            # video = self.load_video_lmdb(data_filename, start_frame, end_frame, fps)
            # load_video_lmdb_time = time.time() - start_time
            # print(f"load_video_lmdb time: {load_video_lmdb_time:.4f} seconds")
            # print(video.shape)

            # exit()

            if self.args.lmdb:
                video = self.load_video_lmdb(data_filename, start_frame, end_frame, fps)
            else:
                video = self.load_video(data_filename, start_frame, end_frame, fps)

            # preprocess video frames by a batch
            # landmarks = self.landmarks_detector(video)
            # video = self.video_process(video, landmarks)
            # video = torch.tensor(video, device=device)
            # video = video.permute((0, 3, 1, 2))
            # video = self.video_transform(video)

            # preprocess video frames window by window
            processed_windows = []
            for i in range(0, video.shape[0], window_size):
                window = video[i:i+window_size]  # Extract a window of frames
                try:
                    landmarks = self.landmarks_detector(window)  # Detect landmarks for this window
                    window = self.video_process(window, landmarks)  # Apply video processing
                    window = torch.tensor(window, device=device)  # Convert to tensor
                    window = window.permute((0, 3, 1, 2))  # Change dimensions to (N, C, H, W)
                    window = self.video_transform(window)  # Apply final transformations
                    processed_windows.append(window)  # Store processed window
                except (AssertionError, OverflowError) as e:
                    print(f'WARNING: preprocessing frame {start_frame + i} raises error ({e}), filling with 0s...')
                    processed_windows.append(torch.zeros(window_size, 1, 88, 88, device=device))
            # Concatenate all processed windows back to get final output tensor
            video = torch.cat(processed_windows, dim=0)

            pad_size = (batch_size * window_size) - video.shape[0]
            if pad_size > 0:
                padding = torch.zeros((pad_size, *video.shape[1:]), device=device)
                video = torch.cat([video, padding], dim=0)

            input_tensor = video.view(batch_size, window_size, *video.shape[1:])
            # print(input_tensor.shape)

            with torch.no_grad():
                # FIXME: decoding only supports batch_size=1
                if batch_size == 1:
                    text, feat = self.modelmodule(input_tensor.squeeze(0))
                    feat = feat.unsqueeze(0)
                    texts.append(text)
                else:
                    feat = self.modelmodule.forward_encoder(input_tensor)
                feat = feat.view(-1, *feat.shape[2:])  # Merge batch_size and window_size back
                feat = feat[:video.shape[0] - pad_size]  # Remove padded frames
                    
                feats.append(feat)

                # print(feat.shape)
                # print(feat[0])
                # exit()
            
        feats = torch.cat(feats)
        assert feats.shape[0] == (num_frames - zero_frame), 'feature and video should have same number of frames.'
        
        return texts, feats


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str, default="/athenahomes/zifan/auto_avsr/vsr_trlrs3_base.pth", help="Path to the model checkpoint")
    parser.add_argument("--modality", type=str, default="video", help="Modality of the input data (default: video)")
    parser.add_argument("--video_dir", type=str, default="/scratch/shared/beegfs/zifan/bobsl/original_videos/", help="Directory containing input videos")
    parser.add_argument("--video_path", type=str, default=None, help="Path to a single video file")
    parser.add_argument('--lmdb', action='store_true', help='Whether to read frames from lmdb')
    parser.add_argument("--rgb_lmdb_file", type=str, default="/users/zifan/BOBSL/derivatives/video_features/video_frames/lmdb/")
    parser.add_argument("--window_size", type=int, default=200, help="Number of frames per window")
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size for processing")
    parser.add_argument("--save_dir", type=str, default="/scratch/shared/beegfs/zifan/bobsl/video_features/auto_asvr", help="Directory to save extracted features")
    parser.add_argument("--overwrite", action='store_true', help="Overwrite existing feature files if set")
    args = parser.parse_args()

    # Ensure the save directory exists
    os.makedirs(args.save_dir, exist_ok=True)

    # VSR Inference
    pipeline = InferencePipeline(args, args.model_path, detector="mediapipe")

    if args.video_path:
        video_files = [args.video_path]
    else:
        video_files = [os.path.join(args.video_dir, f) for f in os.listdir(args.video_dir) if f.endswith(".mp4")]

    for video_file in video_files:
        save_path = os.path.join(args.save_dir, f"{get_base_filename(video_file)}.npy")

        # Check if the feature file exists and skip processing if overwrite is False
        if not args.overwrite and os.path.exists(save_path):
            print(f"Skipping {get_base_filename(video_file)}, feature file already exists: {save_path}")
            continue

        print(f'Processing {video_file} ...')
        transcripts, feats = pipeline(video_file, args.window_size, args.batch_size)

        print(f"Transcript for {video_file}: {transcripts}")
        print(feats.shape)

        # Save the extracted features to the specified directory
        np.save(save_path, feats.cpu().numpy())