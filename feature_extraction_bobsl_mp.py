import sys
import os
import time
import argparse
from pathlib import Path
import multiprocessing

import cv2
import lmdb
import numpy as np
from tqdm import tqdm

import torch
import torchvision

sys.path.insert(0, "../")
from lightning import ModelModule
from datamodule.transforms import VideoTransform

# Global device; this will be recomputed in each worker if needed.
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
    def __init__(self, args, ckpt_path, detector="retinaface", worker_id=0):
        super(InferencePipeline, self).__init__()
        self.worker_id = worker_id  # Store worker id for progress bar position
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

        # Use the worker_id to set the position and description of the tqdm progress bar
        for i in tqdm(range(zero_frame, num_frames, num_frames_per_batch),
                      position=self.worker_id, desc=f"Worker {self.worker_id}"):
            start_frame = i
            end_frame = min(i + num_frames_per_batch, num_frames)

            start_time = time.time()
            if self.args.lmdb:
                video = self.load_video_lmdb(data_filename, start_frame, end_frame, fps)
            else:
                video = self.load_video(data_filename, start_frame, end_frame, fps)
            used_time = time.time() - start_time
            # print(f"load video time: {used_time:.4f} seconds")

            start_time = time.time()
            processed_windows = []
            for j in range(0, video.shape[0], window_size):
                window = video[j:j+window_size]  # Extract a window of frames
                try:
                    landmarks = self.landmarks_detector(window)  # Detect landmarks for this window
                    window = self.video_process(window, landmarks)  # Apply video processing
                    window = torch.tensor(window, device=device)  # Convert to tensor
                    window = window.permute((0, 3, 1, 2))  # Change dimensions to (N, C, H, W)
                    window = self.video_transform(window)  # Apply final transformations
                    processed_windows.append(window)  # Store processed window
                except (AssertionError, OverflowError) as e:
                    print(f'WARNING: preprocessing {data_filename} frame {start_frame + j} raises error ({e}), filling with 0s...')
                    processed_windows.append(torch.zeros(window.shape[0], 1, 88, 88, device=device))
            video = torch.cat(processed_windows, dim=0)
            used_time = time.time() - start_time
            # print(f"process video time: {used_time:.4f} seconds")

            start_time = time.time()
            pad_size = (batch_size * window_size) - video.shape[0]
            if pad_size > 0:
                padding = torch.zeros((pad_size, *video.shape[1:]), device=device)
                video = torch.cat([video, padding], dim=0)

            input_tensor = video.view(batch_size, window_size, *video.shape[1:])
            
            with torch.no_grad():
                if batch_size == 1:
                    text, feat = self.modelmodule(input_tensor.squeeze(0))
                    feat = feat.unsqueeze(0)
                    texts.append(text)
                else:
                    feat = self.modelmodule.forward_encoder(input_tensor)
                feat = feat.view(-1, *feat.shape[2:])
                feat = feat[:video.shape[0] - pad_size]
                    
                feats.append(feat)
            used_time = time.time() - start_time
            # print(f"inference video time: {used_time:.4f} seconds")
            
        feats = torch.cat(feats)
        assert feats.shape[0] == (num_frames - zero_frame), 'feature and video should have same number of frames.'
        
        return texts, feats

       

def partition_list(lst, n):
    """
    Partition the list lst into n chunks (some chunks may be empty).
    """
    if n <= 0:
        return []
    chunks = []
    chunk_size = len(lst) // n
    remainder = len(lst) % n
    start = 0
    for i in range(n):
        end = start + chunk_size + (1 if i < remainder else 0)
        chunks.append(lst[start:end])
        start = end
    return chunks


def process_worker(video_files, args, worker_id):
    # Set the GPU for this worker (assumes number of GPUs equals number_workers)
    os.environ["CUDA_VISIBLE_DEVICES"] = str(worker_id)
    global device
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"Worker {worker_id} using device {device}")

    # Pass worker_id to the pipeline
    pipeline = InferencePipeline(args, args.model_path, detector="mediapipe", worker_id=worker_id)
    
    for video_file in video_files:
        save_path = os.path.join(args.save_dir, f"{get_base_filename(video_file)}.npy")
        if not args.overwrite and os.path.exists(save_path):
            print(f"Worker {worker_id}: Skipping {get_base_filename(video_file)}, feature file already exists: {save_path}")
            continue

        print(f"Worker {worker_id} processing {video_file} ...")
        transcripts, feats = pipeline(video_file, args.window_size, args.batch_size)

        print(f"Worker {worker_id} transcript for {video_file}: {transcripts}")
        print(f"Worker {worker_id} feature shape: {feats.shape}")

        np.save(save_path, feats.cpu().numpy())


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
    parser.add_argument("--number_workers", type=int, default=1, help="Number of parallel workers to process videos concurrently")
    args = parser.parse_args()

    # Ensure the save directory exists
    os.makedirs(args.save_dir, exist_ok=True)

    # Generate list of video files
    if args.video_path:
        video_files = [args.video_path]
    else:
        video_files = [os.path.join(args.video_dir, f) for f in os.listdir(args.video_dir) if f.endswith(".mp4")]
        video_files = [video_file for video_file in video_files if not os.path.exists(os.path.join(args.save_dir, f"{get_base_filename(video_file)}.npy"))]
        video_files = video_files[:15]

    print(f'Found {len(video_files)} videos.')

    if args.number_workers > 1:
        # Partition the video files into args.number_workers chunks
        video_chunks = partition_list(video_files, args.number_workers)
        processes = []
        for worker_id, chunk in enumerate(video_chunks):
            # It's possible some chunks are empty
            if not chunk:
                continue
            p = multiprocessing.Process(target=process_worker, args=(chunk, args, worker_id))
            p.start()
            processes.append(p)
        for p in processes:
            p.join()
    else:
        # Single-worker processing (use global device as already defined)
        pipeline = InferencePipeline(args, args.model_path, detector="mediapipe")
        for video_file in video_files:
            save_path = os.path.join(args.save_dir, f"{get_base_filename(video_file)}.npy")
            if not args.overwrite and os.path.exists(save_path):
                print(f"Skipping {get_base_filename(video_file)}, feature file already exists: {save_path}")
                continue

            print(f'Processing {video_file} ...')
            transcripts, feats = pipeline(video_file, args.window_size, args.batch_size)

            print(f"Transcript for {video_file}: {transcripts}")
            print(feats.shape)

            np.save(save_path, feats.cpu().numpy())
