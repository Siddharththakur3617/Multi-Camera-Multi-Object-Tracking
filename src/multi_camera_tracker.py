import cv2
import torch
import torch.nn.functional as F
import numpy as np
from ultralytics import YOLO
import torchreid
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
import threading
import random
from .config import Config
from .logger import setup_logging
from .track import Track
from .byte_tracker import BYTETracker

class TimeoutError(Exception):
    pass

def with_timeout(timeout_seconds):
    def decorator(func):
        def wrapper(*args, **kwargs):
            result = [None]
            exception = [None]
            event = threading.Event()

            def target():
                try:
                    result[0] = func(*args, **kwargs)
                except Exception as e:
                    exception[0] = e
                finally:
                    event.set()

            thread = threading.Thread(target=target)
            thread.daemon = True
            thread.start()
            if not event.wait(timeout_seconds):
                raise TimeoutError(f"Function {func.__name__} timed out after {timeout_seconds} seconds")
            if exception[0] is not None:
                raise exception[0]
            return result[0]
        return wrapper
    return decorator

class MultiCameraTracker:
    def __init__(self, config):
        self.config = config
        self.logger = setup_logging(config.output_dir)
        try:
            cuda_available = torch.cuda.is_available()
            self.device = torch.device("cuda" if cuda_available else "cpu")
            self.logger.info(f"CUDA available: {cuda_available}")
            if cuda_available:
                self.logger.info(f"GPU detected: {torch.cuda.get_device_name(0)}, CUDA version: {torch.version.cuda}")
                self.logger.info(f"Initial GPU memory allocated: {torch.cuda.memory_allocated(0)/1e6:.2f} MB")
            else:
                self.logger.warning("No GPU detected, falling back to CPU")
        except Exception as e:
            self.logger.error(f"Error detecting device: {e}")
            self.device = torch.device("cpu")
        self.logger.info(f"Using device: {self.device}")
        self.class_names = config.class_names
        self.color_map = config.color_map
        self.global_tracks = defaultdict(list)
        self.global_id_counter = 0
        self.gid_color = {}
        self.global_gid_centroids = {}
        self._initialize_models()
        self._initialize_trackers()
        self.track_buffers = [[] for _ in range(len(config.camera_streams))]

    def _initialize_models(self):
        try:
            self.yolo = YOLO(self.config.yolo_model_path).to(self.device)
            self.reid = torchreid.models.build_model(name=self.config.reid_model, num_classes=1000, pretrained=True).to(self.device).eval()
            self.logger.info("Models initialized successfully")
            self.logger.info(f"YOLO class names: {self.yolo.model.names}")
        except Exception as e:
            self.logger.error(f"Failed to initialize models: {e}")
            raise

    def _initialize_trackers(self):
        self.trackers = []
        for cam_id in range(len(self.config.camera_streams)):
            trackers_per_cam = []
            for cls_id in range(len(self.config.class_names)):
                if cls_id in self.config.class_configs:
                    try:
                        tracker = BYTETracker(
                            **self.config.class_configs[cls_id],
                            cls_id=cls_id,
                            device=self.device
                        )
                        trackers_per_cam.append(tracker)
                        self.logger.debug(f"Initialized tracker for cam_id {cam_id}, cls_id {cls_id} ({self.config.class_names[cls_id]})")
                    except Exception as e:
                        self.logger.error(f"Failed to initialize tracker for cam_id {cam_id}, cls_id {cls_id}: {str(e)}")
                        raise
                else:
                    self.logger.warning(f"No configuration for cls_id {cls_id} in class_configs, skipping tracker initialization")
            self.trackers.append(trackers_per_cam)
        num_cams = len(self.trackers)
        num_classes = len(self.trackers[0]) if self.trackers else 0
        self.logger.info(f"Initialized trackers for {num_cams} cameras, {num_classes} classes")

    def preload_all_frames_synced(self):
        w, h = self.config.target_size
        caps = []
        valid_streams = []
        frame_counts = []
        for i, path in enumerate(self.config.camera_streams):
            self.logger.debug(f"Attempting to open video stream: {path}")
            cap = cv2.VideoCapture(path)
            if not cap.isOpened():
                self.logger.error(f"Failed to open video stream: {path}")
                cap.release()
                continue
            frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            self.logger.info(f"Camera {i} stream {path}: {frame_count} frames")
            caps.append(cap)
            valid_streams.append(i)
            frame_counts.append(frame_count)
        if not caps:
            self.logger.error("No valid video streams available")
            return [[] for _ in self.config.camera_streams], []
        
        all_frames = [[] for _ in self.config.camera_streams]
        max_len = max(frame_counts) if frame_counts else 0
        if self.config.max_frames is not None:
            max_len = min(max_len, self.config.max_frames)
            self.logger.info(f"Limiting processing to {max_len} frames due to max_frames setting")
        else:
            self.logger.info(f"Synchronizing to longest video length: {max_len} frames")
        valid_frame_flags = [[True] * min(frame_count, max_len) + [False] * (max_len - min(frame_count, max_len)) for frame_count in frame_counts]
        
        for i, cap in zip(valid_streams, caps):
            frame_idx = 0
            while frame_idx < min(frame_counts[i], max_len):
                ret, frame = cap.read()
                if not ret:
                    self.logger.debug(f"Camera {i}: End of stream at frame {frame_idx}")
                    break
                frame = cv2.resize(frame, (w, h))
                frame_tensor = torch.from_numpy(frame).permute(2, 0, 1).float() / 255.0
                all_frames[i].append(frame_tensor.to(self.device))
                frame_idx += 1
            cap.release()
        
        black_frame = torch.zeros((3, h, w), dtype=torch.float32, device=self.device)
        for i in range(len(all_frames)):
            if not all_frames[i]:
                self.logger.warning(f"No frames loaded for camera {i}")
                all_frames[i] = [black_frame.clone() for _ in range(max_len)]
            while len(all_frames[i]) < max_len:
                all_frames[i].append(black_frame.clone())
        
        self.logger.info(f"Preloaded {max_len} synchronized frames across {len(valid_streams)} cameras")
        return all_frames, valid_frame_flags

    def extract_features(self, frames, boxes):
        self.logger.debug(f"Extracting features for {len(frames)} frames with {len(boxes)} box lists")
        if not boxes or not any(len(box_list) > 0 for box_list in boxes):
            self.logger.info("No valid boxes provided for feature extraction")
            return torch.zeros((0, 512), dtype=torch.float32, device=self.device)
        patches = []
        frames_np = [(frame.permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8) for frame in frames]
        for frame_np, box_list in zip(frames_np, boxes):
            for box in box_list:
                try:
                    x1, y1, x2, y2 = map(int, box[:4].cpu().numpy())
                    x1, y1 = max(0, x1), max(0, y1)
                    x2, y2 = min(frame_np.shape[1], x2), min(frame_np.shape[0], y2)
                    if x2 <= x1 or y2 <= y1:
                        self.logger.warning(f"Invalid bounding box [{x1}, {y1}, {x2}, {y2}] skipped")
                        continue
                    patch = frame_np[y1:y2, x1:x2]
                    patch = cv2.resize(patch, (128, 256))
                    patches.append(patch)
                except Exception as e:
                    self.logger.error(f"Error processing box {box}: {e}")
                    continue
        if not patches:
            self.logger.info("No valid patches extracted")
            return torch.zeros((0, 512), dtype=torch.float32, device=self.device)
        try:
            patches = np.array(patches, dtype=np.float32) / 255.0
            patches = torch.from_numpy(patches).permute(0, 3, 1, 2).to(self.device)
            with torch.no_grad():
                feats = self.reid(patches)
            self.logger.debug(f"Extracted {feats.shape[0]} features")
            return F.normalize(feats, dim=-1)
        except Exception as e:
            self.logger.error(f"Error in feature extraction: {e}")
            return torch.zeros((0, 512), dtype=torch.float32, device=self.device)

    def assign_global(self, cam_id, track_ids, feats, cls_ids, bboxes, scores):
        self.logger.debug(f"Assigning global IDs for cam_id {cam_id}, {len(track_ids)} tracks")
        if not track_ids.numel() or len(track_ids) != len(feats) or len(feats) != len(cls_ids) or len(cls_ids) != len(bboxes) or len(bboxes) != len(scores):
            self.logger.error(f"Input length mismatch: track_ids={len(track_ids)}, feats={len(feats)}, cls_ids={len(cls_ids)}, bboxes={len(bboxes)}, scores={len(scores)}")
            return {}
        new_detections = []
        for feat, cls_id, box, tid, score in zip(feats, cls_ids, bboxes, track_ids, scores):
            try:
                new_detections.append(Track(tlbr=box, tid=tid, feat=feat, score=score, cls_id=cls_id, device=self.device))
            except Exception as e:
                self.logger.error(f"Error creating Track for tid {tid}: {e}")
                continue
        unmatched_dets = torch.arange(len(new_detections), device=self.device)
        unmatched_gids = list(self.global_gid_centroids.keys())
        global_tracks_fused = []
        for gid in unmatched_gids:
            if not self.global_tracks[gid]:
                continue
            last_entry = self.global_tracks[gid][-1]
            tbox, tid, f, cls_id = last_entry
            tbox = torch.tensor(tbox, dtype=torch.float32, device=self.device) if not isinstance(tbox, torch.Tensor) else tbox
            f = torch.tensor(f, dtype=torch.float32, device=self.device) if not isinstance(f, torch.Tensor) else f
            global_tracks_fused.append(Track(tlbr=tbox, tid=tid, feat=f, score=1.0, cls_id=cls_id, gid=gid, device=self.device))
        if len(global_tracks_fused) > 0 and len(new_detections) > 0:
            cost_m = torch.zeros((len(global_tracks_fused), len(new_detections)), device=self.device)
            for g, gt in enumerate(global_tracks_fused):
                for d, dt in enumerate(new_detections):
                    cost_m[g, d] = 1.0 - BYTETracker.cosine_similarity(gt.smooth_feat, dt.smooth_feat)
            _, row_idx, col_idx = lapjv(cost_m.cpu().numpy(), extend_cost=True)
            matches = [(row, col) for row, col in zip(row_idx, col_idx) if col >= 0 and row < cost_m.shape[0] and col < cost_m.shape[1] and cost_m[row, col] < self.config.class_threshold]
            for row, col in matches:
                gid = global_tracks_fused[row].gid
                new_detections[col].gid = gid
                self.global_tracks[gid].append((new_detections[col].tlbr.cpu().numpy(), new_detections[col].tid, new_detections[col].curr_feat.cpu().numpy(), new_detections[col].cls_id))
                unmatched_dets = unmatched_dets[unmatched_dets != col]
        for i in unmatched_dets:
            self.global_id_counter += 1
            new_detections[i].gid = self.global_id_counter
            self.global_tracks[self.global_id_counter] = [(new_detections[i].tlbr.cpu().numpy(), new_detections[i].tid, new_detections[i].curr_feat.cpu().numpy(), new_detections[i].cls_id)]
            if self.global_id_counter not in self.gid_color:
                self.gid_color[self.global_id_counter] = tuple(random.choices(range(256), k=3))
        return {t.tid: t.gid for t in new_detections if t.gid is not None}

    @with_timeout(600)
    def process_batch(self, frames_batch, cam_id, tracker_list, writer, track_file, frame_indices, valid_frames):
        self.logger.debug(f"Processing batch for cam_id {cam_id}, frames {frame_indices[0]} to {frame_indices[-1]}")
        all_tracks = []
        with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU, torch.profiler.ProfilerActivity.CUDA]) as prof:
            for frame_idx, frame, is_valid in zip(frame_indices, frames_batch, valid_frames):
                if not is_valid:
                    self.logger.debug(f"Skipping frame {frame_idx} for cam_id {cam_id}: End of stream")
                    frame_np = (frame.permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8)
                    writer.write(frame_np)
                    self.logger.info(f"Frame {frame_idx} for cam_id {cam_id} written to video with 0 bounding boxes")
                    self.logger.info(f"Completed frame {frame_idx} for cam_id {cam_id}")
                    continue
                try:
                    results = self.yolo(frame[None], conf=self.config.conf_thres, iou=self.config.iou_thres)
                    det_raw = results[0].boxes.data
                    det_raw = torch.tensor(det_raw, device=self.device) if len(det_raw) > 0 else torch.empty((0, 6), device=self.device)
                    self.logger.debug(f"Frame {frame_idx}, cam_id {cam_id}: {len(det_raw)} detections, class IDs: {det_raw[:, 5].int().tolist() if len(det_raw) > 0 else 'none'}")
                    if len(det_raw) == 0:
                        self.logger.debug(f"No detections for frame {frame_idx}, cam_id {cam_id}")
                        frame_np = (frame.permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8)
                        writer.write(frame_np)
                        self.logger.info(f"Frame {frame_idx} for cam_id {cam_id} written to video with 0 bounding boxes")
                        self.logger.info(f"Completed frame {frame_idx} for cam_id {cam_id}")
                        continue
                    feats = self.extract_features([frame], [det_raw])
                    unique_cls_ids = torch.unique(det_raw[:, 5]).int().tolist()
                    self.logger.debug(f"Unique class IDs: {unique_cls_ids}")
                    frame_tracks = []
                    for cls_id in unique_cls_ids:
                        if cls_id not in self.config.class_configs:
                            continue
                        cls_mask = det_raw[:, 5] == cls_id
                        cls_dets = det_raw[cls_mask]
                        cls_feats = feats[cls_mask]
                        self.logger.debug(f"Class {cls_id} ({self.config.class_names[cls_id]}): {len(cls_dets)} detections, {len(cls_feats)} features")
                        tracker = tracker_list[cls_id]
                        track_arr, tids, track_feats, class_ids, track_scores = tracker.update(cls_dets, cls_feats)
                        self.logger.debug(f"Tracker for class {cls_id} returned {len(track_arr)} tracks, tids shape: {tids.shape}")
                        if len(track_arr) > 0:
                            frame_tracks.append((track_arr, tids, track_feats, class_ids, track_scores))
                    if frame_tracks:
                        track_arr = torch.cat([t[0] for t in frame_tracks], dim=0)
                        tids = torch.cat([t[1] for t in frame_tracks])
                        track_feats = torch.cat([t[2] for t in frame_tracks], dim=0)
                        class_ids = torch.cat([t[3] for t in frame_tracks])
                        track_scores = torch.cat([t[4] for t in frame_tracks])
                        tid_to_gid = self.assign_global(cam_id, tids, track_feats, class_ids, track_arr[:, :4], track_scores)
                        self.logger.debug(f"Frame {frame_idx}, cam_id {cam_id}: {len(tid_to_gid)} global ID assignments")
                        frame_np = (frame.permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8)
                        for i, (box, tid, cls_id, score) in enumerate(zip(track_arr[:, :4], tids, class_ids, track_scores)):
                            gid = tid_to_gid.get(tid.item(), None)
                            if gid is None:
                                continue
                            x1, y1, x2, y2 = map(int, box.cpu().numpy())
                            color = self.gid_color.get(gid, self.config.color_map[int(cls_id)])
                            cv2.rectangle(frame_np, (x1, y1), (x2, y2), color, 2)
                            label = f"{self.config.class_names[int(cls_id)]} ID:{gid}"
                            cv2.putText(frame_np, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
                            self.track_buffers[cam_id].append(f"{frame_idx},{gid},{cls_id},{x1},{y1},{x2},{y2},{score.item()}\n")
                        writer.write(frame_np)
                        self.logger.info(f"Frame {frame_idx} for cam_id {cam_id} written to video with {len(track_arr)} bounding boxes")
                    else:
                        frame_np = (frame.permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8)
                        writer.write(frame_np)
                        self.logger.info(f"Frame {frame_idx} for cam_id {cam_id} written to video with 0 bounding boxes")
                    self.logger.info(f"Completed frame {frame_idx} for cam_id {cam_id}")
                    self.logger.debug(f"GPU memory after frame {frame_idx} for cam_id {cam_id}: {torch.cuda.memory_allocated(0)/1e6:.2f} MB")
                except Exception as e:
                    self.logger.error(f"Error processing frame {frame_idx} for cam_id {cam_id}: {e}")
                    frame_np = (frame.permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8)
                    writer.write(frame_np)
                    self.logger.info(f"Frame {frame_idx} for cam_id {cam_id} written to video with 0 bounding boxes")
                    self.logger.info(f"Completed frame {frame_idx} for cam_id {cam_id}")
                    continue
                finally:
                    torch.cuda.empty_cache()
        for line in self.track_buffers[cam_id]:
            track_file.write(line)
        self.track_buffers[cam_id].clear()
        return all_tracks

    def run(self):
        self.logger.info("Starting run method")
        os.makedirs(self.config.output_dir, exist_ok=True)
        writers = []
        track_files = []
        for i in range(len(self.config.camera_streams)):
            output_video = os.path.join(self.config.output_dir, f"output_camera_{i}.mp4")
            track_file_path = os.path.join(self.config.output_dir, f"tracks_camera_{i}.txt")
            writer = cv2.VideoWriter(
                output_video,
                cv2.VideoWriter_fourcc(*'mp4v'),
                self.config.fps,
                self.config.target_size
            )
            track_file = open(track_file_path, 'w')
            track_file.write("frame_id,global_id,class_id,x1,y1,x2,y2,score\n")
            writers.append(writer)
            track_files.append(track_file)
            self.logger.info(f"Initialized track file {track_file_path} and video writer for camera {i}")
        all_frames, valid_frame_flags = self.preload_all_frames_synced()
        max_len = len(all_frames[0]) if all_frames else 0
        self.logger.info(f"Processing {max_len} frames with batch size {self.config.batch_size}")
        def process_camera(cam_id):
            tracker_list = self.trackers[cam_id]
            writer = writers[cam_id]
            track_file = track_files[cam_id]
            for start_idx in range(0, max_len, self.config.batch_size):
                end_idx = min(start_idx + self.config.batch_size, max_len)
                frames_batch = all_frames[cam_id][start_idx:end_idx]
                frame_indices = list(range(start_idx, end_idx))
                valid_frames = valid_frame_flags[cam_id][start_idx:end_idx]
                try:
                    self.process_batch(frames_batch, cam_id, tracker_list, writer, track_file, frame_indices, valid_frames)
                except Exception as e:
                    self.logger.error(f"Error in batch processing for cam_id {cam_id}, frames {start_idx} to {end_idx-1}: {e}")
                    continue
                finally:
                    torch.cuda.empty_cache()
        with ThreadPoolExecutor(max_workers=len(self.config.camera_streams)) as executor:
            executor.map(process_camera, range(len(self.config.camera_streams)))
        self.logger.info(f"Processed {max_len} frames")
        for w, t in zip(writers, track_files):
            if w:
                w.release()
            if t:
                t.close()
        self.logger.info(f"Saved videos and tracks: {os.listdir(self.config.output_dir)}")