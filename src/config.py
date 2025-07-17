import json
import os

class Config:
    def __init__(self, config_path="config/config.json"):
        # Default values
        self.yolo_model_path = "/path/to/custom_yolov11.pt"
        self.reid_model = "osnet_x0_25"
        self.class_names = ['ball', 'goalkeeper', 'player', 'referee']
        self.color_map = {
            0: (0, 255, 0),    # Green for ball
            1: (255, 0, 0),    # Red for goalkeeper
            2: (0, 0, 255),    # Blue for player
            3: (255, 255, 0)   # Yellow for referee
        }
        self.camera_streams = ["/path/to/broadcast.mp4", "/path/to/tacticam.mp4"]
        self.output_dir = "output_videos"
        self.fps = 30
        self.target_size = (960, 540)
        self.conf_thres = 0.6
        self.iou_thres = 0.4
        self.batch_size = 16
        self.max_frames = 20
        self.class_configs = {
            0: {"track_thresh": 0.5, "buffer": 20, "max_tracks": 5},   # Ball
            1: {"track_thresh": 0.4, "buffer": 30, "max_tracks": 2},   # Goalkeeper
            2: {"track_thresh": 0.5, "buffer": 40, "max_tracks": 15},  # Player
            3: {"track_thresh": 0.45, "buffer": 20, "max_tracks": 5}   # Referee
        }
        self.memory_per_gid = 100
        self.class_threshold = 0.7

        # Load from config file if it exists
        if os.path.exists(config_path):
            with open(config_path, 'r') as f:
                config_data = json.load(f)
                for key, value in config_data.items():
                    setattr(self, key, value)