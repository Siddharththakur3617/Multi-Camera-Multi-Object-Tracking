import argparse
from .config import Config
from .multi_camera_tracker import MultiCameraTracker

def main():
    parser = argparse.ArgumentParser(description="Multi-Camera Object Tracking with YOLOv11, ByteTrack, and ReID")
    parser.add_argument("--batch_size", type=int, default=16, help="Batch size for processing frames")
    parser.add_argument("--config_path", type=str, default="config/config.json", help="Path to configuration file")
    args = parser.parse_args()

    config = Config(config_path=args.config_path)
    config.batch_size = args.batch_size

    tracker = MultiCameraTracker(config)
    tracker.run()

if __name__ == "__main__":
    main()