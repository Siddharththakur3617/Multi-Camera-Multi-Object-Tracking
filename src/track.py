import torch
from collections import deque
from .kalman_filter import MyKalmanFilter

class Track:
    def __init__(self, tlbr, score, tid, cls_id, feat=None, gid=None, max_feat_history=50, device='cuda'):
        if isinstance(tlbr, list):
            tlbr = torch.tensor(tlbr, dtype=torch.float32, device=device)
        elif isinstance(tlbr, np.ndarray):
            tlbr = torch.from_numpy(tlbr).float().to(device)
        assert isinstance(tlbr, torch.Tensor), f"Invalid tlbr: {tlbr}"

        self.device = device
        self.tlbr = tlbr
        self.score = score
        self.tid = tid
        self.cls_id = cls_id
        self.gid = gid
        self.kf = MyKalmanFilter(device=device)
        self.mean, self.covariance = self.kf.initiate(self.tlbr)
        self.state = 'Tracked'
        self.time_since_update = 0
        self.age = 1
        self.tracklet_len = 0
        self.features = deque(maxlen=max_feat_history)
        self.curr_feat = feat.to(self.device) if feat is not None else torch.zeros(512, dtype=torch.float32, device=self.device)
        self.smooth_feat = self.curr_feat.clone()
        if feat is not None:
            self.features.append(feat)

    def predict(self):
        if self.state != 'Tracked':
            self.mean[4:] = 0
        self.mean, self.covariance = self.kf.predict(self.mean, self.covariance)
        self.tlbr = self.mean[:4]
        self.age += 1
        self.time_since_update += 1

    def update(self, tlbr, score, feat=None, alpha=0.9):
        if isinstance(tlbr, list):
            tlbr = torch.tensor(tlbr, dtype=torch.float32, device=self.device)
        elif isinstance(tlbr, np.ndarray):
            tlbr = torch.from_numpy(tlbr).float().to(self.device)
        self.mean, self.covariance = self.kf.update(self.mean, self.covariance, tlbr)
        self.tlbr = tlbr
        self.score = score
        self.state = 'Tracked'
        self.time_since_update = 0
        self.tracklet_len += 1
        if feat is not None:
            feat = feat.to(self.device)
            self.curr_feat = feat
            self.features.append(feat)
            self.smooth_feat = alpha * self.smooth_feat + (1 - alpha) * feat
            norm = torch.norm(self.smooth_feat)
            if norm > 0:
                self.smooth_feat = self.smooth_feat / norm

    def mark_lost(self):
        self.state = 'Lost'

    def mark_removed(self):
        self.state = 'Removed'