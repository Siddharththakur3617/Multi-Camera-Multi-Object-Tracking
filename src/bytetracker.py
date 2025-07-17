import torch
import numpy as np
from lap import lapjv
from .track import Track
from .kalman_filter import MyKalmanFilter

class BYTETracker:
    def __init__(self, track_thresh=0.2, match_thresh_high=0.9, match_thresh_low=0.5, buffer=250, cls_id=0, device='cpu', max_tracks=10):
        self.device = device
        self.track_thresh = track_thresh
        self.match_thresh_high = match_thresh_high
        self.match_thresh_low = match_thresh_low
        self.buffer = int(buffer)
        self.cls_id = cls_id
        self.max_tracks = max_tracks
        self.tracked_tracks = []
        self.lost_tracks = []
        self.removed_tracks = []
        self.max_tid = 0
        self.frame_id = 0
        self.kf = MyKalmanFilter(device=device)

    def iou(self, b1, b2):
        x1 = torch.max(b1[0], b2[0])
        y1 = torch.max(b1[1], b2[1])
        x2 = torch.min(b1[2], b2[2])
        y2 = torch.min(b1[3], b2[3])
        inter = torch.clamp(x2 - x1, min=0) * torch.clamp(y2 - y1, min=0)
        area1 = (b1[2] - b1[0]) * (b1[3] - b1[1])
        area2 = (b2[2] - b2[0]) * (b2[3] - b2[1])
        return inter / (area1 + area2 - inter + 1e-6)

    def gate_cost_matrix(self, cost_matrix, tracks, dets, use_strict=False):
        if cost_matrix.numel() == 0:
            return cost_matrix
        measurements = torch.stack([det[:4].to(self.device) for det in dets])
        for i, tr in enumerate(tracks):
            gating_distance = self.kf.gating_distance(tr.mean, tr.covariance, measurements, only_position=False)
            mask = gating_distance > self.kf.chi2inv95[4]
            cost_matrix[i][mask] = 1e6
        return cost_matrix

    @staticmethod
    def cosine_similarity(a, b, eps=1e-6):
        a = torch.tensor(a) if not isinstance(a, torch.Tensor) else a
        b = torch.tensor(b) if not isinstance(b, torch.Tensor) else b
        dot_product = torch.dot(a, b)
        norm = torch.norm(a) * torch.norm(b)
        if norm < eps:
            return 0.0
        sim = dot_product / (norm + eps)
        return torch.clamp(sim, 0.0, 1.0).item()

    def _match(self, dets, feats):
        if not self.tracked_tracks or len(dets) == 0:
            return [], [], list(range(len(dets)))
        dets = dets.clone().detach().to(self.device) if len(dets) > 0 else torch.empty((0, 6), dtype=torch.float32, device=self.device)
        feats = feats.clone().detach().to(self.device) if len(feats) > 0 else torch.empty((0, 512), dtype=torch.float32, device=self.device)
        high_score = dets[:, 4] >= self.track_thresh
        low_score = (dets[:, 4] >= 0.1) & (dets[:, 4] < self.track_thresh)
        dets_high, feats_high = dets[high_score], feats[high_score]
        dets_low, feats_low = dets[low_score], feats[low_score]
        tracks = self.tracked_tracks + self.lost_tracks
        for tr in tracks:
            tr.predict()
        iou_m = torch.zeros((len(tracks), len(dets_high)), device=self.device)
        reid_m = torch.zeros_like(iou_m)
        for t, tr in enumerate(tracks):
            for d, dt in enumerate(dets_high):
                iou_val = self.iou(tr.tlbr, dt[:4])
                reid_val = self.cosine_similarity(tr.features[-1], feats_high[d]) if tr.features else 0.0
                if tr.cls_id != int(dt[5].item()):
                    iou_val *= 0.2
                    reid_val *= 0.2
                iou_m[t, d] = iou_val
                reid_m[t, d] = reid_val
        combined_m = 0.5 * iou_m + 0.5 * reid_m
        combined_m = self.gate_cost_matrix(combined_m, tracks, dets_high, use_strict=False)
        _, row_idx, col_idx = lapjv(combined_m.cpu().numpy(), extend_cost=True)
        matches = [(row, col) for row, col in zip(row_idx, col_idx) if col >= 0 and row < combined_m.shape[0] and col < combined_m.shape[1] and combined_m[row, col] < self.match_thresh_high]
        r_tracks = [tracks[t] for t in range(len(tracks)) if t not in {m[0] for m in matches} and tracks[t].state == 'Tracked']
        iou_m = torch.zeros((len(r_tracks), len(dets_low)), device=self.device)
        reid_m = torch.zeros_like(iou_m)
        for t, tr in enumerate(r_tracks):
            for d, dt in enumerate(dets_low):
                iou_val = self.iou(tr.tlbr, dt[:4])
                reid_val = self.cosine_similarity(tr.features[-1], feats_low[d]) if tr.features else 0.0
                if tr.cls_id != int(dt[5].item()):
                    iou_val *= 0.2
                    reid_val *= 0.2
                iou_m[t, d] = iou_val
                reid_m[t, d] = reid_val
        combined_m = 0.5 * iou_m + 0.5 * reid_m
        combined_m = self.gate_cost_matrix(combined_m, r_tracks, dets_low, use_strict=True)
        _, row_idx, col_idx = lapjv(combined_m.cpu().numpy(), extend_cost=True)
        matches2 = [(row, col) for row, col in zip(row_idx, col_idx) if col >= 0 and row < combined_m.shape[0] and col < combined_m.shape[1] and combined_m[row, col] < self.match_thresh_low]
        m = [(tracks[t], d) for t, d in matches]
        m += [(r_tracks[t], d + len(dets_high)) for t, d in matches2]
        matched_dets = [d for _, d in m]
        unmatched_dets = [d for d in range(len(dets)) if d not in matched_dets]
        return m, [tracks[i] for i in range(len(tracks)) if i not in {m[0] for m in matches}], unmatched_dets

    def update(self, dets, feats):
        self.frame_id += 1
        activated, refound, lost, removed = [], [], [], []
        dets = dets.clone().detach().to(self.device) if len(dets) > 0 else torch.empty((0, 6), dtype=torch.float32, device=self.device)
        feats = feats.clone().detach().to(self.device) if len(feats) > 0 else torch.empty((0, 512), dtype=torch.float32, device=self.device)
        m, u_tr, u_dt = self._match(dets, feats)
        for t, d in m:
            track, det = t, dets[d]
            track.update(det[:4], det[4], feats[d])
            track.state = 'Tracked'
            activated.append(track)
        all_tracks = self.tracked_tracks + self.lost_tracks
        for t in u_tr:
            if t.tid >= len(all_tracks):
                continue
            track = all_tracks[t.tid]
            track.mark_lost()
            if track.state != 'Removed':
                lost.append(track)
        u_dt = u_dt[:self.max_tracks - len(self.tracked_tracks) - len(activated)] if len(u_dt) > 0 else u_dt
        for d in u_dt:
            if d < len(dets):
                self.max_tid += 1
                det = dets[d]
                feat = feats[d]
                track = Track(det[:4], det[4], self.max_tid, int(det[5].item()) if det.shape[0] > 5 else -1, feat=feat, device=self.device)
                track.update(det[:4], det[4], feat)
                activated.append(track)
        self.tracked_tracks = [t for t in self.tracked_tracks if t.state == 'Tracked'] + activated + refound
        self.lost_tracks = [t for t in self.lost_tracks if t.state != 'Removed'] + lost
        self.removed_tracks += [t for t in self.lost_tracks if self.frame_id - t.age > self.buffer]
        self.lost_tracks = [t for t in self.lost_tracks if self.frame_id - t.age <= self.buffer]
        self.tracked_tracks = [t for t in self.tracked_tracks if t.time_since_update == 0]
        self.tracked_tracks = self.tracked_tracks[:self.max_tracks]
        active_tracks = self.tracked_tracks
        if active_tracks:
            track_arr = torch.stack([torch.cat((t.tlbr, torch.tensor([t.tid, t.cls_id], dtype=torch.float32, device=self.device))) for t in active_tracks])
        else:
            track_arr = torch.empty((0, 6), dtype=torch.float32, device=self.device)
        track_feats = []
        for t in active_tracks:
            if t.features:
                feat = torch.stack(list(t.features))
                track_feats.append(torch.mean(feat, dim=0))
            else:
                track_feats.append(torch.zeros(512, device=self.device))
        track_feats = torch.stack(track_feats) if track_feats else torch.empty((0, 512), device=self.device)
        track_ids = torch.tensor([t.tid for t in active_tracks], dtype=torch.int32, device=self.device)
        class_ids = torch.tensor([t.cls_id for t in active_tracks], dtype=torch.int32, device=self.device)
        track_scores = torch.tensor([t.score for t in active_tracks], dtype=torch.float32, device=self.device)
        return track_arr, track_ids, track_feats, class_ids, track_scores