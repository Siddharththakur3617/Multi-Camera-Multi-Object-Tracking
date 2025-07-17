import torch

class MyKalmanFilter:
    def __init__(self, device='cpu'):
        self.device = device
        self.dim_x = 8  # [x1, y1, x2, y2, vx1, vy1, vx2, vy2]
        self.dim_z = 4  # [x, y, w, h]
        self.P = torch.eye(self.dim_x, dtype=torch.float32, device=self.device) * 1000.0
        self.Q = torch.eye(self.dim_x, dtype=torch.float32, device=self.device) * 0.8
        self.R = torch.eye(self.dim_z, dtype=torch.float32, device=self.device) * 10.0
        self.chi2inv95 = {4: 7.81}

    def f(self, x):
        x_new = torch.zeros_like(x)
        x_new[0] = x[0] + x[4]
        x_new[1] = x[1] + x[5]
        x_new[2] = x[2] + x[6]
        x_new[3] = x[3] + x[7]
        x_new[4:] = x[4:]
        return x_new

    def F_jacobian(self, x):
        F = torch.eye(8, dtype=torch.float32, device=self.device)
        F[0, 4] = 1
        F[1, 5] = 1
        F[2, 6] = 1
        F[3, 7] = 1
        return F

    def h(self, x):
        return x[:4]

    def H_jacobian(self, x):
        H = torch.zeros((4, 8), dtype=torch.float32, device=self.device)
        H[0, 0] = 1
        H[1, 1] = 1
        H[2, 2] = 1
        H[3, 3] = 1
        return H

    def initiate(self, measurement):
        mean = torch.zeros(8, dtype=torch.float32, device=self.device)
        mean[:4] = measurement
        return mean, self.P.clone()

    def predict(self, mean, covariance):
        F = self.F_jacobian(mean)
        mean = self.f(mean)
        covariance = F @ covariance @ F.T + self.Q
        return mean, covariance

    def update(self, mean, covariance, measurement):
        H = self.H_jacobian(mean)
        z_pred = self.h(mean)
        y = measurement - z_pred
        S = H @ covariance @ H.T + self.R
        K = covariance @ H.T @ torch.linalg.inv(S)
        mean = mean + K @ y
        I = torch.eye(self.dim_x, device=self.device)
        covariance = (I - K @ H) @ covariance
        return mean, covariance

    def gating_distance(self, mean, covariance, measurements, only_position=False):
        if only_position:
            H = self.H_jacobian(mean)[:2, :]
            R = self.R[:2, :2]
            z_pred = self.h(mean)[:2]
        else:
            H = self.H_jacobian(mean)
            R = self.R
            z_pred = self.h(mean)
        S = H @ covariance @ H.T + R
        invS = torch.linalg.inv(S)
        diff = measurements - z_pred
        dists = torch.einsum('ij,jk,ik->i', diff, invS, diff)
        return dists