from typing import Tuple

import torch

"""
Table for the 0.95 quantile of the chi-square distribution with N degrees of
freedom (contains values for N=1, ..., 9). Taken from MATLAB/Octave's chi2inv
function and used as Mahalanobis gating threshold.
"""
chi2inv95 = {
    1: 3.8415,
    2: 5.9915,
    3: 7.8147,
    4: 9.4877,
    5: 11.070,
    6: 12.592,
    7: 14.067,
    8: 15.507,
    9: 16.919,
}


@torch.jit.script
class KalmanFilter(object):
    """
    A simple Kalman filter for tracking bounding boxes in image space.
    The 8-dimensional state space
        x, y, a, h, vx, vy, va, vh
    contains the bounding box center position (x, y), aspect ratio a, height h,
    and their respective velocities.
    Object motion follows a constant velocity model. The bounding box location
    (x, y, a, h) is taken as direct observation of the state space (linear
    observation model).
    """

    def __init__(self, device: str = "cuda"):
        ndim, dt = 4, 1.0

        # Create Kalman filter model matrices.
        self._motion_mat = torch.eye(2 * ndim, 2 * ndim, device=device)
        for i in range(ndim):
            self._motion_mat[i, ndim + i] = dt

        self._update_mat = torch.eye(ndim, 2 * ndim, device=device)

        # Motion and observation uncertainty are chosen relative to the current
        # state estimate. These weights control the amount of uncertainty in
        # the model. This is a bit hacky.
        self._std_weight_position = 1.0 / 20
        self._std_weight_velocity = 1.0 / 160

    def initiate(self, measurement: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Create track from unassociated measurement.
        Parameters
        ----------
        measurement : torch.Tensor
            Bounding box coordinates (x, y, a, h) with center position (x, y),
            aspect ratio a, and height h.
        Returns
        -------
        (torch.Tensor, torch.Tensor)
            Returns the mean vector (8 dimensional) and covariance matrix (8x8
            dimensional) of the new track. Unobserved velocities are initialized
            to 0 mean.
        """
        # [4]
        mean_pos = measurement
        # [4]
        mean_vel = torch.zeros_like(mean_pos)
        # [8]
        mean = torch.cat([mean_pos, mean_vel])
        # [8]
        std = torch.stack(
            [
                2 * self._std_weight_position * measurement[0],  # the center point x
                2 * self._std_weight_position * measurement[1],  # the center point y
                1 * measurement[2],  # the ratio of width/height
                2 * self._std_weight_position * measurement[3],  # the height
                10 * self._std_weight_velocity * measurement[0],
                10 * self._std_weight_velocity * measurement[1],
                0.1 * measurement[2],
                10 * self._std_weight_velocity * measurement[3],
            ]
        )
        # [8, 8]
        covariance = torch.diag(torch.square(std))
        return mean, covariance

    def predict(
        self, mean: torch.Tensor, covariance: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Run Kalman filter prediction step.
        Parameters
        ----------
        mean : torch.Tensor
            The 8 dimensional mean vector of the object state at the previous
            time step.
        covariance : torch.Tensor
            The 8x8 dimensional covariance matrix of the object state at the
            previous time step.
        Returns
        -------
        (torch.Tensor, torch.Tensor)
            Returns the mean vector and covariance matrix of the predicted
            state. Unobserved velocities are initialized to 0 mean.
        """
        # [4]
        std_pos = torch.stack(
            [
                self._std_weight_position * mean[0],
                self._std_weight_position * mean[1],
                1 * mean[2],
                self._std_weight_position * mean[3],
            ]
        )
        # [4]
        std_vel = torch.stack(
            [
                self._std_weight_velocity * mean[0],
                self._std_weight_velocity * mean[1],
                0.1 * mean[2],
                self._std_weight_velocity * mean[3],
            ]
        )
        # [8, 8]
        motion_cov = torch.diag(torch.square(torch.cat([std_pos, std_vel])))
        # [8]
        new_mean = torch.mm(self._motion_mat, mean.unsqueeze(1)).squeeze(1)
        # [8, 8]
        new_covariance = (
            torch.linalg.multi_dot((self._motion_mat, covariance, self._motion_mat.T))
            + motion_cov
        )

        # # TODO: remove, used for checking consistency
        # mean2 = np.dot(self._motion_mat.cpu().numpy(), mean.cpu().numpy())
        # covariance2 = (
        #     np.linalg.multi_dot(
        #         (
        #             self._motion_mat.cpu().numpy(),
        #             covariance.cpu().numpy(),
        #             self._motion_mat.T.cpu().numpy(),
        #         )
        #     )
        #     + motion_cov.cpu().numpy()
        # )
        # assert np.allclose(new_mean.cpu().numpy(), mean2)
        # assert np.allclose(new_covariance.cpu().numpy(), covariance2)

        return new_mean, new_covariance

    def project(
        self,
        mean: torch.Tensor,
        covariance: torch.Tensor,
        confidence: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Project state distribution to measurement space.
        Parameters
        ----------
        mean : ndarray
            The state's mean vector (8 dimensional array).
        covariance : ndarray
            The state's covariance matrix (8x8 dimensional).
        confidence: (dyh) 检测框置信度
        Returns
        -------
        (ndarray, ndarray)
            Returns the projected mean and covariance matrix of the given state
            estimate.
        """
        # [4]
        std = torch.stack(
            [
                self._std_weight_position * mean[3],
                self._std_weight_position * mean[3],
                torch.tensor(1e-1, device=mean.device),
                self._std_weight_position * mean[3],
            ]
        )
        # [4]
        std = torch.stack([(1 - confidence) * x for x in std])
        # [4, 4]
        innovation_cov = torch.diag(torch.square(std))

        # [4]
        new_mean = torch.mm(self._update_mat, mean.unsqueeze(1)).squeeze(1)
        # [4, 4]
        new_covariance = torch.linalg.multi_dot(
            (self._update_mat, covariance, self._update_mat.T)
        )

        # # ##########################################
        # # TODO: remove, used for checking consistency
        # # [4, 4]
        # innovation_cov2 = np.diag(np.square(std.cpu().numpy()))
        # # [4]
        # mean2 = np.dot(self._update_mat.cpu().numpy(), mean.cpu().numpy())
        # # [4, 4]
        # covariance2 = np.linalg.multi_dot(
        #     (
        #         self._update_mat.cpu().numpy(),
        #         covariance.cpu().numpy(),
        #         self._update_mat.T.cpu().numpy(),
        #     )
        # )
        # assert np.allclose(new_mean.cpu().numpy(), mean2)
        # assert np.allclose(new_covariance.cpu().numpy(), covariance2)
        # assert np.allclose(innovation_cov.cpu().numpy(), innovation_cov2)
        # # ##########################################
        return new_mean, new_covariance + innovation_cov

    def update(
        self,
        mean: torch.Tensor,
        covariance: torch.Tensor,
        measurement: torch.Tensor,
        confidence: torch.Tensor = torch.tensor(0.0, device="cuda"),
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Run Kalman filter correction step.
        Parameters
        ----------
        mean : ndarray
            The predicted state's mean vector (8 dimensional).
        covariance : ndarray
            The state's covariance matrix (8x8 dimensional).
        measurement : ndarray
            The 4 dimensional measurement vector (x, y, a, h), where (x, y)
            is the center position, a the aspect ratio, and h the height of the
            bounding box.
        confidence: (dyh)检测框置信度
        Returns
        -------
        (ndarray, ndarray)
            Returns the measurement-corrected state distribution.
        """
        projected_mean, projected_cov = self.project(mean, covariance, confidence)

        chol_factor = torch.linalg.cholesky(projected_cov, upper=False)

        b_m = torch.mm(covariance, self._update_mat.T).T
        kalman_gain = torch.cholesky_solve(b_m, chol_factor).T

        innovation = measurement - projected_mean

        new_mean = mean + torch.mm(innovation.unsqueeze(0), kalman_gain.T).squeeze(0)
        new_covariance = covariance - torch.linalg.multi_dot(
            (kalman_gain, projected_cov, kalman_gain.T)
        )

        # # ##########################################
        # # TODO: remove, used for checking consistency
        # # [4, 4]
        # chol_factor2, lower2 = scipy.linalg.cho_factor(
        #     projected_cov.cpu().numpy(), lower=True, check_finite=False
        # )
        # # [8, 4]
        # kalman_gain2 = scipy.linalg.cho_solve(
        #     (chol_factor2, lower2),
        #     np.dot(covariance.cpu().numpy(), self._update_mat.T.cpu().numpy()).T,
        #     check_finite=False,
        # ).T

        # new_mean_2 = mean.cpu().numpy() + np.dot(
        #     innovation.cpu().numpy(), kalman_gain2.T
        # )
        # # new_covariance_2 = covariance.cpu().numpy() - np.linalg.multi_dot(
        # #     (kalman_gain2, projected_cov.cpu().numpy(), kalman_gain2.T)
        # # )
        # assert np.allclose(kalman_gain2, kalman_gain.cpu().numpy())
        # assert np.allclose(new_mean_2, new_mean.cpu().numpy())
        # # ##########################################

        return new_mean, new_covariance

    def gating_distance(
        self,
        mean: torch.Tensor,
        covariance: torch.Tensor,
        measurements: torch.Tensor,
        only_position: bool = False,
    ) -> torch.Tensor:
        """Compute gating distance between state distribution and measurements.
        A suitable distance threshold can be obtained from `chi2inv95`. If
        `only_position` is False, the chi-square distribution has 4 degrees of
        freedom, otherwise 2.
        Parameters
        ----------
        mean : torch.Tensor
            Mean vector over the state distribution (8 dimensional).
        covariance : torch.Tensor
            Covariance of the state distribution (8x8 dimensional).
        measurements : torch.Tensor
            An Nx4 dimensional matrix of N measurements, each in
            format (x, y, a, h) where (x, y) is the bounding box center
            position, a the aspect ratio, and h the height.
        only_position : Optional[bool]
            If True, distance computation is done with respect to the bounding
            box center position only.
        Returns
        -------
        ndarray
            Returns an array of length N, where the i-th element contains the
            squared Mahalanobis distance between (mean, covariance) and
            `measurements[i]`.
        """
        # [4], [4, 4]
        mean, covariance = self.project(
            mean,
            covariance,
            confidence=torch.tensor(0.0, device=covariance.device),
        )

        if only_position:
            mean, covariance = mean[:2], covariance[:2, :2]
            measurements = measurements[:, :2]

        # [4, 4]
        cholesky_factor = torch.linalg.cholesky(covariance)
        # [7, 4]
        d = measurements - mean
        # [4, 7]
        z = torch.linalg.solve_triangular(cholesky_factor, d.T, upper=False)
        # [7]
        squared_maha = torch.sum(z * z, dim=0)

        # # ##########################################
        # # TODO: remove, used for checking consistency
        # cholesky_factor2 = np.linalg.cholesky(covariance.cpu().numpy())
        # z2 = scipy.linalg.solve_triangular(
        #     cholesky_factor2,
        #     d.T.cpu().numpy(),
        #     lower=True,
        #     check_finite=False,
        #     overwrite_b=True,
        # )
        # squared_maha2 = np.sum(z2 * z2, axis=0)

        # assert np.allclose(cholesky_factor.cpu().numpy(), cholesky_factor2)
        # assert np.allclose(z.cpu().numpy(), z2)
        # assert np.allclose(squared_maha.cpu().numpy(), squared_maha2)
        # # ##########################################
        return squared_maha
