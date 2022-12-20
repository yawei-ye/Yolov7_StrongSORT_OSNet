# vim: expandtab:ts=4:sw=4
from enum import Enum
from typing import List, Optional

import torch

from strong_sort.sort.detection import Detection
from strong_sort.sort.kalman_filter import KalmanFilter


class TrackState(Enum):
    """
    Enumeration type for the single target track state. Newly created tracks are
    classified as `tentative` until enough evidence has been collected. Then,
    the track state is changed to `confirmed`. Tracks that are no longer alive
    are classified as `deleted` to mark them for removal from the set of active
    tracks.

    """

    Tentative = 1
    Confirmed = 2
    Deleted = 3


@torch.jit.script
class Track:
    """
    A single target track with state space `(x, y, a, h)` and associated
    velocities, where `(x, y)` is the center of the bounding box, `a` is the
    aspect ratio and `h` is the height.

    Parameters
    ----------
    mean : ndarray
        Mean vector of the initial state distribution.
    covariance : ndarray
        Covariance matrix of the initial state distribution.
    track_id : int
        A unique track identifier.
    n_init : int
        Number of consecutive detections before the track is confirmed. The
        track state is set to `Deleted` if a miss occurs within the first
        `n_init` frames.
    max_age : int
        The maximum number of consecutive misses before the track state is
        set to `Deleted`.
    feature : Optional[ndarray]
        Feature vector of the detection this track originates from. If not None,
        this feature is added to the `features` cache.

    Attributes
    ----------
    mean : ndarray
        Mean vector of the initial state distribution.
    covariance : ndarray
        Covariance matrix of the initial state distribution.
    track_id : int
        A unique track identifier.
    hits : int
        Total number of measurement updates.
    age : int
        Total number of frames since first occurance.
    time_since_update : int
        Total number of frames since last measurement update.
    state : TrackState
        The current track state.
    features : List[ndarray]
        A cache of features. On each measurement update, the associated feature
        vector is added to this list.

    """

    def __init__(
        self,
        detection: torch.Tensor,
        track_id: int,
        class_id: torch.Tensor,
        conf: torch.Tensor,
        n_init: int,
        max_age: int,
        ema_alpha: float,
        feature: Optional[torch.Tensor] = None,
    ):
        self.track_id: int = track_id
        self.class_id: int = int(class_id)
        self.hits: int = 1
        self.age: int = 1
        self.time_since_update: int = 0
        self.ema_alpha: float = ema_alpha

        self.state = TrackState.Tentative
        self.features: List[torch.Tensor] = []
        if feature is not None:
            feature /= torch.norm(feature)
            self.features.append(feature)

        self.conf: torch.Tensor = conf
        self._n_init: int = n_init
        self._max_age: int = max_age

        self.kf = KalmanFilter()
        self.mean, self.covariance = self.kf.initiate(detection)

    def to_tlwh(self):
        """Get current position in bounding box format `(top left x, top left y,
        width, height)`.

        Returns
        -------
        ndarray
            The bounding box.

        """
        ret = self.mean[:4].clone()
        ret[2] *= ret[3]
        ret[:2] -= ret[2:] / 2
        return ret

    def to_tlbr(self):
        """Get kf estimated current position in bounding box format `(min x, miny, max x,
        max y)`.

        Returns
        -------
        ndarray
            The predicted kf bounding box.

        """
        ret = self.to_tlwh()
        ret[2:] = ret[:2] + ret[2:]
        return ret

    def get_matrix(self, matrix: torch.Tensor):
        eye = torch.eye(3, device=matrix.device)
        dist = torch.linalg.norm(eye - matrix)
        if dist < 100:
            return matrix
        else:
            return eye

    def increment_age(self):
        self.age += 1
        self.time_since_update += 1

    def predict(self, kf):
        """Propagate the state distribution to the current time step using a
        Kalman filter prediction step.

        Parameters
        ----------
        kf : kalman_filter.KalmanFilter
            The Kalman filter.

        """
        self.mean, self.covariance = self.kf.predict(self.mean, self.covariance)
        self.age += 1
        self.time_since_update += 1

    def update(self, detection: Detection, class_id: torch.Tensor, conf: torch.Tensor):
        """Perform Kalman filter measurement update step and update the feature
        cache.
        Parameters
        ----------
        detection : Detection
            The associated detection.
        """
        self.conf = conf
        self.class_id = int(class_id)
        self.mean, self.covariance = self.kf.update(
            self.mean, self.covariance, detection.to_xyah(), detection.confidence
        )

        feature = detection.feature / torch.linalg.norm(detection.feature)

        smooth_feat = (
            self.ema_alpha * self.features[-1] + (1 - self.ema_alpha) * feature
        )
        smooth_feat /= torch.linalg.norm(smooth_feat)
        self.features = [smooth_feat]

        self.hits += 1
        self.time_since_update = 0
        if self.state == TrackState.Tentative and self.hits >= self._n_init:
            self.state = TrackState.Confirmed

    def mark_missed(self):
        """Mark this track as missed (no association at the current time step)."""
        if self.state == TrackState.Tentative:
            self.state = TrackState.Deleted
        elif self.time_since_update > self._max_age:
            self.state = TrackState.Deleted

    def is_tentative(self):
        """Returns True if this track is tentative (unconfirmed)."""
        return self.state == TrackState.Tentative

    def is_confirmed(self):
        """Returns True if this track is confirmed."""
        return self.state == TrackState.Confirmed

    def is_deleted(self):
        """Returns True if this track is dead and should be deleted."""
        return self.state == TrackState.Deleted
