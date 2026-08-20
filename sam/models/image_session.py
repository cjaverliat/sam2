# SPDX-License-Identifier: LicenseRef-SAM
"""The session wrapper over an image predictor's encode + detect.

An :class:`ImageSession` binds a predictor to one image the way a
:class:`~sam.models.video_session.VideoSession` binds one to a video: the state it
owns is the encoded image, and every ``process`` call reuses it. It adds no model
logic -- ``session.process(concept)`` runs the same detect as
``predictor.predict(image, concept)``, on features that call would have recomputed.

That recomputation is the point. The PE vision encoder is roughly 70% of a
``predict`` call (136 ms of 189 ms for an 1800x1200 image on a RTX 3090), and it does
not depend on the prompt, so a threshold sweep or a set of exemplars pays it once
here instead of once per prompt.

Sessions are independent: hold several over one loaded model and interleave them.
"""
from __future__ import annotations

from typing import Any

import torch


class ImageSession:
    """One image's worth of prompting over a shared predictor.

    Built by a predictor's ``start_image_session`` -- not directly. The image is
    encoded once, at construction, under the session's ``dtype``.

    Args:
        predictor: the predictor whose detect path this session drives.
        img_embeddings: the encoded image (the detector's pyramid view).
        img_pos_embeddings: the matching position encodings.
        image_hw: ``(height, width)`` of the original image, for un-scaling boxes.
        dtype: autocast dtype the image was encoded under; reused for every detect.
    """

    def __init__(
        self,
        predictor: Any,
        img_embeddings: list[torch.Tensor],
        img_pos_embeddings: list[torch.Tensor],
        image_hw: tuple[int, int],
        dtype: torch.dtype = torch.bfloat16,
    ) -> None:
        self._predictor = predictor
        self._img_embeddings = img_embeddings
        self._img_pos_embeddings = img_pos_embeddings
        self._image_hw = image_hw
        self._dtype = dtype

    @property
    def image_hw(self) -> tuple[int, int]:
        """``(height, width)`` of the image this session is bound to."""
        return self._image_hw

    @property
    def img_embeddings(self) -> list[torch.Tensor]:
        """The encoded image, for callers that want the block methods directly."""
        return self._img_embeddings

    @torch.inference_mode()
    def process(self, concept, confidence_threshold: float = 0.5, geometry=None):
        """Detect every instance of ``concept`` -> ``Sam3DetectionResult``.

        Identical to ``predictor.predict(image, concept, ...)`` for the image this
        session holds, minus the re-encode.

        Args:
            concept: the :class:`~sam.prompts.ConceptPrompt` (or a phrase).
            confidence_threshold: presence-weighted score threshold (default 0.5).
            geometry: optional :class:`~sam.prompts.GeometryPrompt` exemplar --
                points and/or boxes that bias the search, as on ``predict``.
        """
        return self._predictor._detect_encoded(
            self._img_embeddings,
            self._img_pos_embeddings,
            self._image_hw,
            concept,
            confidence_threshold=confidence_threshold,
            geometry=geometry,
            dtype=self._dtype,
        )
