# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

import enum

import torch


class PromptRoute(enum.Enum):
    """Which half of SAM 3 a geometry prompt is talking to.

    A prompt is one thing or the other, never both: you are either pointing at ONE
    object you want back (TRACKER, SAM 2 semantics) or describing what the concept
    search should look for (DETECTOR, SAM 3 only). The named constructors set this
    for you -- ``click`` / ``box`` / ``mask`` are TRACKER, ``exemplar_point`` /
    ``exemplar_box`` are DETECTOR -- so callers rarely name the route itself.

    TRACKER:
        The default. Points, box corners (labels 2 and 3) and masks go to the
        tracker's prompt encoder and seed or refine ONE object under the ``obj_id``
        you chose. Detection does not run, so nothing else can appear. Video only:
        the image predictor owns no tracker.
    DETECTOR:
        Points and boxes go to the detector's geometric slot and bias that frame's
        concept search; every instance the concept matches is returned, not only the
        one you marked, and the ids are the detector's. Needs a concept -- a phrase,
        or the box-only ``PLACEHOLDER`` caption. Masks have no detector slot: neither
        checkpoint ships ``mask_encoder`` weights.
    """

    TRACKER = "tracker"
    DETECTOR = "detector"


# The name this enum shipped under when only boxes could be routed.
BoxRoute = PromptRoute


class GeometryPrompt:
    def __init__(
        self,
        obj_id: int,
        points_coords: torch.Tensor | None = None,
        points_labels: torch.Tensor | None = None,
        boxes: torch.Tensor | None = None,
        boxes_labels: torch.Tensor | None = None,
        masks_logits: torch.Tensor | None = None,
        is_normalized: bool = False,
        route: PromptRoute = PromptRoute.TRACKER,
    ):
        if (
            points_coords is None
            and points_labels is None
            and boxes is None
            and masks_logits is None
        ):
            raise ValueError(
                "At least one of points_coords, points_labels, boxes, or masks_logits must be provided"
            )

        if points_coords is not None and points_labels is None:
            raise ValueError(
                "points_labels must be provided if points_coords is provided"
            )
        
        if points_coords is not None and (points_coords.ndim != 2 or points_coords.shape[1] != 2):
            raise ValueError(f"Expected points_coords to be of shape (N, 2), got {points_coords.shape}")
        
        if points_labels is not None and (points_labels.ndim != 1 or points_labels.shape[0] != points_coords.shape[0]):
            raise ValueError(f"Expected points_labels to be of shape (N,), got {points_labels.shape}")
        
        if boxes is not None and (boxes.ndim != 2 or boxes.shape[1] != 4):
            raise ValueError(f"Expected boxes to be of shape (N, 4), got {boxes.shape}")


        if boxes_labels is not None and boxes is None:
            raise ValueError("boxes must be provided if boxes_labels is provided")

        if boxes_labels is not None and (boxes_labels.ndim != 1 or boxes_labels.shape[0] != boxes.shape[0]):
            raise ValueError(f"Expected boxes_labels to be of shape (N,), got {boxes_labels.shape}")

        if masks_logits is not None and route is PromptRoute.DETECTOR:
            raise NotImplementedError(
                "the detector has no mask slot: neither SAM 3 checkpoint ships "
                "mask_encoder weights. Use exemplar_box / exemplar_point to bias the "
                "search, or GeometryPrompt.mask(obj_id, mask) to prompt the tracker"
            )

        if masks_logits is not None:
            mask_res = masks_logits.shape[-2:]
            masks_logits = masks_logits.reshape(1, *mask_res) # Reshape to (1, H, W)

        self.obj_id = obj_id
        self.points_coords = points_coords
        self.points_labels = points_labels
        self.boxes = boxes
        self.boxes_labels = boxes_labels
        self.masks_logits = masks_logits
        self.is_normalized = is_normalized
        self.route = route

    @classmethod
    def click(cls, obj_id: int, xy, label: int = 1) -> GeometryPrompt:
        """A click on ONE object at pixel ``xy``; ``label`` 1 positive, 0 negative.

        The SAM 2 gesture: this seeds or refines the object you name with ``obj_id``
        through the tracker, and nothing else comes back. To bias a concept search
        instead, use :meth:`exemplar_point`.
        """
        coords = torch.as_tensor(xy, dtype=torch.float32).reshape(-1)
        if coords.numel() != 2:
            raise ValueError(f"click expects an (x, y) pair, got {tuple(coords.shape)}")
        return cls(
            obj_id=obj_id,
            points_coords=coords.reshape(1, 2),
            points_labels=torch.tensor([label], dtype=torch.int32),
        )

    @classmethod
    def clicks(cls, obj_id: int, xys, labels=None) -> GeometryPrompt:
        """Several clicks on ONE object -- the SAM 2 refinement gesture.

        One prompt carries all of an object's evidence for a frame, so refining a
        selection means more points in THIS prompt, not a second prompt under the same
        ``obj_id`` (which is rejected: two prompts would be two answers to "what is
        object 3").

        Args:
            obj_id: the object every point refers to.
            xys: ``[(x, y), ...]`` in pixels.
            labels: one per point, 1 to include and 0 to exclude; all 1 by default.
        """
        coords = torch.as_tensor(xys, dtype=torch.float32).reshape(-1, 2)
        if labels is None:
            point_labels = torch.ones(coords.shape[0], dtype=torch.int32)
        else:
            point_labels = torch.as_tensor(labels, dtype=torch.int32).reshape(-1)
        if point_labels.shape[0] != coords.shape[0]:
            raise ValueError(
                f"clicks got {coords.shape[0]} point(s) and {point_labels.shape[0]} "
                "label(s); pass one label per point or none at all"
            )
        return cls(obj_id=obj_id, points_coords=coords, points_labels=point_labels)

    @classmethod
    def box(cls, obj_id: int, xyxy) -> GeometryPrompt:
        """A box around ONE object (interactive VOS): seeds it from pixel ``xyxy``.

        The SAM 2 gesture, encoded as the box's two corners (labels 2 and 3, which is
        exactly what the prompt encoder's native box path builds). Detection does not
        run, so only this object is tracked. Several boxes for one object:
        :meth:`boxes`. To bias a concept search instead: :meth:`exemplar_box`.
        """
        coords = torch.as_tensor(xyxy, dtype=torch.float32).reshape(-1)
        if coords.numel() != 4:
            raise ValueError(f"box expects (xmin, ymin, xmax, ymax), got {tuple(coords.shape)}")
        return cls(obj_id=obj_id, boxes=coords.reshape(1, 4))

    @classmethod
    def exemplar_box(cls, xyxy, label: int = 1) -> GeometryPrompt:
        """An EXAMPLE of what to find (SAM 3): biases the concept search on this frame.

        The concept still decides what comes back; the box says "more like this one".
        ``label`` 1 makes it an example, 0 a counter-example -- "everything matching the
        concept EXCEPT this". Needs a concept: a phrase, or the box-only ``PLACEHOLDER``
        caption (``start_concept_session`` -- or ``set_concept`` /
        ``set_placeholder_concept`` on an explicit state).

        Takes no ``obj_id``: an example names nothing, so the instances it helps surface
        get their ids from detection. Contrast :meth:`box`, which picks out ONE object
        and returns it alone.
        """
        coords = torch.as_tensor(xyxy, dtype=torch.float32).reshape(-1)
        if coords.numel() != 4:
            raise ValueError(f"exemplar_box expects (xmin, ymin, xmax, ymax), got {tuple(coords.shape)}")
        return cls(
            obj_id=-1,  # unused: detection mints the ids
            boxes=coords.reshape(1, 4),
            boxes_labels=None if label == 1 else torch.tensor([label]),
            route=PromptRoute.DETECTOR,
        )

    @classmethod
    def exemplar_point(cls, xy, label: int = 1) -> GeometryPrompt:
        """An EXAMPLE at pixel ``xy`` (SAM 3): biases the concept search toward it.

        The point form of :meth:`exemplar_box`, and the same contract: the concept
        still decides WHAT comes back, this only says where to look harder. ``label``
        1 marks the point as an example, 0 as a counter-example ("everything matching
        the concept EXCEPT this one"). A point carries no extent, so it steers more
        weakly than a box -- measurably so on a negative label.

        Needs a concept -- a phrase, or the predictor's ``PLACEHOLDER`` for the
        box-only caption. Takes no ``obj_id``: a detector point selects nothing, so
        the ids come from detection.

        Contrast :meth:`click`, which is the SAM 2 gesture: that one picks out ONE
        object and returns it alone.
        """
        coords = torch.as_tensor(xy, dtype=torch.float32).reshape(-1)
        if coords.numel() != 2:
            raise ValueError(
                f"exemplar_point expects an (x, y) pair, got {tuple(coords.shape)}")
        if label not in (0, 1):
            raise ValueError(f"exemplar_point label must be 1 or 0, got {label!r}")
        return cls(
            obj_id=-1,  # unused: detection mints the ids
            points_coords=coords.reshape(1, 2),
            points_labels=torch.tensor([label], dtype=torch.int32),
            route=PromptRoute.DETECTOR,
        )

    @classmethod
    def boxes(cls, obj_id: int, xyxys) -> GeometryPrompt:
        """Several boxes bounding ONE object, each encoded as its two corners.

        The tracker's prompt encoder takes any number of corner pairs for an object
        (``sam2_predictor`` repeats labels 2/3 per box), so this is for an object one
        rectangle describes badly. For several *objects*, use one prompt each; for
        several examples of a concept, :meth:`exemplar_boxes`.

        Args:
            obj_id: the object every box refers to.
            xyxys: ``[(x0, y0, x1, y1), ...]`` in pixels.
        """
        coords = torch.as_tensor(xyxys, dtype=torch.float32).reshape(-1, 4)
        return cls(obj_id=obj_id, boxes=coords)

    @classmethod
    def exemplar_points(cls, xys, labels=None) -> GeometryPrompt:
        """Several example points for one concept search.

        The detector takes as many examples as you have; they all belong in one prompt
        because they describe one search, not one object each.

        Args:
            xys: ``[(x, y), ...]`` in pixels.
            labels: one per point, 1 for an example and 0 for a counter-example; all 1
                by default.
        """
        coords = torch.as_tensor(xys, dtype=torch.float32).reshape(-1, 2)
        point_labels = (
            torch.ones(coords.shape[0], dtype=torch.int32) if labels is None
            else torch.as_tensor(labels, dtype=torch.int32).reshape(-1)
        )
        if point_labels.shape[0] != coords.shape[0]:
            raise ValueError(
                f"exemplar_points got {coords.shape[0]} point(s) and "
                f"{point_labels.shape[0]} label(s); pass one label per point or none"
            )
        return cls(
            obj_id=-1,  # unused: detection mints the ids
            points_coords=coords,
            points_labels=point_labels,
            route=PromptRoute.DETECTOR,
        )

    @classmethod
    def exemplar_boxes(cls, xyxys, labels=None) -> GeometryPrompt:
        """Several example boxes for one concept search.

        Args:
            xyxys: ``[(x0, y0, x1, y1), ...]`` in pixels.
            labels: one per box, 1 for an example and 0 for a counter-example ("every
                match EXCEPT this one"); all 1 by default.
        """
        boxes = torch.as_tensor(xyxys, dtype=torch.float32).reshape(-1, 4)
        box_labels = (
            None if labels is None
            else torch.as_tensor(labels, dtype=torch.int32).reshape(-1)
        )
        if box_labels is not None and box_labels.shape[0] != boxes.shape[0]:
            raise ValueError(
                f"exemplar_boxes got {boxes.shape[0]} box(es) and "
                f"{box_labels.shape[0]} label(s); pass one label per box or none"
            )
        return cls(
            obj_id=-1,
            boxes=boxes,
            boxes_labels=box_labels,
            route=PromptRoute.DETECTOR,
        )

    @classmethod
    def mask(cls, obj_id: int, mask) -> GeometryPrompt:
        """A mask over ONE object, from an ``(H, W)`` boolean mask or float logits.

        Tracker-only: the detector has no mask slot in either SAM 3 checkpoint, so
        there is no ``concept_mask`` counterpart.
        """
        m = torch.as_tensor(mask)
        if m.dtype == torch.bool:
            m = m.float() * 20.0 - 10.0  # binarising at 0 recovers the input
        return cls(obj_id=obj_id, masks_logits=m.float())

    @property
    def to_detector(self) -> bool:
        """Whether this prompt carries a box bound for the detector's geometric slot."""
        return self.route is PromptRoute.DETECTOR

    def tracker_points(self) -> tuple[torch.Tensor, torch.Tensor] | None:
        """The points this prompt feeds to the tracker's prompt encoder.

        Clicks keep their own labels (1 foreground / 0 background); a TRACKER-route box
        contributes its corners as labels 2 (top-left) and 3 (bottom-right), the same
        encoding :class:`~sam.models.sam2_predictor.Sam2VideoPredictor` uses. Coordinates
        stay in whatever space the prompt was built in (pixel unless ``is_normalized``).

        Returns:
            ``(coords, labels)`` of shapes ``(N, 2)`` / ``(N,)``, or None when the prompt
            has nothing for the tracker (a DETECTOR-route box, or a mask alone).
        """
        if self.route is PromptRoute.DETECTOR:
            return None  # every DETECTOR geometry belongs to the geometric slot
        parts = []
        if self.points_coords is not None:
            parts.append((self.points_coords, self.points_labels))
        if self.boxes is not None and self.route is PromptRoute.TRACKER:
            corners = self.boxes.reshape(-1, 2)
            labels = torch.tensor(
                [2, 3], dtype=torch.int32, device=corners.device
            ).repeat(self.boxes.shape[0])
            parts.append((corners, labels))
        if not parts:
            return None
        coords = torch.cat([c for c, _ in parts], dim=0)
        labels = torch.cat([lb.to(coords.device) for _, lb in parts], dim=0)
        return coords, labels

    def to(self, device: torch.device) -> GeometryPrompt:
        points_coords = (
            self.points_coords.to(device) if self.points_coords is not None else None
        )
        points_labels = (
            self.points_labels.to(device) if self.points_labels is not None else None
        )
        boxes = self.boxes.to(device) if self.boxes is not None else None
        boxes_labels = (
            self.boxes_labels.to(device) if self.boxes_labels is not None else None
        )
        masks_logits = (
            self.masks_logits.to(device) if self.masks_logits is not None else None
        )
        return GeometryPrompt(
            obj_id=self.obj_id,
            points_coords=points_coords,
            points_labels=points_labels,
            boxes=boxes,
            boxes_labels=boxes_labels,
            masks_logits=masks_logits,
            is_normalized=self.is_normalized,
            route=self.route,
        )

    def clone(self) -> GeometryPrompt:
        return GeometryPrompt(
            obj_id=self.obj_id,
            points_coords=self.points_coords.clone() if self.points_coords is not None else None,
            points_labels=self.points_labels.clone() if self.points_labels is not None else None,
            boxes=self.boxes.clone() if self.boxes is not None else None,
            boxes_labels=self.boxes_labels.clone() if self.boxes_labels is not None else None,
            masks_logits=self.masks_logits.clone() if self.masks_logits is not None else None,
            is_normalized=self.is_normalized,
            route=self.route,
        )


# SPDX-License-Identifier: LicenseRef-SAM
# SAM 3 concept prompt — carries the concept text.  Encoding (text → embeddings)
# is the predictor's responsibility; this type is a pure data carrier.


class ConceptPrompt:
    """Per-concept SAM 3 prompt.

    Text is the only field. Upstream SAM 3 has no inference-time semantics for
    a negative phrase (see :meth:`sam.models.sam3_predictor.Sam3Predictor.encode_text`)
    nor for a VISUAL-slot exemplar: ``visual_prompt_embed`` reaches a live
    consumer but no released code path — training included — ever builds one.
    Reference geometry is expressed as a :class:`GeometryPrompt` box/point, which
    is what upstream's own "image exemplar" resolves to.

    Args:
        text: Positive description of the concept to segment (e.g. "cat").
    """

    def __init__(self, text: str):
        self.text = text
