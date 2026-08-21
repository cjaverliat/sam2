# SPDX-License-Identifier: Apache-2.0
"""What a caller hands a predictor: geometry prompts and concept prompts.

A :class:`GeometryPrompt` says WHERE, and its route says which half of the model is
being addressed -- ``click`` / ``clicks`` / ``box`` / ``boxes`` / ``mask`` mark ONE
object for the tracker (SAM 2 semantics), while ``exemplar_point(s)`` /
``exemplar_box(es)`` give a concept search examples to bias it. A
:class:`ConceptPrompt` says WHAT, and only SAM 3 takes one.

Prompts are per-frame evidence about one object, so several for the same object are
merged rather than refused -- see :func:`merge_object_prompts`, which every predictor
runs before consuming a frame's prompts.
"""
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


# A DETECTOR-route prompt names no object: detection mints the ids it produces.
_NO_OBJECT = -1


def _xy(xy, what):
    """``xy`` as a flat ``(x, y)`` list, or raise naming the constructor."""
    coords = torch.as_tensor(xy, dtype=torch.float32).reshape(-1)
    if coords.numel() != 2:
        raise ValueError(f"{what} expects an (x, y) pair, got {tuple(coords.shape)}")
    return coords.tolist()


def _xyxy(xyxy, what):
    """``xyxy`` as a flat corner list, or raise naming the constructor."""
    coords = torch.as_tensor(xyxy, dtype=torch.float32).reshape(-1)
    if coords.numel() != 4:
        raise ValueError(
            f"{what} expects (xmin, ymin, xmax, ymax), got {tuple(coords.shape)}")
    return coords.tolist()


def _sign(label, what):
    """A prompt sign: 1 for an example, 0 for a counter-example."""
    if label not in (0, 1):
        raise ValueError(f"{what} label must be 1 or 0, got {label!r}")
    return label


def _labels(count, labels, what, unit):
    """One int32 label per item, defaulting to all-positive.

    Raises:
        ValueError: if the caller passed labels but not one per item -- a mismatch
            means one of the two lists is wrong, and guessing which would be worse.
    """
    if labels is None:
        return torch.ones(count, dtype=torch.int32)
    out = torch.as_tensor(labels, dtype=torch.int32).reshape(-1)
    if out.shape[0] != count:
        raise ValueError(
            f"{what} got {count} {unit}(s) and {out.shape[0]} label(s); pass one label "
            f"per {unit} or none at all"
        )
    return out


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

        if points_labels is not None and points_coords is None:
            raise ValueError(
                "points_coords must be provided if points_labels is provided"
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

    # -- Selecting ONE object: the SAM 2 gestures, routed to the tracker --------

    @classmethod
    def click(cls, obj_id: int, xy, label: int = 1) -> GeometryPrompt:
        """A click on ONE object at pixel ``xy``; ``label`` 1 positive, 0 negative.

        The SAM 2 gesture: this seeds or refines the object you name with ``obj_id``
        through the tracker, and nothing else comes back. To bias a concept search
        instead, use :meth:`exemplar_point`.
        """
        return cls.clicks(obj_id, [_xy(xy, "click")], [label])

    @classmethod
    def clicks(cls, obj_id: int, xys, labels=None) -> GeometryPrompt:
        """Several clicks on ONE object -- the SAM 2 refinement gesture.

        Args:
            obj_id: the object every point refers to.
            xys: ``[(x, y), ...]`` in pixels.
            labels: one per point, 1 to include and 0 to exclude; all 1 by default.
        """
        coords = torch.as_tensor(xys, dtype=torch.float32).reshape(-1, 2)
        return cls(
            obj_id=obj_id,
            points_coords=coords,
            points_labels=_labels(coords.shape[0], labels, "clicks", "point"),
        )

    @classmethod
    def box(cls, obj_id: int, xyxy) -> GeometryPrompt:
        """A box around ONE object (interactive VOS): seeds it from pixel ``xyxy``.

        The SAM 2 gesture, encoded as the box's two corners (labels 2 and 3, which is
        exactly what the prompt encoder's native box path builds). Detection does not
        run, so only this object is tracked. Several boxes for one object:
        :meth:`boxes`. To bias a concept search instead: :meth:`exemplar_box`.
        """
        return cls.boxes(obj_id, [_xyxy(xyxy, "box")])

    @classmethod
    def boxes(cls, obj_id: int, xyxys) -> GeometryPrompt:
        """Several boxes bounding ONE object, each encoded as its two corners.

        The prompt encoder takes any number of corner pairs for an object, so this is
        for an object one rectangle describes badly. For several *objects*, use one
        prompt each; for several examples of a concept, :meth:`exemplar_boxes`.

        Args:
            obj_id: the object every box refers to.
            xyxys: ``[(x0, y0, x1, y1), ...]`` in pixels.
        """
        return cls(obj_id=obj_id, boxes=torch.as_tensor(xyxys, dtype=torch.float32).reshape(-1, 4))

    @classmethod
    def mask(cls, obj_id: int, mask) -> GeometryPrompt:
        """A mask over ONE object, from an ``(H, W)`` boolean mask or float logits.

        Tracker-only: the detector has no mask slot in either SAM 3 checkpoint, so
        there is no ``exemplar_mask`` counterpart.
        """
        m = torch.as_tensor(mask)
        if m.dtype == torch.bool:
            m = m.float() * 20.0 - 10.0  # binarising at 0 recovers the input
        return cls(obj_id=obj_id, masks_logits=m.float())

    # -- Steering a concept search: examples, routed to the detector -----------

    @classmethod
    def exemplar_point(cls, xy, label: int = 1) -> GeometryPrompt:
        """An EXAMPLE at pixel ``xy`` (SAM 3): biases the concept search toward it.

        The concept still decides WHAT comes back; this only says where to look harder.
        ``label`` 1 marks an example, 0 a counter-example ("everything matching the
        concept EXCEPT this"). A point carries no extent, so it steers more weakly than
        a box -- measurably so on a negative label.

        Contrast :meth:`click`, the SAM 2 gesture, which returns that one object alone.
        """
        return cls.exemplar_points([_xy(xy, "exemplar_point")], [_sign(label, "exemplar_point")])

    @classmethod
    def exemplar_points(cls, xys, labels=None) -> GeometryPrompt:
        """Several example points for one concept search.

        They belong in one prompt because they describe one search, not one object
        each, and so this takes no ``obj_id``: the ids come from detection.

        Args:
            xys: ``[(x, y), ...]`` in pixels.
            labels: one per point, 1 for an example and 0 for a counter-example; all 1
                by default.
        """
        coords = torch.as_tensor(xys, dtype=torch.float32).reshape(-1, 2)
        return cls(
            obj_id=_NO_OBJECT,
            points_coords=coords,
            points_labels=_labels(coords.shape[0], labels, "exemplar_points", "point"),
            route=PromptRoute.DETECTOR,
        )

    @classmethod
    def exemplar_box(cls, xyxy, label: int = 1) -> GeometryPrompt:
        """An EXAMPLE of what to find (SAM 3): biases the concept search on this frame.

        The concept still decides what comes back; the box says "more like this one".
        ``label`` 1 makes it an example, 0 a counter-example. Needs a concept: a phrase,
        or the box-only ``PLACEHOLDER`` caption.

        Contrast :meth:`box`, which picks out ONE object and returns it alone.
        """
        signed = None if label == 1 else [_sign(label, "exemplar_box")]
        return cls.exemplar_boxes([_xyxy(xyxy, "exemplar_box")], signed)

    @classmethod
    def exemplar_boxes(cls, xyxys, labels=None) -> GeometryPrompt:
        """Several example boxes for one concept search.

        Args:
            xyxys: ``[(x0, y0, x1, y1), ...]`` in pixels.
            labels: one per box, 1 for an example and 0 for a counter-example ("every
                match EXCEPT this one"). Left unset they are all examples, which the
                packer reads as all-positive without building a tensor.
        """
        boxes = torch.as_tensor(xyxys, dtype=torch.float32).reshape(-1, 4)
        return cls(
            obj_id=_NO_OBJECT,
            boxes=boxes,
            boxes_labels=(None if labels is None
                          else _labels(boxes.shape[0], labels, "exemplar_boxes", "box")),
            route=PromptRoute.DETECTOR,
        )

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


def merge_object_prompts(prompts, image_hw=None) -> list["GeometryPrompt"]:
    """Combine prompts that describe the same object into one prompt each.

    A frame's prompts are evidence, and evidence adds up: ``clicks(2, ...)`` alongside
    ``boxes(2, ...)`` is one object described two ways, not two objects. Merging here
    means the named constructors compose by listing, instead of every mixed prompt
    dropping the caller back to the generic ``GeometryPrompt(...)`` form.

    Points and boxes concatenate in call order. Order matters to nobody downstream --
    the prompt encoder sums the embeddings -- but keeping it makes the merged prompt
    read like the calls that built it.

    Args:
        prompts: this frame's prompts, in any order (None or empty is fine).
        image_hw: ``(height, width)``, needed only when one object's prompts disagree
            about ``is_normalized`` -- the pixel ones are divided through so the merge
            has one space to work in. That is the same division the predictors apply
            later, so it changes no numbers.

    Returns:
        One prompt per ``(obj_id, route)``, in first-seen order. Single prompts are
        passed through untouched.

    Raises:
        ValueError: if one object is described in both routes (the tracker and the
            detector answer different questions), if its prompts mix coordinate spaces
            and no ``image_hw`` is given to reconcile them, or if it carries two masks --
            a mask is an object's whole extent, so a second one contradicts the first
            rather than adding to it.
    """
    if not prompts:
        return []

    groups: dict[tuple, list[GeometryPrompt]] = {}
    for prompt in prompts:
        groups.setdefault((prompt.obj_id, prompt.route), []).append(prompt)

    routes_per_obj: dict[int, set] = {}
    for obj_id, route in groups:
        routes_per_obj.setdefault(obj_id, set()).add(route)
    mixed = sorted(o for o, routes in routes_per_obj.items() if len(routes) > 1)
    if mixed:
        raise ValueError(
            f"obj_id {mixed} is prompted on both routes in one frame: click/box/mask "
            "select an object through the tracker while exemplar_* bias a concept "
            "search, and one object cannot be both. Send them as separate calls"
        )

    return [
        group[0] if len(group) == 1 else _merge_one_object(obj_id, route, group, image_hw)
        for (obj_id, route), group in groups.items()
    ]


def _normalized_copy(prompt: "GeometryPrompt", image_hw) -> "GeometryPrompt":
    """The prompt with pixel coordinates divided through by ``(W, H)``.

    The same division the predictors do on the way to the model, applied early so a
    merge has one coordinate space; doing it here changes no downstream numbers.
    """
    height, width = image_hw
    scale_xy = torch.tensor([width, height], dtype=torch.float32)
    scale_box = torch.tensor([width, height, width, height], dtype=torch.float32)
    coords = prompt.points_coords
    boxes = prompt.boxes
    return GeometryPrompt(
        obj_id=prompt.obj_id,
        points_coords=None if coords is None else coords / scale_xy.to(coords.device),
        points_labels=prompt.points_labels,
        boxes=None if boxes is None else boxes / scale_box.to(boxes.device),
        boxes_labels=prompt.boxes_labels,
        masks_logits=prompt.masks_logits,
        is_normalized=True,
        route=prompt.route,
    )


def _merge_one_object(obj_id, route, group, image_hw) -> "GeometryPrompt":
    """Concatenate one object's prompts into a single :class:`GeometryPrompt`."""
    if len({p.is_normalized for p in group}) > 1:
        if image_hw is None:
            raise ValueError(
                f"obj_id {obj_id} mixes normalized and pixel coordinates in one frame "
                "and there is no image size here to reconcile them; build every prompt "
                "for an object in the same space"
            )
        group = [p if p.is_normalized else _normalized_copy(p, image_hw) for p in group]

    masks = [p.masks_logits for p in group if p.masks_logits is not None]
    if len(masks) > 1:
        raise ValueError(
            f"obj_id {obj_id} got {len(masks)} masks in one frame: a mask is the "
            "object's whole extent, so a second one contradicts the first. Send one "
            "mask, optionally with clicks to refine it"
        )

    coords = [p.points_coords for p in group if p.points_coords is not None]
    labels = [p.points_labels for p in group if p.points_labels is not None]
    boxed = [p for p in group if p.boxes is not None]

    boxes_labels = None
    if boxed and any(p.boxes_labels is not None for p in boxed):
        # one prompt signed its boxes -> the unsigned ones are positive, as ever
        boxes_labels = torch.cat([
            p.boxes_labels.to(torch.int32) if p.boxes_labels is not None
            else torch.ones(p.boxes.shape[0], dtype=torch.int32, device=p.boxes.device)
            for p in boxed
        ])

    return GeometryPrompt(
        obj_id=obj_id,
        points_coords=torch.cat(coords) if coords else None,
        points_labels=torch.cat(labels) if labels else None,
        boxes=torch.cat([p.boxes for p in boxed]) if boxed else None,
        boxes_labels=boxes_labels,
        masks_logits=masks[0] if masks else None,
        is_normalized=group[0].is_normalized,
        route=route,
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
