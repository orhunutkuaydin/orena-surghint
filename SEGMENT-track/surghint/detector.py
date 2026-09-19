"""RF-DETR-L detection with fixed batching and class-wise suppression."""

from importlib.metadata import version

from .rendering import EvidenceConfig, filter_detections
from .weights import references, verify_file

BATCH_SIZE = 32


class Detector:
    def __init__(self, checkpoint):
        import torch
        from rfdetr import from_checkpoint

        if version("rfdetr") != "1.9.1":
            raise RuntimeError("RF-DETR inference requires rfdetr==1.9.1")
        verify_file(checkpoint, references()["custom/rfdetr-large.pth"])
        if not torch.cuda.is_available():
            raise RuntimeError("RF-DETR inference requires a CUDA GPU")
        self.config = EvidenceConfig()
        self.model = from_checkpoint(
            str(checkpoint), trust_checkpoint=True, device="cuda:0"
        )
        self.model.inference(
            compile=False, batch_size=BATCH_SIZE, dtype="float16", inplace=True
        )

    def predict(self, frames):
        count = len(frames)
        if not 0 < count <= BATCH_SIZE:
            raise ValueError("Invalid detector batch size")
        images = list(frames)
        images += [images[-1]] * (BATCH_SIZE - count)
        predictions = self.model.predict(
            images, threshold=self.config.threshold, include_source_image=False
        )
        if not isinstance(predictions, list):
            predictions = [predictions]
        if len(predictions) != BATCH_SIZE:
            raise RuntimeError("RF-DETR did not return one prediction per input frame")
        result = []
        for rgb, prediction in zip(frames, predictions[:count], strict=True):
            if len(prediction.xyxy) == 0:
                result.append({"boxes": [], "scores": [], "classes": []})
                continue
            if prediction.confidence is None or prediction.class_id is None:
                raise ValueError("Nonempty predictions lack confidence or class IDs")
            result.append(
                filter_detections(
                    prediction.xyxy,
                    prediction.confidence,
                    prediction.class_id,
                    rgb.shape[1],
                    rgb.shape[0],
                    self.config,
                )
            )
        return result
