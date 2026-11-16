from pathlib import Path

import cv2
import numpy as np
import torch
import segmentation_models_pytorch as smp


BASE_DIR = Path(__file__).resolve().parents[3]

CHECKPOINT_PATH = (
    BASE_DIR / "backend" / "models" / "UNetPlusPlus_best.pth"
)

DEVICE = torch.device(
    "mps" if torch.backends.mps.is_available() else "cpu"
)

IMAGE_SIZE = 256


class NucleusSegmenter:

    def __init__(self):

        self.device = DEVICE

        self.model = smp.UnetPlusPlus(
            encoder_name="resnet34",
            encoder_weights=None,
            in_channels=3,
            classes=1,
            activation=None
        )

        checkpoint = torch.load(
            CHECKPOINT_PATH,
            map_location="cpu",
            weights_only=False
        )

        self.model.load_state_dict(
            checkpoint["model_state_dict"]
        )

        self.model.to(self.device)
        self.model.eval()

        print("U-Net++ loaded successfully")
        print("Device:", self.device)

    def preprocess(self, image):

        height, width = image.shape[:2]

        scale = IMAGE_SIZE / max(height, width)

        new_width = max(1, int(width * scale))
        new_height = max(1, int(height * scale))

        resized = cv2.resize(
            image,
            (new_width, new_height),
            interpolation=cv2.INTER_LINEAR
        )

        canvas = np.zeros(
            (IMAGE_SIZE, IMAGE_SIZE, 3),
            dtype=np.uint8
        )

        x_offset = (IMAGE_SIZE - new_width) // 2
        y_offset = (IMAGE_SIZE - new_height) // 2

        canvas[
            y_offset:y_offset + new_height,
            x_offset:x_offset + new_width
        ] = resized

        tensor = (
            torch.from_numpy(canvas)
            .float()
            .permute(2, 0, 1)
            .unsqueeze(0) / 255.0
        )

        return (
            tensor.to(self.device),
            x_offset,
            y_offset,
            new_width,
            new_height
        )

    @torch.inference_mode()
    def predict(self, image):

        tensor, x, y, new_width, new_height = (
            self.preprocess(image)
        )

        logits = self.model(tensor)

        probability = torch.sigmoid(logits)

        prediction = (
            probability.squeeze().cpu().numpy() > 0.5
        ).astype(np.uint8)

        prediction = prediction[
            y:y + new_height,
            x:x + new_width
        ]

        original_height, original_width = image.shape[:2]

        mask = cv2.resize(
            prediction,
            (original_width, original_height),
            interpolation=cv2.INTER_NEAREST
        )

        return mask


segmenter = NucleusSegmenter()