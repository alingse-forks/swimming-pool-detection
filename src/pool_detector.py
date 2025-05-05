import cv2
import numpy as np
import torch
from PIL import Image
from torchvision import transforms

from .models.architectures.resnet import ResNet as Model


class PoolDetector:
    """Swimming pool detector.

    Args:
        weights_path (str): Path of CNN weights.
        device (str, optional): device for torch computations.

    Example::

        pool_detector = PoolDetector("weights.pth")
        pool_coords = pool_detector.detect(img_path)
    """

    def __init__(self, weights_path, device=None):
        self.device = torch.device(device) or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = self.load_model(weights_path)

    def load_model(self, weights_path):
        # Force CPU device for model loading
        device = torch.device('cpu')
        model = Model().to(device)
        model.load_state_dict(torch.load(weights_path, map_location=device))
        model.eval()
        # Move to target device if different
        if self.device.type != 'cpu':
            model = model.to(self.device)
        return model

    def detect(self, img, heatmap_thresh=170):
        """Main method for swimming pool detection.

        Args:
            img: Can be either:
                - str: Path of image to process
                - np.array: Image as numpy array (BGR or RGB format)
            heatmap_thresh (int, optional): Heatmap threshold. Defaults to 170.

        Returns:
            list: List that contains swimming pool coordinates.
        """
        if isinstance(img, str):
            img = Image.open(img)
            img = np.array(img)
        elif isinstance(img, np.ndarray):
            # Handle different image formats
            if len(img.shape) == 2:  # Grayscale (1 channel)
                img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
            elif len(img.shape) == 3:
                if img.shape[2] == 4:  # RGBA (PNG with alpha)
                    img = cv2.cvtColor(img, cv2.COLOR_BGRA2RGB)
                elif img.shape[2] == 3:  # BGR (OpenCV default)
                    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        heatmap = self.generate_heatmap(img)
        pools_list = self.find_pools(heatmap, heatmap_thresh)
        # Filter based on blue pixels
        filtered_pools = self.filter_blue(img, pools_list)
        return filtered_pools

    def generate_heatmap(self, img):
        """Method to generate a heatmap from CNN class activation maps (CAMs).

        Args:
            img (np.array): Input image.
            window_size (int): Tile size to feed into the CNN.

        Returns:
            np.array: Heatmap
        """

        # convert numpy to tensor,  scale, todevice
        batch = transforms.ToTensor()(img)
        batch = torch.unsqueeze(batch, 0)
        batch = batch.to(self.device)

        # inference
        with torch.no_grad():
            cam = self.model(batch, apply_avgpool=False)  # (1, 256, 100, 200)
            cam = cam * self.model.fc.weight.view((1, -1, 1, 1)).flip(1)  # class activation map weights
            cam = cam.mean(dim=1, keepdim=True)  # (1, 1, 100, 200)
            # Get original image dimensions for interpolation
            height, width = img.shape[:2]
            cam = torch.nn.functional.interpolate(
                cam,
                size=(height, width),
                mode="bicubic",
                align_corners=True,
            )

        # Post process
        cam = cam.cpu().numpy()
        heatmap = cam[0, 0, ...]
        heatmap = (heatmap - np.min(heatmap)) / (np.max(heatmap) - np.min(heatmap))
        heatmap = 1 - heatmap
        heatmap = (heatmap * 255).astype(np.uint8)
        return heatmap

    @staticmethod
    def filter_blue(img, pools_list, box_size=40, min_blue=200):
        """Filter pools based on blue pixel count in surrounding area.

        Args:
            img: Original RGB image
            pools_list: List of pool coordinates (y,x)
            box_size: Size of area to check around each point
            min_blue: Minimum blue pixels required to keep

        Returns:
            Filtered list of pool coordinates
        """
        filtered = []
        hsv = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)

        # Define blue color range in HSV
        #lower_blue = np.array([100, 50, 50])
        #upper_blue = np.array([140, 255, 255])
        lower_blue = np.array([80, 70, 100])
        upper_blue = np.array([130, 255, 255])

        for y, x in pools_list:
            # Get box area
            y1 = max(0, y - box_size//2)
            y2 = min(img.shape[0], y + box_size//2)
            x1 = max(0, x - box_size//2)
            x2 = min(img.shape[1], x + box_size//2)

            # Count blue pixels
            area = hsv[y1:y2, x1:x2]
            mask = cv2.inRange(area, lower_blue, upper_blue)
            blue_count = cv2.countNonZero(mask)
            print(blue_count, "blue_count")
            if blue_count >= min_blue:
                filtered.append((y, x))

        return filtered

    @staticmethod
    def find_pools(heatmap, threshold, min_contour_area=2):
        """Find swimming pool coordinates from heatmap.

        Algorithm:
        1) Blur.
        2) Binarize.
        3) Detect contours.
        4) Compute contour centers.

        Args:
            heatmap (np.array): Heatmap.
            threshold (int): Threshold keep only high activations.
            min_contour_area (int, optional): Filter contours by area. Defaults to 2.

        Returns:
            list: List that contains swimming pool coordinates.
        """
        heatmap = cv2.GaussianBlur(heatmap, (15, 15), 0)
        _, heatmap = cv2.threshold(heatmap, threshold, 255, 0)
        contours, _ = cv2.findContours(heatmap, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        contours = [x for x in contours if cv2.contourArea(x) >= min_contour_area]
        pools = []
        for cnt in contours:
            M = cv2.moments(cnt)
            cx = int(M["m10"] / M["m00"])
            cy = int(M["m01"] / M["m00"])
            pools.append((cy, cx))
        return pools
