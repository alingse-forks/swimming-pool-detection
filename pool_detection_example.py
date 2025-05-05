import os
import sys
import cv2
import matplotlib.pyplot as plt

from src.pool_detector import PoolDetector


def plot_boxes(img, pool_coords, box_size=40):
    for cy, cx in pool_coords:
        cv2.rectangle(
            img,
            (cx - box_size // 2, cy - box_size // 2),
            (cx + box_size // 2, cy + box_size // 2),
            (255, 0, 0),
            2,
        )
    return img


def main():
    pool_detector = PoolDetector(
        weights_path="./models/ResNet_acc=98.48_loss=0.04981_SGD_bs=64_ep=32_wd=0.0001.pth", device="cpu"
    )

    img_path = sys.argv[1]
    print("Processing image: ", img_path)
    img = cv2.imread(img_path)
    pool_coordinates = pool_detector.detect(img)
    print(pool_coordinates)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = plot_boxes(img, pool_coordinates)
    # Convert back to BGR for saving with OpenCV
    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    cv2.imwrite(sys.argv[2], img)
    plt.imshow(img)
    plt.figure()
    plt.show()
    input()


if __name__ == "__main__":
    main()
