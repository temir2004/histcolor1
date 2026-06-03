import cv2
import numpy as np
import random

def add_noise(image, sigma_range=(5,30)):
    sigma = random.uniform(*sigma_range)
    noise = np.random.normal(0, sigma, image.shape).astype(np.uint8)
    return cv2.add(image, noise)

def add_scratches(image, max_scratches=5):
    h, w = image.shape[:2]
    result = image.copy()
    for _ in range(random.randint(0, max_scratches)):
        x1 = random.randint(0, w)
        y1 = random.randint(0, h)
        x2 = random.randint(0, w)
        y2 = random.randint(0, h)
        thickness = random.randint(1,3)
        color = tuple(random.randint(0,255) for _ in range(3))
        cv2.line(result, (x1,y1), (x2,y2), color, thickness)
    return result

def sepia_tone(image):
    sepia_filter = np.array([[0.272, 0.534, 0.131],
                             [0.349, 0.686, 0.168],
                             [0.393, 0.769, 0.189]])
    image = image.astype(np.float32)
    sepia = cv2.transform(image, sepia_filter)
    sepia = np.clip(sepia, 0, 255).astype(np.uint8)
    return sepia

def blur(image, radius_range=(1,3)):
    ksize = random.choice([3,5])
    return cv2.GaussianBlur(image, (ksize,ksize), random.uniform(*radius_range))

def reduce_dynamic_range(image, factor_range=(0.5,0.9)):
    factor = random.uniform(*factor_range)
    return np.clip(image * factor, 0, 255).astype(np.uint8)

def historical_augmentation(image):
    # image: RGB numpy array uint8
    aug_type = random.choice(['noise', 'scratches', 'sepia', 'blur', 'dynamic', 'composite'])
    if aug_type == 'noise':
        image = add_noise(image)
    elif aug_type == 'scratches':
        image = add_scratches(image)
    elif aug_type == 'sepia':
        image = sepia_tone(image)
    elif aug_type == 'blur':
        image = blur(image)
    elif aug_type == 'dynamic':
        image = reduce_dynamic_range(image)
    else:  # composite
        if random.random() > 0.5:
            image = add_noise(image)
        if random.random() > 0.5:
            image = add_scratches(image)
        if random.random() > 0.5:
            image = sepia_tone(image)
        if random.random() > 0.5:
            image = blur(image)
        if random.random() > 0.5:
            image = reduce_dynamic_range(image)
    return image