
import os
import random
from PIL import Image, ImageEnhance, ImageFilter

# ==========================
# CONFIGURATION
# ==========================
train_dir = r"C:\Users\Dell\Downloads\archive\bd_traffic_signs\test"
TARGET_SAMPLES = 250          # total final images per class
BLUR_SAMPLES   = 20           # how many of the TARGET_SAMPLES should be blurred

# ==========================
# AUGMENTATION FUNCTIONS
# ==========================
def random_rotation(image):
    """Random rotation between -15 and +15 degrees."""
    return image.rotate(random.randint(-15, 15))

def random_saturation(image):
    """Adjust saturation by ±40%."""
    enhancer = ImageEnhance.Color(image)
    factor = random.uniform(0.6, 1.4)
    return enhancer.enhance(factor)

def random_brightness(image):
    """Adjust brightness by ±30%."""
    enhancer = ImageEnhance.Brightness(image)
    factor = random.uniform(0.7, 1.3)
    return enhancer.enhance(factor)

def random_exposure(image):
    """Adjust exposure (contrast) by ±13%."""
    enhancer = ImageEnhance.Contrast(image)
    factor = random.uniform(0.87, 1.13)
    return enhancer.enhance(factor)

AUGMENTATIONS = [
    random_rotation,
    random_saturation,
    random_brightness,
    random_exposure
]

# ==========================
# MAIN
# ==========================
print("Starting dataset balancing and blur augmentation...")

class_folders = [
    os.path.join(train_dir, d)
    for d in os.listdir(train_dir)
    if os.path.isdir(os.path.join(train_dir, d))
]

for folder in class_folders:
    image_files = [
        os.path.join(folder, f)
        for f in os.listdir(folder)
        if f.lower().endswith(('.jpg', '.jpeg', '.png'))
    ]
    current_count = len(image_files)

    # ---------------------
    # Step 1: Undersample if needed
    # ---------------------
    if current_count > TARGET_SAMPLES:
        to_remove = random.sample(image_files, current_count - TARGET_SAMPLES)
        for p in to_remove:
            os.remove(p)
        print(f"[{os.path.basename(folder)}] Reduced to {TARGET_SAMPLES} images.")
        # refresh file list
        image_files = [
            os.path.join(folder, f)
            for f in os.listdir(folder)
            if f.lower().endswith(('.jpg', '.jpeg', '.png'))
        ]
        current_count = len(image_files)

    # ---------------------
    # Step 2: Oversample with random augmentations (except blur)
    # ---------------------
    while current_count < (TARGET_SAMPLES - BLUR_SAMPLES):
        img_path = random.choice(image_files)
        image = Image.open(img_path).convert("RGB")

        # Randomly decide how many augmentations to apply
        num_augs = random.choices([0,1,2,3,4], weights=[0.3,0.1,0.15,0.3,0.15])[0]
        selected = random.sample(AUGMENTATIONS, num_augs)

        for aug in selected:
            image = aug(image)

        new_path = os.path.join(folder, f"aug_{random.randint(10000,99999)}.jpg")
        image.save(new_path)
        current_count += 1

    # Refresh list after normal augmentation
    image_files = [
        os.path.join(folder, f)
        for f in os.listdir(folder)
        if f.lower().endswith(('.jpg', '.jpeg', '.png'))
    ]

    # ---------------------
    # Step 3: Add BLUR_SAMPLES blurred images
    # ---------------------
    added_blur = 0
    while current_count < TARGET_SAMPLES:
        img_path = random.choice(image_files)
        image = Image.open(img_path).convert("RGB")

        blurred = image.filter(ImageFilter.GaussianBlur(3))
        new_path = os.path.join(folder, f"blur_{random.randint(10000,99999)}.jpg")
        blurred.save(new_path)

        current_count += 1
        added_blur += 1

    print(f"[{os.path.basename(folder)}] Balanced to {TARGET_SAMPLES} images "
          f"(including {BLUR_SAMPLES} blurred).")

print("✅ Dataset balancing and blur augmentation complete!")
