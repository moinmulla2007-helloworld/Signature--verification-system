import os
import cv2
import numpy as np
import random

genuine_dir = os.path.join("dataset", "genuine")
forged_dir = os.path.join("dataset", "forged")

os.makedirs(genuine_dir, exist_ok=True)
os.makedirs(forged_dir, exist_ok=True)

# Clear old sample images
for f in os.listdir(genuine_dir): os.remove(os.path.join(genuine_dir, f))
for f in os.listdir(forged_dir): os.remove(os.path.join(forged_dir, f))

print("Generating 100 realistic Genuine and 100 Forged signature samples...")

fonts_genuine = [cv2.FONT_HERSHEY_SCRIPT_SIMPLEX, cv2.FONT_HERSHEY_SCRIPT_COMPLEX]
fonts_forged = [cv2.FONT_HERSHEY_SIMPLEX, cv2.FONT_HERSHEY_COMPLEX, cv2.FONT_HERSHEY_TRIPLEX, cv2.FONT_ITALIC]

# Generate Genuine Signatures (John Doe style with variations)
for i in range(100):
    img = np.ones((200, 400), dtype=np.uint8) * 255
    font = random.choice(fonts_genuine)
    thickness = random.randint(3, 5)
    angle = random.randint(10, 20)
    
    # Signature stroke elements
    cv2.ellipse(img, (150 + random.randint(-5, 5), 100 + random.randint(-5, 5)), 
                (80 + random.randint(-5, 5), 30 + random.randint(-5, 5)), angle, 0, 360, (0,), thickness)
    cv2.line(img, (70 + random.randint(-10, 10), 130 + random.randint(-10, 10)), 
             (330 + random.randint(-10, 10), 70 + random.randint(-10, 10)), (0,), thickness + 1)
    cv2.putText(img, "John Doe", (90 + random.randint(-5, 5), 115 + random.randint(-5, 5)), font, 1.7, (0,), thickness)
    
    cv2.imwrite(os.path.join(genuine_dir, f"gen_{i:03d}.png"), img)

# Generate Forged Signatures (Completely different stroke styles)
for i in range(100):
    img = np.ones((200, 400), dtype=np.uint8) * 255
    font = random.choice(fonts_forged)
    thickness = random.randint(1, 2)
    angle = random.randint(-30, -10)
    
    # Completely different structure
    cv2.ellipse(img, (200 + random.randint(-10, 10), 90 + random.randint(-10, 10)), 
                (50 + random.randint(-5, 5), 50 + random.randint(-5, 5)), angle, 0, 360, (0,), thickness)
    cv2.line(img, (40 + random.randint(-10, 10), 160 + random.randint(-10, 10)), 
             (360 + random.randint(-10, 10), 50 + random.randint(-10, 10)), (0,), thickness)
    cv2.putText(img, "John Doe", (70 + random.randint(-10, 10), 130 + random.randint(-10, 10)), font, 1.3, (0,), thickness)
    
    cv2.imwrite(os.path.join(forged_dir, f"forg_{i:03d}.png"), img)

print("Dataset successfully rebuilt!")