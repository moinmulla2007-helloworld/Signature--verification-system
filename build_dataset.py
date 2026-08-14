import os
import cv2
import numpy as np
import random

genuine_dir = os.path.join("dataset", "genuine")
forged_dir = os.path.join("dataset", "forged")

os.makedirs(genuine_dir, exist_ok=True)
os.makedirs(forged_dir, exist_ok=True)

for f in os.listdir(genuine_dir): os.remove(os.path.join(genuine_dir, f))
for f in os.listdir(forged_dir): os.remove(os.path.join(forged_dir, f))

print("Generating 100 Synthetic Genuine and 100 Synthetic HARD Forgeries...")

# FORCING THE AI TO WORK: Both classes use the exact same cursive fonts
fonts = [cv2.FONT_HERSHEY_SCRIPT_SIMPLEX, cv2.FONT_HERSHEY_SCRIPT_COMPLEX]

# 1. Generate Genuine Signatures
for i in range(100):
    img = np.ones((200, 400), dtype=np.uint8) * 255
    font = random.choice(fonts)
    
    # Standard human variations
    thickness = random.randint(2, 4) 
    angle = random.randint(10, 20)
    
    cv2.ellipse(img, (150 + random.randint(-3, 3), 100 + random.randint(-3, 3)), 
                (80 + random.randint(-3, 3), 30 + random.randint(-3, 3)), angle, 0, 360, (0,), thickness)
    cv2.line(img, (70 + random.randint(-5, 5), 130 + random.randint(-5, 5)), 
             (330 + random.randint(-5, 5), 70 + random.randint(-5, 5)), (0,), thickness)
    cv2.putText(img, "John Doe", (90 + random.randint(-3, 3), 115 + random.randint(-3, 3)), font, 1.7, (0,), thickness)
    
    cv2.imwrite(os.path.join(genuine_dir, f"gen_{i:03d}.png"), img)

# 2. Generate HARD Forged Signatures
for i in range(100):
    img = np.ones((200, 400), dtype=np.uint8) * 255
    font = random.choice(fonts) # Uses the same cursive fonts
    
    # Overlapping thickness so the network can't cheat
    thickness = random.randint(2, 4) 
    
    # The forger tries to mimic the angle, but slightly misses
    angle = random.randint(5, 25) 
    
    # The forger messes up the structural geometry (wrong ellipse size/position)
    cv2.ellipse(img, (160 + random.randint(-5, 5), 90 + random.randint(-5, 5)), 
                (70 + random.randint(-5, 5), 40 + random.randint(-5, 5)), angle, 0, 360, (0,), thickness)
    
    # The underline is drawn slightly too high or short
    cv2.line(img, (80 + random.randint(-5, 5), 120 + random.randint(-5, 5)), 
             (310 + random.randint(-5, 5), 80 + random.randint(-5, 5)), (0,), thickness)
             
    # The text is spaced slightly differently
    cv2.putText(img, "John Doe", (95 + random.randint(-5, 5), 110 + random.randint(-5, 5)), font, 1.6, (0,), thickness)
    
    cv2.imwrite(os.path.join(forged_dir, f"forg_{i:03d}.png"), img)

print("Hard negative dataset successfully rebuilt!")