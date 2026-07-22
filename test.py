import cv2
import numpy as np

def create_canvas():
    return np.ones((200, 400), dtype=np.uint8) * 255

# 1. Genuine Signature
genuine = create_canvas()
cv2.ellipse(genuine, (150, 100), (80, 30), 15, 0, 360, (0,), 4)
cv2.line(genuine, (80, 120), (320, 80), (0,), 5)
cv2.putText(genuine, "John Doe", (100, 110), cv2.FONT_HERSHEY_SCRIPT_SIMPLEX, 1.8, (0,), 4)
cv2.imwrite('genuine_test.png', genuine)

# 2. Stronger Forgery Sample (Noticeably different stroke structure)
forged = create_canvas()
# Different loop shape & position
cv2.ellipse(forged, (180, 90), (60, 45), -20, 0, 360, (0,), 1)  
# Different stroke angle & thickness
cv2.line(forged, (50, 160), (350, 60), (0,), 2)                
# Using FONT_HERSHEY_COMPLEX to create clear structural difference
cv2.putText(forged, "John Doe", (80, 125), cv2.FONT_HERSHEY_COMPLEX, 1.3, (0,), 2)
cv2.imwrite('forged_test.png', forged)

print("New test images generated! Upload genuine_test.png vs forged_test.png now.")