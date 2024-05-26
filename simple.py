import torch
import cv2
import numpy as np

# Load YOLOv5 model
model = torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=True)

# Load the class names
class_names = model.names

# Read the input image
image_path = 'cowboys.png'  # Replace with your image path
image = cv2.imread(image_path)

if image is None:
    print(f"Error: Could not load image at {image_path}")
    exit()

# Perform inference
results = model(image)

# Print detailed debugging information
print("Inference Results:")
print(results)

# Get the predictions
predictions = results.pred[0]

# Print the predictions
print("\nPredictions:")
for i, pred in enumerate(predictions):
    print(f"Prediction {i + 1}:")
    print(f"  Coordinates: {pred[:4]}")
    print(f"  Confidence: {pred[4]}")
    print(f"  Class: {pred[5]} ({class_names[int(pred[5])]})")

# Draw bounding boxes on the image
for pred in predictions:
    x1, y1, x2, y2, conf, cls_id = pred
    label = f"{class_names[int(cls_id)]} {conf:.2f}"
    
    # Convert coordinates to integers
    x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
    
    # Draw the bounding box
    cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)
    
    # Put the label
    cv2.putText(image, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

# Display the output image
cv2.imshow('Image with Cowboy Hat Detection', image)
cv2.waitKey(0)
cv2.destroyAllWindows()

# Save the output image
output_image_path = 'cowboys_detection.jpg'
cv2.imwrite(output_image_path, image)