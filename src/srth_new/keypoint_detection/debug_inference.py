from PIL import Image
from ultralytics import YOLO

# Load a model
model = YOLO("/home/grayson/surpass/srth-new/src/srth_new/keypoint_detection/runs/pose/train-2/weights/best.pt")  # pretrained YOLO26n model

# Run batched inference on a list of images
img = Image.open("/mnt/sda1/surpass_data/Cholecystectomy/Tissue#21/Philip/unzipping/4_hook_tissue/20260608-165848-732553/left_img_dir/frame000044_left_1780952330_169463878.jpg")

results = model([img])  # return a list of Results objects

# Process results list
for result in results:
    boxes = result.boxes  # Boxes object for bounding box outputs
    masks = result.masks  # Masks object for segmentation masks outputs
    keypoints = result.keypoints  # Keypoints object for pose outputs
    probs = result.probs  # Probs object for classification outputs
    obb = result.obb  # Oriented boxes object for OBB outputs
    result.show()  # display to screen
    result.save(filename="result.jpg")  # save to disk