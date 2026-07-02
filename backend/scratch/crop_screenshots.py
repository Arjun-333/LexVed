import os
from PIL import Image

screenshot_dir = "/home/arjun/Desktop/LexVed/results_screenshots"
crop_pixels = 38

for filename in os.listdir(screenshot_dir):
    if filename.lower().endswith(".png"):
        filepath = os.path.join(screenshot_dir, filename)
        try:
            with Image.open(filepath) as img:
                width, height = img.size
                print(f"Original size of {filename}: {width}x{height}")
                
                # Bounding box for crop: (left, top, right, bottom)
                box = (crop_pixels, crop_pixels, width - crop_pixels, height - crop_pixels)
                cropped_img = img.crop(box)
                
                # Save losslessly back to the same path
                cropped_img.save(filepath, format="PNG", optimize=True)
                print(f"Cropped and saved {filename} to size: {cropped_img.size}")
        except Exception as e:
            print(f"Error processing {filename}: {e}")
