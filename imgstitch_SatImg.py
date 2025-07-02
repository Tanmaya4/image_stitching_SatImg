import cv2
import os
import matplotlib.pyplot as plt
import rasterio
import numpy as np

def load_images(image_dir):
    """
    Loads all .tif images from a directory, converting them to a format
    suitable for OpenCV's stitcher.
    """
    images = []
    print("Loading images...")
    # Ensure a consistent processing order
    for filename in sorted(os.listdir(image_dir)):
        if filename.endswith('.tif'):
            path = os.path.join(image_dir, filename)
            try:
                with rasterio.open(path) as src:
                    # The stitcher works best with 8-bit images.
                    # We read the first 3 bands (assuming RGB) and normalize to 0-255.
                    img_f = src.read([1, 2, 3]).astype(np.float32)
                    img_8bit = cv2.normalize(img_f, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)
                    
                    # Rasterio reads as (channels, height, width), convert to (height, width, channels)
                    img_hwc = np.transpose(img_8bit, (1, 2, 0))
                    
                    # The stitcher expects BGR format, so convert from RGB
                    img_bgr = cv2.cvtColor(img_hwc, cv2.COLOR_RGB2BGR)
                    
                    images.append(img_bgr)
            except Exception as e:
                print(f"Could not load or process image {filename}: {e}")

    print(f"Loaded {len(images)} images.")
    return images

def main():
    image_dir = r'IMAGE_PATH'

    if not os.path.isdir(image_dir):
        print(f"Error: Directory not found at {image_dir}")
        return

    images = load_images(image_dir)
    if len(images) < 2:
        print("Error: Need at least two images to stitch.")
        return

    print("Attempting to stitch all images with OpenCV's Stitcher...")
    # Use the High-level Stitcher class
    stitcher = cv2.Stitcher.create(cv2.Stitcher_PANORAMA)
    status, pano = stitcher.stitch(images)

    if status == cv2.Stitcher_OK:
        print("Stitching successful!")
        # Save the result
        output_filename = "stitched_panorama.png"
        cv2.imwrite(output_filename, pano)
        print(f"Panorama saved to {output_filename}")

        # Display the result
        plt.figure(figsize=(20, 10))
        # Convert BGR from OpenCV to RGB for Matplotlib
        plt.imshow(cv2.cvtColor(pano, cv2.COLOR_BGR2RGB))
        plt.title("Stitched Panorama")
        plt.axis('off')
        plt.show()
    else:
        # Provide more descriptive error messages
        print(f"Stitching failed with status code: {status}")
        if status == cv2.Stitcher_ERR_NEED_MORE_IMGS:
            print("Error: Not enough keypoints were detected in the images to create a confident match. The images may be too different or of low quality.")
        elif status == cv2.Stitcher_ERR_HOMOGRAPHY_EST_FAIL:
            print("Error: Homography estimation failed. The images might not overlap enough, or the scene is not flat.")
        else:
            print("An unknown stitching error occurred.")

if __name__ == '__main__':
    main()






