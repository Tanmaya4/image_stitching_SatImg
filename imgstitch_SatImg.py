# import cv2
# import os
# import matplotlib.pyplot as plt
# import rasterio
# import numpy as np

# def load_images(image_dir):
#     """
#     Loads all .tif images from a directory, converting them to a format
#     suitable for OpenCV's stitcher.
#     """
#     images = []
#     print("Loading images...")
#     # Ensure a consistent processing order
#     for filename in sorted(os.listdir(image_dir)):
#         if filename.endswith('.tif'):
#             path = os.path.join(image_dir, filename)
#             try:
#                 with rasterio.open(path) as src:
#                     # The stitcher works best with 8-bit images.
#                     # We read the first 3 bands (assuming RGB) and normalize to 0-255.
#                     img_f = src.read([1, 2, 3]).astype(np.float32)
#                     img_8bit = cv2.normalize(img_f, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)
                    
#                     # Rasterio reads as (channels, height, width), convert to (height, width, channels)
#                     img_hwc = np.transpose(img_8bit, (1, 2, 0))
                    
#                     # The stitcher expects BGR format, so convert from RGB
#                     img_bgr = cv2.cvtColor(img_hwc, cv2.COLOR_RGB2BGR)
                    
#                     images.append(img_bgr)
#             except Exception as e:
#                 print(f"Could not load or process image {filename}: {e}")

#     print(f"Loaded {len(images)} images.")
#     return images

# def main():
#     image_dir = r'C:\Users\TANMAYA\Downloads\Udemy CV\Satellite-image-stitcher\images-data'

#     if not os.path.isdir(image_dir):
#         print(f"Error: Directory not found at {image_dir}")
#         return

#     images = load_images(image_dir)
#     if len(images) < 2:
#         print("Error: Need at least two images to stitch.")
#         return

#     print("Attempting to stitch all images with OpenCV's Stitcher...")
#     # Use the High-level Stitcher class
#     stitcher = cv2.Stitcher.create(cv2.Stitcher_PANORAMA)
#     status, pano = stitcher.stitch(images)

#     if status == cv2.Stitcher_OK:
#         print("Stitching successful!")
#         # Save the result
#         output_filename = "stitched_panorama.png"
#         cv2.imwrite(output_filename, pano)
#         print(f"Panorama saved to {output_filename}")

#         # Display the result
#         plt.figure(figsize=(20, 10))
#         # Convert BGR from OpenCV to RGB for Matplotlib
#         plt.imshow(cv2.cvtColor(pano, cv2.COLOR_BGR2RGB))
#         plt.title("Stitched Panorama")
#         plt.axis('off')
#         plt.show()
#     else:
#         # Provide more descriptive error messages
#         print(f"Stitching failed with status code: {status}")
#         if status == cv2.Stitcher_ERR_NEED_MORE_IMGS:
#             print("Error: Not enough keypoints were detected in the images to create a confident match. The images may be too different or of low quality.")
#         elif status == cv2.Stitcher_ERR_HOMOGRAPHY_EST_FAIL:
#             print("Error: Homography estimation failed. The images might not overlap enough, or the scene is not flat.")
#         else:
#             print("An unknown stitching error occurred.")

# if __name__ == '__main__':
#     main()













import rasterio
import os
import numpy as np
import cv2
import matplotlib.pyplot as plt

# Qiskit for Quantum Optimization
from qiskit_algorithms import QAOA
from qiskit_algorithms.optimizers import COBYLA
from qiskit.primitives import Sampler
from qiskit_aer.backends import AerSimulator
from qiskit_optimization.problems import QuadraticProgram
from qiskit_optimization.algorithms import MinimumEigenOptimizer

def load_images(image_dir):
    """
    Loads all .tif images from a directory, converting them to 8-bit RGB.
    """
    images = []
    print("Loading images...")
    for filename in sorted(os.listdir(image_dir)):
        if filename.endswith('.tif'):
            path = os.path.join(image_dir, filename)
            try:
                with rasterio.open(path) as src:
                    img_f = src.read([1, 2, 3]).astype(np.float32)
                    img_8bit = cv2.normalize(img_f, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)
                    img_hwc = np.transpose(img_8bit, (1, 2, 0))
                    images.append(img_hwc) # Keep as RGB for processing
            except Exception as e:
                print(f"Could not load or process image {filename}: {e}")
    print(f"Loaded {len(images)} images.")
    return images

def detect_and_compute_features(image):
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    orb = cv2.ORB_create(nfeatures=1000)
    keypoints, descriptors = orb.detectAndCompute(gray, None)
    return keypoints, descriptors

def match_features(desc1, desc2):
    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
    matches = bf.match(desc1, desc2)
    matches = sorted(matches, key=lambda x: x.distance)
    return matches

def quantum_optimize_feature_matches(matches):
    """
    Uses QAOA to select an optimal subset of feature matches.
    """
    print("Optimizing feature matches with QAOA...")
    qp = QuadraticProgram("Feature Match Optimization")
    for i, _ in enumerate(matches):
        qp.binary_var(f"x{i}")

    # Objective: Minimize the sum of distances for the selected matches.
    # A simple objective for demonstration purposes.
    objective = {f"x{i}": m.distance for i, m in enumerate(matches)}
    qp.minimize(linear=objective)
    
    # If we simply minimize, the ground state will be to select no matches (all zeros).
    # To make it useful, we add a constraint to select a certain number of matches.
    # Let's aim to select about 25% of the best original matches, but no more than 50.
    num_to_select = min(50, len(matches) // 4)
    if num_to_select < 10: # ensure a minimum number for transform estimation
        num_to_select = min(10, len(matches))
        
    qp.linear_constraint(linear={f"x{i}": 1 for i, _ in enumerate(matches)}, sense='==', rhs=num_to_select, name='num_matches')

    # The default Sampler uses a local statevector simulator.
    sampler = Sampler()
    qaoa = QAOA(sampler=sampler, optimizer=COBYLA(), reps=1)
    optimizer = MinimumEigenOptimizer(qaoa)
    result = optimizer.solve(qp)

    selected_indices = [i for i, x in enumerate(result.x) if x > 0.5]
    selected_matches = [matches[i] for i in selected_indices]
    
    print(f"Quantum optimizer selected {len(selected_matches)} matches.")
    return selected_matches

def estimate_transform(kp1, kp2, matches):
    src_pts = np.float32([kp1[m.queryIdx].pt for m in matches])
    dst_pts = np.float32([kp2[m.trainIdx].pt for m in matches])
    matrix, _ = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)
    return matrix

def stitch_images(base_img, next_img, matrix):
    h1, w1 = base_img.shape[:2]
    h2, w2 = next_img.shape[:2]
    
    # Get corners of both images
    corners1 = np.float32([[0,0], [0,h1], [w1,h1], [w1,0]]).reshape(-1,1,2)
    corners2 = np.float32([[0,0], [0,h2], [w2,h2], [w2,0]]).reshape(-1,1,2)
    
    # Warp corners of the second image to find the bounding box of the combined image
    warped_corners2 = cv2.perspectiveTransform(corners2, matrix)
    all_corners = np.concatenate((corners1, warped_corners2), axis=0)
    
    x_min, y_min = np.int32(all_corners.min(axis=0).ravel() - 0.5)
    x_max, y_max = np.int32(all_corners.max(axis=0).ravel() + 0.5)
    
    # Create a translation matrix to move the stitched image into the visible frame
    translation_dist = [-x_min, -y_min]
    H_trans = np.array([[1, 0, translation_dist[0]], [0, 1, translation_dist[1]], [0, 0, 1]])
    
    # Warp the second image and place the first image onto the canvas
    output_w = x_max - x_min
    output_h = y_max - y_min
    warped_img = cv2.warpPerspective(next_img, H_trans.dot(matrix), (output_w, output_h))
    
    # Place the first image on the translated canvas
    result = warped_img
    result[translation_dist[1]:h1+translation_dist[1], translation_dist[0]:w1+translation_dist[0]] = base_img
    return result

def main():
    image_dir = r'C:\Users\TANMAYA\Downloads\Udemy CV\Satellite-image-stitcher\images-data'

    if not os.path.isdir(image_dir):
        print(f"Error: Directory not found at {image_dir}")
        return

    images = load_images(image_dir)
    if len(images) < 2:
        print("Error: Need at least two images to stitch.")
        return

    panorama = images[0]

    for i in range(len(images) - 1):
        next_img = images[i+1]
        print(f"\nStitching image {i+2}/{len(images)}...")
        
        kp1, desc1 = detect_and_compute_features(panorama)
        kp2, desc2 = detect_and_compute_features(next_img)

        if desc1 is None or desc2 is None or len(desc1) == 0 or len(desc2) == 0:
            print("Could not find features in one of the images. Skipping.")
            continue

        matches = match_features(desc1, desc2)
        print(f"Found {len(matches)} initial matches.")

        if len(matches) < 10:
            print("Not enough initial matches found. Skipping.")
            continue
            
        optimized_matches = quantum_optimize_feature_matches(matches)
        
        if len(optimized_matches) < 4: # RANSAC needs at least 4 points
            print("Quantum optimizer returned too few matches. Skipping.")
            continue

        matrix = estimate_transform(kp1, kp2, optimized_matches)
        if matrix is None:
            print("Could not estimate transform. Skipping.")
            continue

        panorama = stitch_images(panorama, next_img, matrix)

    print("\nStitching complete.")
    plt.figure(figsize=(20, 10))
    plt.imshow(panorama)
    plt.title("Quantum-Optimized Stitched Panorama")
    plt.axis('off')
    plt.show()

if __name__ == '__main__':
    main()