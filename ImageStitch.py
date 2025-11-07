import os
import random
import cv2
import numpy as np
import matplotlib.pyplot as plt
import imageio
import warnings
warnings.filterwarnings("ignore")

cv2.ocl.setUseOpenCL(False)

FEATURE_ALGO = "sift"       
MATCH_METHOD = "knn"        
DRAW_MATCHES_LIMIT = 100
RATIO_TEST = 0.75          
REPROJ_THRESH = 4.0        
OUTPUT_DIR = "./output"
os.makedirs(OUTPUT_DIR, exist_ok=True)




def read_image(path, as_gray=False):
    img = cv2.imread(path)
    if img is None:
        raise FileNotFoundError(f"Image not found: {path}")
    if as_gray:
        return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    return img

def get_descriptor(descriptor_name="sift"):
    descriptor_name = descriptor_name.lower()
    if descriptor_name == "sift":
        if hasattr(cv2, "SIFT_create"):
            return cv2.SIFT_create()
        else:
            descriptor_name = "orb"
    if descriptor_name == "surf":
        if hasattr(cv2, "xfeatures2d") and hasattr(cv2.xfeatures2d, "SURF_create"):
            return cv2.xfeatures2d.SURF_create()
        else:
            descriptor_name = "orb"
    if descriptor_name == "brisk":
        return cv2.BRISK_create()
    return cv2.ORB_create(nfeatures=5000)

def detect_and_compute(image_gray, method=FEATURE_ALGO):
    descriptor = get_descriptor(method)
    kps, desc = descriptor.detectAndCompute(image_gray, None)
    return kps, desc



def create_matcher(descriptor_name="sift", crossCheck=False):
    descriptor_name = descriptor_name.lower()
    if descriptor_name in ("sift", "surf"):
        norm = cv2.NORM_L2
    else:
        norm = cv2.NORM_HAMMING
    return cv2.BFMatcher(norm, crossCheck=crossCheck)

def match_knn(desc1, desc2, method_name, ratio=RATIO_TEST):
    bf = create_matcher(method_name, crossCheck=False)
    raw = bf.knnMatch(desc1, desc2, k=2)
    good = []
    for pair in raw:
        if len(pair) != 2:
            continue
        m, n = pair
        if m.distance < ratio * n.distance:
            good.append(m)
    return good

def match_bruteforce(desc1, desc2, method_name):
    bf = create_matcher(method_name, crossCheck=True)
    matches = bf.match(desc1, desc2)
    matches = sorted(matches, key=lambda x: x.distance)
    return matches


def draw_keypoints_subplot(img_bgr, kps, title, ax):
    img_kp = cv2.drawKeypoints(img_bgr, kps, None, flags=cv2.DRAW_MATCHES_FLAGS_DEFAULT)
    img_kp_rgb = cv2.cvtColor(img_kp, cv2.COLOR_BGR2RGB)
    ax.imshow(img_kp_rgb)
    ax.set_title(title)
    ax.axis("off")

def draw_matches(img1_bgr, kps1, img2_bgr, kps2, matches, max_draw=DRAW_MATCHES_LIMIT):
    draw_count = min(len(matches), max_draw)
    if len(matches) > draw_count:
        sample_matches = random.sample(matches, draw_count)
    else:
        sample_matches = matches[:draw_count]
    out = cv2.drawMatches(
        img1_bgr, kps1, img2_bgr, kps2, sample_matches, None,
        flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS
    )
    return out



def find_homography_and_status(kps1, kps2, matches, reprojThresh=REPROJ_THRESH):
    if len(matches) < 4:
        return None, None
    pts1 = np.float32([kps1[m.queryIdx].pt for m in matches])
    pts2 = np.float32([kps2[m.trainIdx].pt for m in matches])
    H, status = cv2.findHomography(pts1, pts2, cv2.RANSAC, reprojThresh)
    return H, status

def stitch_images(img_train_bgr, img_query_bgr, H):
    h1, w1 = img_train_bgr.shape[:2]
    h2, w2 = img_query_bgr.shape[:2]
    pano_width = w1 + w2
    pano_height = max(h1, h2)

    warped = cv2.warpPerspective(img_train_bgr, H, (pano_width, pano_height))
    result = warped.copy()
    result[0:h2, 0:w2] = img_query_bgr

    mask_query = np.zeros((pano_height, pano_width), dtype=np.uint8)
    mask_query[0:h2, 0:w2] = 255
    mask_warped = cv2.cvtColor(warped, cv2.COLOR_BGR2GRAY) > 0
    overlap = (mask_query > 0) & (mask_warped > 0)
    if overlap.any():
        overlap_idxs = np.where(overlap)
        for y, x in zip(*overlap_idxs):
            alpha = 0.5
            result[y, x] = (img_query_bgr[y, x] * alpha + warped[y, x] * (1 - alpha)).astype(np.uint8)
    gray = cv2.cvtColor(result, cv2.COLOR_BGR2GRAY)
    cols = np.where(gray.max(axis=0) > 0)[0]
    if cols.size:
        right = cols[-1] + 1
        result = result[:, :right]
    return result


if __name__ == "__main__":
    train_path = "train.png"
    query_path = "query.png"

    img_train = read_image(train_path, as_gray=False)
    img_query = read_image(query_path, as_gray=False)
    img_train_gray = cv2.cvtColor(img_train, cv2.COLOR_BGR2GRAY)
    img_query_gray = cv2.cvtColor(img_query, cv2.COLOR_BGR2GRAY)

    kps_train, desc_train = detect_and_compute(img_train_gray, method=FEATURE_ALGO)
    kps_query, desc_query = detect_and_compute(img_query_gray, method=FEATURE_ALGO)

    if desc_train is None or desc_query is None:
        raise RuntimeError("Could not detect enough keypoints.")

    # Visualize keypoints
    fig, axes = plt.subplots(1, 2, figsize=(16, 8))
    draw_keypoints_subplot(img_query, kps_query, "Query Image Keypoints", axes[0])
    draw_keypoints_subplot(img_train, kps_train, "Train Image Keypoints", axes[1])
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, f"{FEATURE_ALGO}_keypoints.jpg"), dpi=300)
    plt.close(fig)

    # Matching
    if MATCH_METHOD == "knn":
        matches = match_knn(desc_train, desc_query, FEATURE_ALGO, ratio=RATIO_TEST)
    else:
        matches = match_bruteforce(desc_train, desc_query, FEATURE_ALGO)

    match_img = draw_matches(img_train, kps_train, img_query, kps_query, matches)
    plt.figure(figsize=(18, 8))
    plt.imshow(cv2.cvtColor(match_img, cv2.COLOR_BGR2RGB))
    plt.axis("off")
    plt.savefig(os.path.join(OUTPUT_DIR, f"{MATCH_METHOD}_matches.jpg"), dpi=300)
    plt.close()

    H, status = find_homography_and_status(kps_train, kps_query, matches)
    if H is None:
        raise RuntimeError("Homography could not be computed.")

    panorama = stitch_images(img_train, img_query, H)
    plt.figure(figsize=(20, 8))
    plt.imshow(cv2.cvtColor(panorama, cv2.COLOR_BGR2RGB))
    plt.axis("off")
    out_path = os.path.join(OUTPUT_DIR, "stitched_panorama.jpg")
    plt.savefig(out_path, dpi=300)
    plt.close()

    imageio.imwrite(os.path.join(OUTPUT_DIR, "stitched_panorama_io.jpg"), cv2.cvtColor(panorama, cv2.COLOR_BGR2RGB))
    print(" Stitching Complete! Saved to:", OUTPUT_DIR)
