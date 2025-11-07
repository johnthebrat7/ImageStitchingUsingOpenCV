ImageStitchingUsingOpenCV
# Image Stitching with OpenCV

Robust panorama stitching: feature detection (SIFT/ORB), matching (FLANN/BF), homography (RANSAC), warping (planar/cylindrical/spherical), and blending (feather/multiband) with exposure compensation. CLI, examples, and debug views included.

## Features
- SIFT or ORB keypoints (configurable nfeatures)
- FLANN/BF matching + Lowe’s ratio test (+ optional cross‑check)
- Robust homography with RANSAC; inlier ratio & reprojection error reported
- Planar/cylindrical/spherical warps; automatic canvas sizing & border handling
- Feather or multiband pyramid blending; exposure compensation & seam finding
- Debug visualizations (matches, masks, overlap heatmaps) and metrics
- Works for pairwise or multi‑image panoramas

## Getting Started
### Requirements
- Python >= 3.9
- opencv-contrib-python >= 4.8
- numpy, scipy, scikit-image (optional), matplotlib (optional)

```bash
```
pip install -r requirements.txt
# or
pip install opencv-contrib-python numpy scipy scikit-image matplotlib


Data
samples/
  pano1/
    01.jpg
    02.jpg
    03.jpg

USAGE

# Typical usage (edit flags to match your script)
python stitch.py \
  --input "samples/pano1/*.jpg" \
  --detector SIFT \
  --matcher FLANN \
  --proj cylindrical \
  --blend multiband \
  --work-megapix 0.6 \
  --out outputs/pano1.jpg \
  --viz


Common flags (replace with yours):

--detector SIFT|ORB
--matcher FLANN|BF
--proj planar|cylindrical|spherical
--blend feather|multiband
--work-megapix <0..1> downscales for speed (e.g., 0.6)
--save-debug to dump matches/overlays

How It Works (Pipeline)
Read & orient images (EXIF)
Optional downscale to work resolution
Detect & describe features (SIFT/ORB)
Match features (FLANN/BF) + ratio test
Estimate homography with RANSAC
Warp to common canvas (planar/cyl/spherical)
Exposure compensation + seam finding
Feather/multiband blending
Auto‑crop black borders
