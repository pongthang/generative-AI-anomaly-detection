import cv2
import numpy as np
# import matplotlib.pyplot as plt
#== Parameters =======================================================================
BLUR = 21
CANNY_THRESH_1 = 10
CANNY_THRESH_2 = 200
MASK_DILATE_ITER = 10
MASK_ERODE_ITER = 10

def get_mask(contrasted=None,canny_thr1 = 7,canny_thr2=20):
    blurred = cv2.GaussianBlur(contrasted, (3, 3), 0)
    edges = cv2.Canny(blurred, 7, 20)
    edges = cv2.dilate(edges, None)
    edges = cv2.erode(edges, None)
    #-- Find contours in edges, sort by area ---------------------------------------------
    contour_info = []
    # _, contours, _ = cv2.findContours(edges, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)
    # Previously, for a previous version of cv2, this line was: 
    contours, _ = cv2.findContours(edges, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)
    # Thanks to notes from commenters, I've updated the code but left this note
    for c in contours:
        contour_info.append((
            c,
            cv2.isContourConvex(c),
            cv2.contourArea(c),
        ))
    contour_info = sorted(contour_info, key=lambda c: c[2], reverse=True)
    max_contour = contour_info[0]
    mask = np.zeros(edges.shape)
    cv2.fillConvexPoly(mask, max_contour[0], (255))
    #-- Smooth mask, then blur it --------------------------------------------------------
    mask = cv2.dilate(mask, None, iterations=MASK_DILATE_ITER)
    mask = cv2.erode(mask, None, iterations=MASK_ERODE_ITER)
    mask = cv2.GaussianBlur(mask, (BLUR, BLUR), 0)
    # mask_stack = np.dstack([mask]*3)    # Create 3-channel alpha mask
    #-- Blend masked img into MASK_COLOR background --------------------------------------
    # mask_stack  = mask_stack.astype('float32') / 255.0          # Use float matrices, 
    # img         = img.astype('float32') / 255.0                 #  for easy blending
    mask_stack = mask/255.0
    # gray = gray/255.0
    masked = (mask_stack * contrasted) + ((1-mask_stack)) # Blend
    masked = (masked).astype('uint8') 
    # masked = masked/255
    return masked