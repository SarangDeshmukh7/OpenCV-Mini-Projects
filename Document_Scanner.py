import sys
import cv2
import imutils
from itertools import combinations
import numpy as np
from numba import njit

import time


"""
Check whether two lines defined by their Hough angle are parallel.
"""
@njit
def are_parallel(θ1, θ2):
    return np.abs(θ1-θ2) < np.pi/8 or np.abs(np.abs(θ1-θ2)-np.pi) < np.pi/8


"""
Check parallelity relations between four lines.
Lines must be parallel in pairs: two parallel lines and two other parallel lines.
"""
@njit
def check_parallelity(θ):
    are_par_01 = are_parallel(θ[0], θ[1])
    are_par_02 = are_parallel(θ[0], θ[2])
    are_par_03 = are_parallel(θ[0], θ[3])
    are_par_12 = are_parallel(θ[1], θ[2])
    are_par_13 = are_parallel(θ[1], θ[3])
    are_par_23 = are_parallel(θ[2], θ[3])
                                                    
    # Check that lines are parallel in pairs.
    if (not (are_par_01 and are_par_23)
        and not (are_par_02 and are_par_13)
        and not (are_par_03 and are_par_12)
    ):
        return True

    # Check that lines of different pairs are not parallel.
    if ((are_par_01 and are_par_02)
        or (are_par_01 and are_par_03)
        or (are_par_01 and are_par_03)
        or (are_par_02 and are_par_03)
        or (are_par_12 and are_par_13)
    ):
        return True
    return False


"""
Find intersection between two Hough lines.
"""
@njit
def line_intersection(ρ1, ρ2, θ1, θ2):
    if np.abs(np.sin(θ1)) < 1e-3:
        x = ρ1/np.cos(θ1)
        y = (ρ2 - x*np.cos(θ2))/np.sin(θ2)
    elif np.abs(np.sin(θ2)) < 1e-3:
        x = ρ2/np.cos(θ2)
        y = (ρ1 - x*np.cos(θ1))/np.sin(θ1)
    else:
        x = (ρ1/np.sin(θ1) - ρ2/np.sin(θ2))/(1/np.tan(θ1) - 1/np.tan(θ2))
        y = (ρ1 - x*np.cos(θ1))/np.sin(θ1)
    return [x, y]


"""
Choose whether a set of four Hough lines are valid document edges.
Find the corners of the four Hough lines, check their parallelity
relations, find the area inside them and, if the area is bigger than
any other set of edges we have, update doc_corners and doc_edges
accordingly.
"""
@njit
def find_doc_edges(edges, doc_corners, doc_edges, max_area):
    ρ = edges[:, 0]
    θ = edges[:, 1]

    # Check that lines are parallel in pairs.
    if check_parallelity(θ):
       return max_area

    # Find corners.
    corners = np.zeros((4, 2))
    if are_parallel(θ[0], θ[1]) and are_parallel(θ[2], θ[3]):
        corners[0] = line_intersection(ρ[0], ρ[2], θ[0], θ[2])
        corners[1] = line_intersection(ρ[0], ρ[3], θ[0], θ[3])
        corners[2] = line_intersection(ρ[1], ρ[2], θ[1], θ[2])
        corners[3] = line_intersection(ρ[1], ρ[3], θ[1], θ[3])
    if are_parallel(θ[0], θ[2]) and are_parallel(θ[1], θ[3]):
        corners[0] = line_intersection(ρ[0], ρ[1], θ[0], θ[1])
        corners[1] = line_intersection(ρ[0], ρ[3], θ[0], θ[3])
        corners[2] = line_intersection(ρ[2], ρ[1], θ[2], θ[1])
        corners[3] = line_intersection(ρ[2], ρ[3], θ[2], θ[3])
    if are_parallel(θ[0], θ[3]) and are_parallel(θ[1], θ[2]):
        corners[0] = line_intersection(ρ[0], ρ[1], θ[0], θ[1])
        corners[1] = line_intersection(ρ[0], ρ[2], θ[0], θ[2])
        corners[2] = line_intersection(ρ[3], ρ[1], θ[3], θ[1])
        corners[3] = line_intersection(ρ[3], ρ[2], θ[3], θ[2])

    # Identify the position of the corners. We select each of the 
    # corners by projecting their coordinates on the lines y = x and 
    # y = -x.
    ix_upleft_corner = np.argmin(corners[:, 0] + corners[:, 1])
    ix_doright_corner = np.argmax(corners[:, 0] + corners[:, 1])
    ix_upright_corner = np.argmax(corners[:, 0] - corners[:, 1])
    ix_doleft_corner = np.argmin(corners[:, 0] - corners[:, 1])

    # Rearrange corners in order.
    cp_corners = np.copy(corners)
    corners[0] = cp_corners[ix_upleft_corner]
    corners[1] = cp_corners[ix_upright_corner]
    corners[2] = cp_corners[ix_doleft_corner]
    corners[3] = cp_corners[ix_doright_corner]

    # Compute area inside quadrilateral.
    x = corners[:, 0]
    y = corners[:, 1]
    area = (
        np.abs((x[1]-x[0])*(y[2]-y[1]) - (x[1]-x[2])*(y[0]-y[1]))/2
        + np.abs((x[1]-x[3])*(y[2]-y[1]) - (x[1]-x[2])*(y[3]-y[1]))/2
    )

    if area > max_area:
        doc_edges[:, 0] = ρ
        doc_edges[:, 1] = θ
        # For some reason we cannot change doc_corners by reference.
        doc_corners[:, :] = corners
        max_area = area

    return max_area


def corner_detection(im):
    orig_imsize = np.shape(im)[0:2]
    # Resize and change color to greys.
    im = imutils.resize(im, height=500)
    img = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
    imsize = np.shape(im)
    
    # Find Hough lines in the image.
    img = cv2.GaussianBlur(img, (3, 3), 0) 
    edges = cv2.Canny(img, 100, 200)
    lines = cv2.HoughLines(edges, 1, np.pi/180, 120)
    lines = lines.reshape(np.shape(lines)[0], 2)

    # Find the document's corners. We select as corners those defined as the intersection
    # of four lines, which are parallel two by two, and have maximal area.
    max_area = -1.0
    # Iterate through every pair of lines. 
    doc_corners = np.zeros((4, 2))
    doc_edges = np.zeros((4, 2))
    for comb in combinations(range(np.shape(lines)[0]), 4):
        # Find the four corners and area inside detected document and
        # check if area is bigger than what we have.
        max_area = find_doc_edges(lines[comb, :], doc_corners, doc_edges, max_area)

    # Check if the algorithm has detected four corners.
    if np.shape(doc_corners)[0] == 4:
        has_detected_corners = True
    else:
        has_detected_corners = False
        return [], [], lines, has_detected_corners, 1.
    
    # Approximate measurement of document's scale.
    # Mean of vertical (height) and horizontal (widht) lenghts.
    µ_hl = (
        np.linalg.norm(doc_corners[2] - doc_corners[0])
        + np.linalg.norm(doc_corners[3] - doc_corners[1])
    )/2
    µ_wl = (
        np.linalg.norm(doc_corners[0] - doc_corners[1])
        + np.linalg.norm(doc_corners[2] - doc_corners[3])
    )/2
    doc_scale = µ_hl/µ_wl
    
    # Rescale document's corners to original size.
    doc_corners[:, 0] *= orig_imsize[0]/imsize[0]
    doc_corners[:, 1] *= orig_imsize[1]/imsize[1]
    
    return doc_corners, doc_edges, lines, has_detected_corners, doc_scale


def perspective_transformation(im, doc_corners, doc_scale):
    imsize = np.shape(im)
    pts1 = np.float32(doc_corners)
    # Height and width of original image.
    H, W = imsize[:2]
    # Put into document's scale.
    H = int(W*doc_scale)
    pts2 = np.float32([[0, 0], [W, 0], [0, H], [W, H]])
    M = cv2.getPerspectiveTransform(pts1, pts2)
    imt = cv2.warpPerspective(im, M, (W, H))
    return imt


def do_image_thresholding(im):
    img = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
    img = cv2.adaptiveThreshold(img, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 51, 5)
    return img


def insert_in_namepath(path, to_insert):
    spath = path.split(".")
    spath.insert(-1, to_insert)
    spath[-1] = "." + spath[-1]
    return "".join(spath)


def scanner_main(image_path, dest_name, debug=False):
    im = cv2.imread(image_path)
    orig = im.copy()
    
    doc_corners, doc_edges, lines, has_detected_corners, doc_scale = corner_detection(im)

    if has_detected_corners:
        im = perspective_transformation(im, doc_corners, doc_scale)
        im = do_image_thresholding(im)
        im = imutils.resize(im, height=1000)

        # Save final image.
        cv2.imwrite(dest_name, im)
    else:
        print("The algorithm didn't detect the corners of the document. Please check with --debug enabled.")

    # Debug: write an image of the middle process.
    if debug == True:
        im_dbg = orig.copy()

        # Overwrite detected document corners.
        circ_radius = int(np.shape(im_dbg)[0]*0.01)
        for corner in doc_corners:
            cv2.circle(im_dbg, (int(corner[0]), int(corner[1])), circ_radius, (0, 0, 255), -1)

        # Overwrite detected lines. If line is vertical return ad hoc values.
        def xy_hough_lines(ρ, θ, x):
            if np.abs(np.sin(θ)) > 1e-3:
                y = (ρ - x*np.cos(θ))/np.sin(θ)
                return int(y)
            else:
                return 0 if x == 0 else 500

        for line in lines:
            ρ, θ = line
            ρ *= np.shape(im_dbg)[0]/500
            x1 = 0
            y1 = xy_hough_lines(ρ, θ, x1)
            x2 = np.shape(im_dbg)[0]
            y2 = xy_hough_lines(ρ, θ, x2)
            cv2.line(im_dbg, (x1, y1), (x2, y2), (255, 0, 0), 5)

        # Overwrite detected document edges.
        for line in doc_edges:
            ρ, θ = line
            ρ *= np.shape(im_dbg)[0]/500
            x1 = 0
            y1 = xy_hough_lines(ρ, θ, x1)
            x2 = np.shape(im_dbg)[0]
            y2 = xy_hough_lines(ρ, θ, x2)
            cv2.line(im_dbg, (x1, y1), (x2, y2), (0, 0, 255), 5)
        
        cv2.imwrite(insert_in_namepath(dest_name, "_dbg"), im_dbg)
    
    return


if __name__ == "__main__":
    if len(sys.argv) >= 3:
        if "--debug" in sys.argv:
            sys.argv.remove("--debug")
            scanner_main(sys.argv[1], sys.argv[2], True)
        else:
            scanner_main(sys.argv[1], sys.argv[2], False)
    else:
        print("Please, introduce the path of the original image and the path of the destination image.")
        sys.exit()
        
  ##############################################################################################################################
  
  #### Usage ###################################################################################################################
  #### $ python pyscanner.py "image_path" "destination_path" ###################################################################
  
  ##############################################################################################################################
