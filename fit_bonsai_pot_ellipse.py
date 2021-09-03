import argparse
import logging
import os
import random
import math
import ransac.core as ransac
from ransac.models.conic_section import ConicSection
import cv2
import numpy as np

logging.basicConfig(level=logging.DEBUG, format='%(asctime)-15s [%(levelname)s] %(message)s')

parser = argparse.ArgumentParser()
parser.add_argument('--imageFilepath', help="The filepath to the image. Default: './images/bonsai.png'", default='./images/bonsai.png')
parser.add_argument('--randomSeed', help="The seed for the random module. Default: 0", type=int, default=0)
parser.add_argument('--outputDirectory', help="The output directory. Default: './outputs/'", default='./outputs/')
parser.add_argument('--blurringKernelSize', help="The size of the blurring kernel. Default: 9", type=int, default=9)
parser.add_argument('--cannyThreshold1', help="The 1st Canny threshold. Default: 30", type=int, default=30)
parser.add_argument('--cannyThreshold2', help="The 2nd Canny threshold. Default: 150", type=int, default=150)
parser.add_argument('--ransacMaximumNumberOfPoints', help="The maximum number of points used by the RANSAC algorithm. Default: 400", type=int, default=400)
parser.add_argument('--ransacNumberOfTrials', help="The number of RANSAC trials. Default: 1000", type=int, default=1000)
parser.add_argument('--ransacAcceptableError', help="The acceptable error for the RANSAC algorithm. Default: 10", type=float, default=10)
args = parser.parse_args()

def main():
    logging.info("fit_bonsai_pot_ellipse.py main()")

    if not os.path.exists(args.outputDirectory):
        os.makedirs(args.outputDirectory)

    original_img = cv2.imread(args.imageFilepath, cv2.IMREAD_COLOR)

    # Convert to grayscale
    grayscale_img = cv2.cvtColor(original_img, cv2.COLOR_BGR2GRAY)
    cv2.imwrite(os.path.join(args.outputDirectory, "fitBonsaiPot_main_grayscale.png"), grayscale_img)

    # blur
    blurred_img = cv2.blur(grayscale_img, (args.blurringKernelSize, args.blurringKernelSize))
    cv2.imwrite(os.path.join(args.outputDirectory, "fitBonsaiPot_main_blurred.png"), blurred_img)

    # Canny
    canny_img = cv2.Canny(blurred_img, args.cannyThreshold1, args.cannyThreshold2)
    cv2.imwrite(os.path.join(args.outputDirectory, "fitBonsaiPot_main_canny.png"), canny_img)

    green_dominated_inverse_map = GreenDominatedInverseMap(original_img)
    cv2.imwrite(os.path.join(args.outputDirectory, "fitBonsaiPot_main_greenDominatedInverseMap.png"), green_dominated_inverse_map)

    masked_canny_img = cv2.min(canny_img, green_dominated_inverse_map)
    cv2.imwrite(os.path.join(args.outputDirectory, "fitBonsaiPot_main_maskedCanny.png"), masked_canny_img)
    # List the points
    xy_tuples = []
    for y in range(masked_canny_img.shape[0]):
        for x in range(masked_canny_img.shape[1]):
            if masked_canny_img[y, x] > 0:
                xy_tuples.append(((x, y), 0))
    xy_tuples = random.sample(xy_tuples, min(len(xy_tuples), args.ransacMaximumNumberOfPoints))

    modeller = ransac.Modeler(ConicSection, number_of_trials=args.ransacNumberOfTrials, acceptable_error=args.ransacAcceptableError)
    logging.info("Calling modeller.ConsensusModel(xy_tuples)...")
    consensus_conic_section, inliers, outliers = modeller.ConsensusModel(xy_tuples)
    logging.info("Done!")

    annotated_img = original_img.copy()
    ellipse_points = consensus_conic_section.EllipsePoints()
    logging.debug("len(ellipse_points) = {}".format(len(ellipse_points)))
    for ellipse_pt_ndx in range(0, len(ellipse_points) - 1):
        p1 = ellipse_points[ellipse_pt_ndx]
        p2 = ellipse_points[ellipse_pt_ndx + 1]
        cv2.line(annotated_img, p1, p2, (255, 0, 0), thickness=3)
    cv2.line(annotated_img, ellipse_points[0], ellipse_points[-1], (255, 0, 0), thickness=3)

    for ((x, y), d) in inliers:
        cv2.circle(annotated_img, (x, y), 2, (0, 255, 0))
    for ((x, y), d) in outliers:
        cv2.circle(annotated_img, (x, y), 2, (0, 0, 255))

    cv2.imwrite(os.path.join(args.outputDirectory, "fitBonsaiPot_main_annotated.png"), annotated_img)

    # Ellipse parameters
    center, a, b, theta = consensus_conic_section.EllipseParameters()
    logging.debug("center = {}".format(center))
    logging.debug("a = {}".format(a))
    logging.debug("b = {}".format(b))
    logging.debug("theta = {}".format(theta))

def GreenDominatedInverseMap(image):
    img_sizeHWC = image.shape
    green_dominated_map = np.zeros((img_sizeHWC[0], img_sizeHWC[1]), dtype=np.uint8)
    for y in range(img_sizeHWC[0]):
        for x in range(img_sizeHWC[1]):
            bgr = image[y, x]
            if bgr[1] > bgr[0] and bgr[1] > bgr[2]:
                green_dominated_map[y, x] = 255
    cv2.imwrite(os.path.join(args.outputDirectory, "fitBonsaiPot_GreenDominatedInverseMap_greenDominatedMap.png"), green_dominated_map)
    return 255 - green_dominated_map

if __name__ == '__main__':
    main()