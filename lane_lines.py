
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np
import cv2
from moviepy.editor import VideoFileClip
import math
import os

# Global parameters

_width = 1280
_margin = 0
_height = 720

# Gaussian smoothing
kernel_size = 31

# Canny Edge Detector
low_threshold = 110
high_threshold = 110

# Region-of-interest vertices
# We want a trapezoid shape, with bottom edge at the bottom of the image
trap_bottom_width = 0.85  # width of bottom edge of trapezoid, expressed as percentage of image width
trap_top_width = 0.07  # ditto for top edge of trapezoid
trap_height = 0.4  # height of the trapezoid expressed as percentage of image height

# Hough Transform
rho = 20  # distance resolution in pixels of the Hough grid
theta = 1 * np.pi / 180  # angular resolution in radians of the Hough grid
threshold = 30  # minimum number of votes (intersections in Hough grid cell)
min_line_length = 30  # minimum number of pixels making up a line
max_line_gap = 60  # maximum gap in pixels between connectable line segments


# Helper functions
def grayscale(img):
    return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

def canny(img, low_threshold, high_threshold):
    return cv2.Canny(img, low_threshold, high_threshold)

def gaussian_blur(img, kernel_size):
    return cv2.GaussianBlur(img, (kernel_size, kernel_size), 0)

def region_of_interest(img, vertices):

    # defining a blank mask to start with
    mask = np.zeros_like(img)

    # defining a 3 channel or 1 channel color to fill the mask with depending on the input image
    if len(img.shape) > 2:
        channel_count = img.shape[2]  # i.e. 3 or 4 depending on your image
        ignore_mask_color = (255,) * channel_count
    else:
        ignore_mask_color = 255

    # filling pixels inside the polygon defined by "vertices" with the fill color
    cv2.fillPoly(mask, vertices, ignore_mask_color)

    # returning the image only where mask pixels are nonzero
    masked_image = cv2.bitwise_and(img, mask)
    return masked_image


def draw_lines(img, lines, color=[255, 0, 0], thickness=10):
    # In case of error, don't draw the line(s)
    if lines is None:
        return
    if len(lines) == 0:
        return
    draw_right = False
    draw_left = False

    # Find slopes of all lines
    # But only care about lines where abs(slope) > slope_threshold
    slope_threshold = 0.7
    slopes = []
    new_lines = []
    for line in lines:
        x1, y1, x2, y2 = line[0] /2 # line = [[x1, y1, x2, y2]]

        # Calculate slope
        if x2 - x1 == 0.:  # corner case, avoiding division by 0
            slope = 999.  # practically infinite slope
        else:
            slope = (y2 - y1) / (x2 - x1)

        # Filter lines based on slope
        if abs(slope) > slope_threshold:
            slopes.append(slope)
            new_lines.append(line)

    lines = new_lines

    # Split lines into right_lines and left_lines, representing the right and left lane lines
    # Right/left lane lines must have positive/negative slope, and be on the right/left half of the image
    right_lines = []
    left_lines = []
    for i, line in enumerate(lines):
        x1, y1, x2, y2 = line[0]
        img_x_center = img.shape[1]   # x coordinate of center of image
        if slopes[i] > 0 and x1 > img_x_center and x2 > img_x_center:
            right_lines.append(line)
        elif slopes[i] < 0 and x1 < img_x_center and x2 < img_x_center:
            left_lines.append(line)

    # Run linear regression to find best fit line for right and left lane lines
    # Right lane lines
    right_lines_x = []
    right_lines_y = []

    for line in right_lines:
        x1, y1, x2, y2 = line[0]

        right_lines_x.append(x1)
        right_lines_x.append(x2)

        right_lines_y.append(y1)
        right_lines_y.append(y2)

    if len(right_lines_x) > 0:
        right_m, right_b = np.polyfit(right_lines_x, right_lines_y, 1)  # y = m*x + b
    else:
        right_m, right_b = 1, 1
        draw_right = False

    # Left lane lines
    left_lines_x = []
    left_lines_y = []

    for line in left_lines:
        x1, y1, x2, y2 = line[0]

        left_lines_x.append(x1)
        left_lines_x.append(x2)

        left_lines_y.append(y1)
        left_lines_y.append(y2)

    if len(left_lines_x) > 0:
        left_m, left_b = np.polyfit(left_lines_x, left_lines_y, 1)  # y = m*x + b
    else:
        left_m, left_b = 1, 1
        draw_left = False

    # Find 2 end points for right and left lines, used for drawing the line
    # y = m*x + b --> x = (y - b)/m
    y1 = img.shape[0]
    y2 = img.shape[0] * (1 - trap_height)

    right_x1 = (y1 - right_b) / right_m
    right_x2 = (y2 - right_b) / right_m

    left_x1 = (y1 - left_b) / left_m
    left_x2 = (y2 - left_b) / left_m

    # Convert calculated end points from float to int
    y1 = int(y1)
    y2 = int(y2)
    right_x1 = int(right_x1)
    right_x2 = int(right_x2)
    left_x1 = int(left_x1)
    left_x2 = int(left_x2)

    # Draw the right and left lines on image
    if draw_right:
        cv2.line(img, (right_x1, y1), (right_x2, y2), color, thickness)
    if draw_left:
        cv2.line(img, (left_x1, y1), (left_x2, y2), color, thickness)

def hough_lines(img, rho, theta, threshold, min_line_len, max_line_gap):
    """
	`img` should be the output of a Canny transform.

	Returns an image with hough lines drawn.
	"""
    lines = cv2.HoughLinesP(img, rho, theta, threshold, np.array([]), minLineLength=min_line_len,
                            maxLineGap=max_line_gap)
    # line_img = np.zeros(img.shape, dtype=np.uint8)
    line_img = np.zeros((*img.shape, 3), dtype=np.uint8)  # 3-channel RGB image
    draw_lines(line_img, lines)
    return line_img

def weighted_img(img, initial_img, α=0.8, β=1., λ=0.):
    """
	`img` is the output of the hough_lines(), An image with lines drawn on it.
	Should be a blank image (all black) with lines drawn on it.

	`initial_img` should be the image before any processing.

	The result image is computed as follows:

	initial_img * α + img * β + λ
	NOTE: initial_img and img must be the same shape!
	"""
    return cv2.addWeighted(initial_img, α, img, β, λ)


def filter_colors(image):
    """
	Filter the image to include only yellow and white pixels
	"""

    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

     # Filter white pixels
    white_threshold = 255 # 130
    lower_white = np.array([white_threshold, 100, 100])
    upper_white = np.array([255, 255, 255])
    white_mask = cv2.inRange(image, lower_white, upper_white)
    white_image = cv2.bitwise_and(image, image, mask=white_mask)

    # Filter  pixels mask image
    lower_yellow = np.array([30, 100, 100])
    upper_yellow = np.array([255, 255, 255])
    yellow_mask = cv2.inRange(hsv, lower_yellow, upper_yellow)

    yellow_image = cv2.bitwise_and(image, image, mask=yellow_mask)

    # Combine the two above images
    image2 = cv2.addWeighted(white_image, 1., yellow_image, 1., 0.)
    # cv2.imshow('frame', image2)
    # cv2.waitKey(0)
    return image2

def annotate_image_array(image_in):
    """ Given an image Numpy array, return the annotated image as a Numpy array """
    # Only keep white and yellow pixels in the image, all other pixels become black
    #image_in = cv2.selectROI(image_in)

    #black = np.zeros((image_in.shape[0], image_in.shape[1], 3), np.uint8)
    #black_cropped = cv2.rectangle(black, (150, 400), (1100, 700), (255, 0, 255), -1)

    # _, thresh = cv2.threshold(image_in, 127, 255, 0)

    image_in = image_in[382: 700, 25: 1255]

    image = filter_colors(image_in)

    # Read in and grayscale the image
    gray = grayscale(image)

    # Apply Gaussian smoothing
    blur_gray = gaussian_blur(gray, kernel_size)

    # Apply Canny Edge Detector

    edges = canny(blur_gray, low_threshold, high_threshold)

    # Create masked edges using trapezoid-shaped region-of-interest
    imshape = image.shape
    vertices = np.array([[ \
        ((imshape[1] * (1 - trap_bottom_width)) // 2, imshape[0]), \
        ((imshape[1] * (1 - trap_top_width)) // 2, imshape[0] - imshape[0] * trap_height), \
        (imshape[1] - (imshape[1] * (1 - trap_top_width)) // 2, imshape[0] - imshape[0] * trap_height), \
        (imshape[1] - (imshape[1] * (1 - trap_bottom_width)) // 2, imshape[0])]] \
        , dtype=np.int32)

    masked_edges = region_of_interest(edges, vertices)

    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (7, 7))

    _, contours, h = cv2.findContours(edges, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)

    for cont in contours:
            approx = cv2.approxPolyDP(cont, 50, True)
            cv2.drawContours(masked_edges, [approx], -1, (255, 0, 0), 10)
            #print(len(contours))

    #//TODO !!! The Mass of Point -> if List == 1 MOP = 0 !! chek length of list

    # get the Point (XY) of mass of the contours


    cnt = contours[0]
    mass = cv2.moments(cnt)


   # print(mass)

    # avoiding divisor of zero
    if mass['m00'] == 0.0:
        mass['m00'] = 590
        mass['m10'] = 590
        mass['m01'] = 590
        xmass = 1
        ymass = 1
    else:
        xmass = mass['m00']
        ymass = mass['m00']

    #if mass['m00'] == 0:
        #ymass = 1
    #else:


    print(mass)
    cx = int(mass['m10'] / xmass)
    cy = int(mass['m01'] / ymass)

    if cx > 585 and cx < 645 or 0:
        lenkeinschlag = 'Keine Lenkbewegung'
    elif cx <= 585:
        lenkeinschlag = 'Rechts einlenken'
    elif cx >= 645:
        lenkeinschlag = 'Links einlenken'
    else:
        lenkeinschlag = 'Wird kalibriert'

    print(cx, cy)

    font = cv2.FONT_HERSHEY_SIMPLEX
    cv2.putText(masked_edges, lenkeinschlag, (591, 46), font, 2, (200, 255, 155), 3, cv2.LINE_AA)

    cv2.imshow('frame', masked_edges)
    cv2.waitKey(0)

    # Run Hough on edge detected image
    line_image = hough_lines(masked_edges, rho, theta, threshold, min_line_length, max_line_gap)




    # Draw lane lines on the original image
    initial_image = image_in.astype('uint8')
    annotated_image = weighted_img(line_image, initial_image)

    return annotated_image


def annotate_image(input_file, output_file):
    """ Given input_file image, save annotated image to output_file """
    annotated_image = annotate_image_array(mpimg.imread(input_file))
    plt.imsave(output_file, annotated_image)


def annotate_video(input_file, output_file):
    """ Given input_file video, save annotated video to output_file """

    video = VideoFileClip(input_file)
    annotated_video = video.fl_image(annotate_image_array)
    annotated_video.write_videofile(output_file, audio=False)

    cv2.resize()


# Main script
if __name__ == '__main__':
    from optparse import OptionParser

    # Configure command line options
    parser = OptionParser()
    parser.add_option("-i", "--input_file", dest="input_file",
                      help="Input video/image file")
    parser.add_option("-o", "--output_file", dest="output_file",
                      help="Output (destination) video/image file")
    parser.add_option("-I", "--image_only",
                      action="store_true", dest="image_only", default=False,
                      help="Annotate image (defaults to annotating video)")

    # Get and parse command line options
    options, args = parser.parse_args()

    input_file = options.input_file
    #input_file = cv2.VideoCapture('udpsrc port=5000 ! application/x-rtp,encoding-name=H265,payload=96 ! rtph265depay ! queue ! h265parse ! queue ! avdec_h265 ! queue ! x264enc ! mp4mux ! appsink', cv2.CAP_GSTREAMER)
    output_file = options.output_file
    image_only = options.image_only

    if image_only:
        annotate_image(input_file, output_file)
    else:
        annotate_video(input_file, output_file)