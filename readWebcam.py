
import numpy as np
import cv2
import time

# Region-of-interest vertices
# We want a trapezoid shape, with bottom edge at the bottom of the image
trap_bottom_width = 0.85  # width of bottom edge of trapezoid, expressed as percentage of image width
trap_top_width = 0.07  # ditto for top edge of trapezoid
trap_height = 0.4  # height of the trapezoid expressed as percentage of image height

# Gaussian smoothing
kernel_size = 31

# Canny Edge Detector
low_threshold = 110
high_threshold = 110


#Helper Functions
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

def imageprocessing(image_in):
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

    print(contours)

    for cont in contours:
            approx = cv2.approxPolyDP(cont, 50, True)
            cv2.drawContours(masked_edges, [approx], -1, (255, 0, 0), 10)
            #print(len(contours))

    #//TODO !!! The Mass of Point -> if List == 1 MOP = 0 !! chek length of list

    # get the Point (XY) of mass of the contours

    if contours == 0:
        print('No lines detected')
    else:
        cnt = contours[0]
        #print(len(cnt))
        #print(cnt)
        mass = cv2.moments(cnt)

        # avoiding divisor of zero
        if mass['m00'] == 0:
            xmass = 1
        else:
            xmass = mass['m00']

        if mass['m00'] == 0:
            ymass = 1
        else:
            ymass = mass['m00']

        print(['m10'])
        print(['m01'])

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

    # Run Hough on edge detected image
    #line_image = hough_lines(masked_edges, rho, theta, threshold, min_line_length, max_line_gap)




    # Draw lane lines on the original image
    #initial_image = image_in.astype('uint8')
    #annotated_image = weighted_img(line_image, initial_image)

    """For Further image processung use annotated image as return"""
    #return annotated_image
    cv2.imshow('frame', masked_edges)
    cv2.waitKey(0)
    return masked_edges

# Main Function
device = -1
cap = cv2.VideoCapture(device)

cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

while(True):
    # Capture frame-by-frame
    ret, frame = cap.read()
    # Our operations on the frame come here
    #gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    # Display the resulting frame

    frame = imageprocessing(frame)


    cv2.imshow('frame', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
# When everything done, release the capture
cap.release()
cv2.destroyAllWindows()




