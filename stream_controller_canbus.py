#!/usr/bin/python3
import cv2, socket, sys, struct, time
import numpy as np
import threading
from evdev import InputDevice


def can_send_msg():
    global l_button
    global r_button
    global r1_trigg
    global timestep

    sock = socket.socket(socket.PF_CAN, socket.SOCK_RAW, socket.CAN_RAW)
    interface = "can1"
    try:
        sock.bind((interface,))
    except OSError:
        sys.stderr.write("Could not bind to interface '%s'\n" % interface)
    while (True):
        time.sleep(timestep)
        if r1_trigg > 0:
            r2_trigg = 255  # 220
            l_button_data = l_button.to_bytes(1, byteorder='big')
            r_button_data = r_button.to_bytes(1, byteorder='big')
            o_button_data = o_button.to_bytes(1, byteorder='big')
            u_button_data = u_button.to_bytes(1, byteorder='big')
            v_button_data = v_button.to_bytes(1, byteorder='big')
            k_button_data = k_button.to_bytes(1, byteorder='big')
            d_button_data = d_button.to_bytes(1, byteorder='big')
            x_button_data = x_button.to_bytes(1, byteorder='big')

            l_stick_x_data = l_stick_x.to_bytes(1, byteorder='big')
            l_stick_y_data = l_stick_y.to_bytes(1, byteorder='big')
            r_stick_x_data = r_stick_x.to_bytes(1, byteorder='big')
            r_stick_y_data = r_stick_y.to_bytes(1, byteorder='big')

            l1_trigg_data = l1_trigg.to_bytes(1, byteorder='big')
            l2_trigg_data = l2_trigg.to_bytes(1, byteorder='big')
            r1_trigg_data = r1_trigg.to_bytes(1, byteorder='big')
            r2_trigg_data = r2_trigg.to_bytes(1, byteorder='big')

            se_button_data = se_button.to_bytes(1, byteorder='big')
            st_button_data = st_button.to_bytes(1, byteorder='big')
            ps_button_data = ps_button.to_bytes(1, byteorder='big')

            data_100 = l_button_data + r_button_data + o_button_data + u_button_data
            data_100 += v_button_data + k_button_data + d_button_data + x_button_data

            data_101 = l_stick_x_data + l_stick_y_data + r_stick_x_data + r_stick_y_data
            data_101 += l1_trigg_data + l2_trigg_data + r1_trigg_data + r2_trigg_data

            data_102 = se_button_data + st_button_data + ps_button_data

            try:
                can_pkt = struct.pack("<IB3x8s", 0x100, len(data_100), data_100)
                sock.send(can_pkt)


            except socket.error:
                print('Error sending CAN frame')

            try:
                can_pkt = struct.pack("<IB3x8s", 0x101, len(data_101), data_101)
                sock.send(can_pkt)


            except socket.error:
                print('Error sending CAN frame')


def controller():
    global l_button
    global r_button
    global o_button
    global u_button
    global v_button
    global k_button
    global d_button
    global x_button

    global l_stick_x
    global l_stick_y
    global r_stick_x
    global r_stick_y
    global l1_trigg
    global l2_trigg
    global r1_trigg
    global r2_trigg

    global se_button
    global st_button
    global ps_button

    sock = socket.socket(socket.PF_CAN, socket.SOCK_RAW, socket.CAN_RAW)
    interface = "can0"

    l_button = 0;
    r_button = 0;
    o_button = 0;
    u_button = 0;
    v_button = 0;
    k_button = 0;
    d_button = 0;
    x_button = 0;

    l_stick_x = 128;
    l_stick_y = 128;
    r_stick_x = 128;
    r_stick_y = 128;
    l1_trigg = 0;
    l2_trigg = 0;
    r1_trigg = 0;
    r2_trigg = 0;

    se_button = 0;
    st_button = 0;
    ps_button = 0;

    try:
        ps3_dev = InputDevice('/dev/input/event6')
    except OSError:
        sys.stderr.write("Could not bind to interface '%s'\n" % Ps3_dev)

    try:
        sock.bind((interface,))
    except OSError:
        sys.stderr.write("Could not bind to interface '%s'\n" % interface)

    while (True):

        for event in ps3_dev.read_loop():
            # print(l_button,r_button)
            if event.type == 3 and event.code == 11 and r1_trigg == 0:
                l_button = event.value

            elif event.type == 3 and event.code == 9 and r1_trigg == 0:
                r_button = event.value

            if event.type == 3 and event.code == 8:
                o_button = event.value

            elif event.type == 3 and event.code == 10:
                u_button = event.value

            elif event.type == 3 and event.code == 29:
                v_button = event.value

            elif event.type == 3 and event.code == 27:
                k_button = event.value

            elif event.type == 3 and event.code == 26:
                d_button = event.value

            elif event.type == 3 and event.code == 28:
                x_button = event.value

            elif event.type == 3 and event.code == 0:
                l_stick_x = event.value + 128

            elif event.type == 3 and event.code == 1:
                l_stick_y = event.value + 128

            elif event.type == 3 and event.code == 3:
                r_stick_x = event.value + 128

            elif event.type == 3 and event.code == 2:
                r_stick_y = event.value + 128

            elif event.type == 3 and event.code == 14:
                l1_trigg = event.value

            elif event.type == 3 and event.code == 12:
                l2_trigg = event.value

            elif event.type == 3 and event.code == 15:
                r1_trigg = event.value

            elif event.type == 3 and event.code == 13:
                r2_trigg = event.value

            elif event.type == 1 and event.code == 288:
                se_button = event.value

            elif event.type == 1 and event.code == 291:
                st_button = event.value

            elif event.type == 1 and event.code == 304:
                ps_button = event.value


            else:
                l_button_data = l_button.to_bytes(1, byteorder='big')
                r_button_data = r_button.to_bytes(1, byteorder='big')
                o_button_data = o_button.to_bytes(1, byteorder='big')
                u_button_data = u_button.to_bytes(1, byteorder='big')
                v_button_data = v_button.to_bytes(1, byteorder='big')
                k_button_data = k_button.to_bytes(1, byteorder='big')
                d_button_data = d_button.to_bytes(1, byteorder='big')
                x_button_data = x_button.to_bytes(1, byteorder='big')

                l_stick_x_data = l_stick_x.to_bytes(1, byteorder='big')
                l_stick_y_data = l_stick_y.to_bytes(1, byteorder='big')
                r_stick_x_data = r_stick_x.to_bytes(1, byteorder='big')
                r_stick_y_data = r_stick_y.to_bytes(1, byteorder='big')

                l1_trigg_data = l1_trigg.to_bytes(1, byteorder='big')
                l2_trigg_data = l2_trigg.to_bytes(1, byteorder='big')
                r1_trigg_data = r1_trigg.to_bytes(1, byteorder='big')
                r2_trigg_data = r2_trigg.to_bytes(1, byteorder='big')

                se_button_data = se_button.to_bytes(1, byteorder='big')
                st_button_data = st_button.to_bytes(1, byteorder='big')
                ps_button_data = ps_button.to_bytes(1, byteorder='big')

                data_100 = l_button_data + r_button_data + o_button_data + u_button_data
                data_100 += v_button_data + k_button_data + d_button_data + x_button_data

                data_101 = l_stick_x_data + l_stick_y_data + r_stick_x_data + r_stick_y_data
                data_101 += l1_trigg_data + l2_trigg_data + r1_trigg_data + r2_trigg_data

                data_102 = se_button_data + st_button_data + ps_button_data

                time.sleep(0.0001)
                try:
                    can_pkt = struct.pack("<IB3x8s", 0x100, len(data_100), data_100)
                    sock.send(can_pkt)


                except socket.error:
                    print('Error sending CAN frame')

                time.sleep(0.0001)
                try:
                    can_pkt = struct.pack("<IB3x8s", 0x101, len(data_101), data_101)
                    sock.send(can_pkt)


                except socket.error:
                    print('Error sending CAN frame')

                time.sleep(0.0001)
                try:
                    can_pkt = struct.pack("<IB3x8s", 0x102, len(data_102), data_102)
                    sock.send(can_pkt)


                except socket.error:
                    print('Error sending CAN frame')


# Region-of-interest vertices
# We want a trapezoid shape, with bottom edge at the bottom of the image
trap_bottom_width = 1  # 0.45 width of bottom edge of trapezoid, expressed as percentage of image width
trap_top_width = 1  # 0.12 ditto for top edge of trapezoid
trap_height = 1  # 0.75 height of the trapezoid expressed as percentage of image height
# Gaussian smoothin
kernel_size = 5

# Canny Edge Detector
low_threshold = 50
high_threshold = 200


# Helper Functions
def filter_colors(image):
    """
	Filter the image to include only yellow and white pixels
	"""

    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

    # Filter white pixels
    white_threshold = 0  # 130
    lower_white = np.array([150, 150, 0])  # 0,245,242
    upper_white = np.array([255, 255, 138])
    white_mask = cv2.inRange(image, lower_white, upper_white)
    white_image = cv2.bitwise_and(image, image, mask=white_mask)
    # cv2.imshow('frame', white_image)
    # test123 = image[305,455]
    # image[300:310,450:460] =[255,255,255]
    # print (test123)
    # cv2.imshow('frame_orig', image)
    return white_image

    # Filter  pixels mask image
    lower_yellow = np.array([30, 100, 100])
    upper_yellow = np.array([255, 255, 255])
    yellow_mask = cv2.inRange(image, lower_yellow, upper_yellow)

    yellow_image = cv2.bitwise_and(hsv, image, mask=yellow_mask)
    # cv2.imshow('frame', yellow_image)
    # Combine the two above images
    image2 = cv2.addWeighted(white_image, 1., yellow_image, 1., 0.)
    # cv2.imshow('frame_2', image2)
    # cv2.waitKey(0)


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
    global l_button
    global r_button
    global r1_trigg
    global l_button_i
    global r_button_i
    global r_button_d
    global l_button_d
    global error_r
    global error_l
    global timestep
    # l_button_d = 0
    # r_button_d = 0

    # Only keep white and yellow pixels in the image, all other pixels become black
    # image_in = cv2.selectROI(image_in)

    # black = np.zeros((image_in.shape[0], image_in.shape[1], 3), np.uint8)
    # black_cropped = cv2.rectangle(black, (150, 400), (1100, 700), (255, 0, 255), -1)

    # _, thresh = cv2.threshold(image_in, 127, 255, 0)

    image_in = image_in[280: 380, 100: 1180]

    image = filter_colors(image_in)

    # Read in and grayscale the image
    gray = grayscale(image)

    # Apply Gaussian smoothing
    # blur_gray = gaussian_blur(gray, kernel_size)
    thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY)[1]
    # thresh = cv2.erode(thresh, None, iterations=2)
    # thresh = cv2.dilate(thresh, None, iterations=2)
    # Apply Canny Edge Detector
    # cv2.imshow('thresh', thresh)
    # edges = canny(thresh, low_threshold, high_threshold)c

    # Create masked edges using trapezoid-shaped region-of-interest
    imshape = image.shape
    vertices = np.array([[ \
        ((imshape[1] * (1 - trap_bottom_width)) // 2, imshape[0]), \
        ((imshape[1] * (1 - trap_top_width)) // 2, imshape[0] - imshape[0] * trap_height), \
        (imshape[1] - (imshape[1] * (1 - trap_top_width)) // 2, imshape[0] - imshape[0] * trap_height), \
        (imshape[1] - (imshape[1] * (1 - trap_bottom_width)) // 2, imshape[0])]] \
        , dtype=np.int32)

    # lower_left = [imshape[1]/9,imshape[0]]
    # lower_right = [imshape[1]-imshape[1]/9,imshape[0]]
    # top_left = [imshape[1]/2-imshape[1]/8,imshape[0]/2+imshape[0]/10]
    # top_right = [imshape[1]/2+imshape[1]/8,imshape[0]/2+imshape[0]/10]
    # vertices = [np.array([lower_left,top_left,top_right,lower_right],dtype=np.int32)]
    masked_edges = region_of_interest(thresh, vertices)

    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (7, 7))

    _, contours, h = cv2.findContours(masked_edges, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)

    # print(contours)

    # for cont in contours:
    #        approx = cv2.approxPolyDP(cont, 50, True)
    #        cv2.drawContours(masked_edges, [approx], -1, (255, 0, 0), 10)
    # print(len(contours))

    # //TODO !!! The Mass of Point -> if List == 1 MOP = 0 !! chek length of list

    # get the Point (XY) of mass of the contours

    # cv2.waitKey(0)

    # cv2.drawContours(masked_edges,contours, -1,(0,255,0),3)
    # cv2.imshow('test', masked_edges)
    if contours == []:
        print('No lines detected')
        lenkeinschlag = 'Wird kalibriert'
        if r1_trigg > 0 and r_button > 0 or r1_trigg > 0 and l_button > 0:
            r_button = 0
            l_button = 0
            l_button_i = 0
            r_button_i = 0
            r_button_d = 0
            l_button_d = 0
            error_l = 0
            error_r = 0
    else:
        cnt = contours[0]
        # print(len(cnt))
        # print(cnt)
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

        # print(['m10'])
        # print(['m01'])

        cx = int(mass['m10'] / xmass)
        cy = int(mass['m01'] / ymass)
        font = cv2.FONT_HERSHEY_SIMPLEX
        # print(cx, cy)
        # print(timestep)
        # print(error_l,error_r)

        # if mass['m10'] != 0:
        # print(mass['m10'],mass['m01'])
        border_high = 580
        border_low = 570
        kp = 0.023  # 0.00055 #0.0015
        ki = 0.018  # 0.001
        kd = 0.0019  # 0.0078 #0.0025
        offset_l = 74  # 80
        offset_r = 74

        if cx < border_high and cx > border_low:
            lenkeinschlag = 'gerade'
            cv2.putText(masked_edges, lenkeinschlag, (0, 46), font, 2, (200, 255, 155), 3, cv2.LINE_AA)
            if r1_trigg > 0:
                r_button = 0
                l_button = 0
                l_button_i = 0
                r_button_i = 0
                r_button_d = 0
                l_button_d = 0
                error_l = 0
                error_r = 0

        elif cx <= border_low and cx > 0:
            lenkeinschlag = 'Links'
            cv2.putText(masked_edges, lenkeinschlag, (0, 46), font, 2, (200, 255, 155), 3, cv2.LINE_AA)
            if r1_trigg > 0:
                r_button = 0
                r_button_i = 0
                l_button = offset_l + int(kp * error_l + ki * l_button_i + kd * l_button_d)
                r_button_d = 0
                error_r = 0
                if l_button >= 255:
                    l_button = 255
                    l_button_i = 0
                    l_button_d = 0
                elif l_button_i >= 250:
                    l_button_i = 250

                l_button_i += error_l * timestep
                l_button_d = ((border_low - cx) + error_l) / timestep

                if l_button_d >= 15000:
                    l_button_d = 15000

            elif r1_trigg == 0:
                l_button_i = 0
                l_button = 0
                l_button_d = 0
            error_l = (border_low - cx)

        elif cx >= border_high:
            lenkeinschlag = 'Rechts'
            cv2.putText(masked_edges, lenkeinschlag, (0, 46), font, 2, (200, 255, 155), 3, cv2.LINE_AA)
            if r1_trigg > 0:
                l_button = 0
                l_button_i = 0
                r_button = offset_r + int(kp * error_r + ki * r_button_i + kd * r_button_d)
                l_button_d = 0
                error_l = 0
                if r_button >= 255:
                    r_button = 255
                    r_button_i = 0
                    r_button_d = 0
                elif r_button_i >= 250:
                    r_button_i = 250

                r_button_i += error_r * timestep
                r_button_d = (cx - border_high + error_r) / timestep

                if r_button_d >= 15000:
                    r_button_d = 15000

            elif r1_trigg == 0:
                r_button_i = 0
                r_button = 0
                r_button_d = 0
            error_r = cx - border_high

        else:
            lenkeinschlag = 'Wird kalibriert'
            # print(cx, cy)
        print(l_button, r_button, "    ", l_button_i, r_button_i, "    ", l_button_d, r_button_d, "    ", error_l,
              error_r, "    ", cx, cy, "    ", timestep)

        cx_2 = cx + 10
        cy_2 = cy + 10
        masked_edges[cy:cy_2, cx:cx_2] = [6000]

    # Run Hough on edge detected image
    # line_image = cv2.HoughLinesP(masked_edges, rho=1, theta=np.pi/180, threshold=20, minLineLength=20, maxLineGap=300)
    # cv2.imshow('Houghlines', line_image)

    # Draw lane lines on the original image
    # initial_image = image_in.astype('uint8')
    # annotated_image = cv2.addWeighted(line_image, masked_edges)
    """For Further image processung use annotated image as return"""
    # return annotated_image
    # cv2.imshow('frame_test', masked_edges)
    # cv2.waitKey(0)
    return masked_edges


# Main Function

cap = cv2.VideoCapture('nvcamerasrc fpsRange="30 30" ! '
                       'video/x-raw(memory:NVMM), width=(int)2592, height=(int)1458, format=(string)I420, framerate=(fraction)30/1 ! '
                       'nvvidconv ! video/x-raw, width=(int)1280, height=(int)720, format=(string)BGRx ! '
                       'videoconvert ! appsink', cv2.CAP_GSTREAMER)


# cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
# cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

def kamera():
    global l_button_i;
    global r_button_i;
    global l_button_d;
    global r_button_d;
    global error_r;
    global error_l;
    global timestep

    l_button_i = 0
    r_button_i = 0
    l_button_d = 0
    r_button_d = 0
    error_r = 0
    error_l = 0
    timestep = 0.03

    while (True):
        start = time.time()
        # Capture frame-by-frame
        ret, frame = cap.read()

        if ret:

            # Our operations on the frame come here
            # gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            # Display the resulting frame
            cv2.imshow('frame_unbearbeitet', frame)
            # cv2.waitKey(0)

            frame = imageprocessing(frame)
            cv2.imshow('frame_bearbeitet', frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        else:
            break
        end = time.time()
        timestep = end - start
    # When everything done, release the capture
    cap.release()
    cv2.destroyAllWindows()


# start everything
controller = threading.Thread(target=controller)
can_send_msg = threading.Thread(target=can_send_msg)
kamera = threading.Thread(target=kamera)

controller.start()
kamera.start()
can_send_msg.start()

# Release everything if job is finished
kamera.join()
can_send_msg.join()
controller.join()