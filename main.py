# coding: utf-8
# Author: Francis (Github @heretic1993)
# License: MIT

import threading
import time

import pyrealsense2 as rs
import numpy as np
import cv2
import matplotlib.pyplot as plt
from Interfaces.camera_interface import camera_interface as ci
from Interfaces.robot_interface import Robot

from Applications.palletizing.palletizing import app, test

# get_ipython().run_line_magic('matplotlib', 'inline')


center = np.array([1, 2, 3], dtype='float64')





class cameraDetection(threading.Thread):
    def __init__(self, threadID, name, robot):
        threading.Thread.__init__(self)
        self.threadID = threadID
        self.name = name
        self.robot = robot
        self.__flag = threading.Event()  # 用于暂停线程的标识
        self.__flag.set()  # 设置为True
        self.__running = threading.Event()  # 用于停止线程的标识
        self.__running.set()  # 将running设置为True

    def run(self):
        pipeline, config = ci.init_camera()
        config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
        config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)

        # Start streaming
        print("Start streaming")
        ##########################
        #         stream = pipeline.get_stream(rs.stream.depth)
        #         intrinsics = stream.get_intrinsics()
        ##########################

        while cv2.waitKey(1) < 0 and self.__running.is_set():
            global color_image

            #             global color_image
            self.__flag.wait()  # 为True时立即返回, 为False时阻塞直到内部的标识位为True后返回
            # Wait for a coherent pair of frames: depth and color
            color_image, depth_frame, _, depth_intrinsics, color_intrinsics = ci.capture_aligned_images(pipeline, config)

            # Convert images to numpy arrays
            depth_image = np.asanyarray(depth_frame.get_data())
            # Apply colormap on depth image (image must be converted to 8-bit per pixel first)
            depth_colormap = cv2.applyColorMap(cv2.convertScaleAbs(depth_image, alpha=0.03), cv2.COLORMAP_JET)
            # Stack both images horizontally
            # images = np.hstack((color_image, depth_colormap))

            # Show images
            # cv2.namedWindow('RealSense', cv2.WINDOW_AUTOSIZE)
            # cv2.imshow("RealSense", images)
            # plt.figure(1)
            # plt.subplot(121)
            # plt.imshow(color_image)
            # plt.subplot(122)
            # plt.imshow(depth_image)
            # cv2.waitKey(1)

            # Our operations on the frame come here
            gray = cv2.cvtColor(color_image, cv2.COLOR_BGR2GRAY)
            aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_6X6_250)
            parameters = cv2.aruco.DetectorParameters()
            detector = cv2.aruco.ArucoDetector(aruco_dict, parameters)

            # lists of ids and the corners beloning to each id
            corners, ids, rejectedImgPoints = detector.detectMarkers(gray)
            #         print(corners)
            # flatten the ArUco IDs list
            if len(corners) > 0:
                # flatten the ArUco IDs list
                ids = ids.flatten()
                # loop over the detected ArUCo corners

                for (markerCorner, markerID) in zip(corners, ids):
                    global center
                    # extract the marker corners (which are always returned in
                    # top-left, top-right, bottom-right, and bottom-left order)
                    corners = markerCorner.reshape((4, 2))
                    (topLeft, topRight, bottomRight, bottomLeft) = corners
                    # convert each of the (x, y)-coordinate pairs to integers
                    topRight = (int(topRight[0]), int(topRight[1]))
                    bottomRight = (int(bottomRight[0]), int(bottomRight[1]))
                    bottomLeft = (int(bottomLeft[0]), int(bottomLeft[1]))
                    topLeft = (int(topLeft[0]), int(topLeft[1]))

                    cv2.line(color_image, topLeft, topRight, (0, 255, 0), 2)
                    cv2.line(color_image, topRight, bottomRight, (0, 255, 0), 2)
                    cv2.line(color_image, bottomRight, bottomLeft, (0, 255, 0), 2)
                    cv2.line(color_image, bottomLeft, topLeft, (0, 255, 0), 2)
                    # compute and draw the center (x, y)-coordinates of the ArUco
                    # marker
                    cX = int((topLeft[0] + bottomRight[0]) / 2.0)
                    cY = int((topLeft[1] + bottomRight[1]) / 2.0)
                    cZ = depth_frame.get_distance(cX, cY)
                    cXW, cYW, cZW = rs.rs2_deproject_pixel_to_point(color_intrinsics, [cX, cY], cZ)
                    center = [cXW, cYW, cZW]
                    # print(f"Centrul este: {center}")

                    cv2.circle(color_image, (cX, cY), 4, (0, 0, 255), -1)
                    # draw the ArUco marker ID on the image
                    cv2.putText(color_image, str(markerID), (topLeft[0], topLeft[1] - 10), cv2.FONT_HERSHEY_SIMPLEX,
                                0.5, (0, 255, 0), 2)
                    # print("[Inference] ArUco marker ID: {}".format(markerID))
                # show the output image
        #                     cv2.imwrite('./color.jpg',color_image)

        # print(rejectedImgPoints)
        # Display the resulting frame
        #             print("about to show!")

            cv2.startWindowThread()
## if uncommented, crash!!!
            cv2.namedWindow('Detection', cv2.WINDOW_AUTOSIZE)
            cv2.imshow("Detection", color_image)
            cv2.waitKey(1)

        # Stop streaming
        cv2.destroyAllWindows()
        pipeline.stop()
        time.sleep(1)

    def pause(self):
        self.__flag.clear()  # 设置为False, 让线程阻塞

    def resume(self):
        self.__flag.set()  # 设置为True, 让线程停止阻塞

    def stop(self):
        self.__flag.set()  # 将线程从暂停状态恢复, 如何已经暂停的话
        self.__running.clear()  # 设置为False


class RobotMovement(threading.Thread):
    def __init__(self, threadID, name, robot):
        threading.Thread.__init__(self)
        self.threadID = threadID
        self.name = name
        self.robot = robot
        self.__flag = threading.Event()  # Used to pause the thread
        self.__flag.set()  # Set to True
        self.__running = threading.Event()  # Used to stop the thread
        self.__running.set()  # Set to True

    def run(self):
        while self.__running.is_set():
            if np.array_equal(center, [1., 2., 3.]):
                time.sleep(1)
                continue
            self.__flag.wait()  # Return immediately if True, block until True if False
            # Here you can add the logic to move the robot
            # For example, you can move the robot to the detected center
            # global center
            # for i in range(len(center)):
            #     center[i]*=1000
            print(f"Centrul este: {center}")
            coordonate_curente = self.robot.get_coords()
            offset_x = -27
            offset_y = -10
            coordonate_curente[0] = (center[0] * 1000) + coordonate_curente[0] + offset_x
            coordonate_curente[1] = -(center[1] * 1000) + coordonate_curente[1] + offset_y
            coordonate_curente[2] = -200
            self.robot.move_to_coords(coordonate_curente, velocity=20, acceleration=30)
            time.sleep(10)  # Pause for 10 second

    def pause(self):
        self.__flag.clear()  # Set to False, blocking the thread

    def resume(self):
        self.__flag.set()  # Set to True, unblocking the thread

    def stop(self):
        self.__flag.set()  # Resume the thread if it was paused
        self.__running.clear()  # Set to False


def passparam(a):
    return a


# [-273.89602287516453, -840.5474798191608, -180.52987310693314, -14.08951335292566, -172.94230173671454, 7.508495889004134] GET COORDS
# [-238.81717398343955, -863.120060175558, -149.97887714553212, -0.08236355383661133, 179.51579057395574, -0.037654463030787406]
def main():
    robot = Robot('UR5')
    robot.connect()
    j = [-89.8990218000652, -99.54692788735923, -63.709450087129355, -106.74283270629115, 90.03790322418257,
         0.24274084099700702]
    robot.move_joints(j, acceleration=20, velocity=20)
    print(robot.get_coords())
    # pose_poza = robot.get_joints()
    # pose_poza_z = robot.get_coords()
    #pose_poza = [-90, -90, -90, -90, 90, 0]
    # robot.move_joints(pose_poza, velocity=20, acceleration=20)
    thread = cameraDetection(1, "rsArucoDetection", robot=robot)
    thread.start()
#
#     c1 = [-0.4394226670265198, -0.3110799789428711, 0.8950000405311584
# ]
#     c2 = [-0.409, -0.256, 0.858]
#     offset_x = -35.079
#     offset_y = 22.573
#     # pose_poza = [-90, -90, -90, -90, 90, 0]
#     # self.robot.move_joints(pose_poza, velocity=20, acceleration=20)
#     pozitie_poza = robot.get_coords()
#     print(f"Coordonatele curente sunt: {pozitie_poza}")
#     new_set = pozitie_poza.copy()
#
#     new_set[0] = -(c1[0] * 1000) + pozitie_poza[0] + offset_x
#     new_set[1] = (c1[1] * 1000) + pozitie_poza[1] + offset_y
#     print(f"Centru dupa adaugare: {new_set}")
#     new_set[2] = pose_poza_z[2] - c1[2] * 1000 + 70
#     robot.move_to_coords(new_set, velocity=10, acceleration=10)
#     print(f"Coord curente dupa mutare: {robot.get_coords()}")

    #robot.move_to_coords(pozitie_poza, velocity=10, acceleration=10)


    thread2 = RobotMovement(2, "RobotMovement", robot=robot)
    thread2.start()

if __name__ == '__main__':
    print(cv2.__version__)
    # main()
    robot = Robot('UR5')
    robot.connect()
    
    
    pick_up = [-89.8990218000652, -99.54692788735923, -63.709450087129355, -106.74283270629115, 90.03790322418257,
         0.24274084099700702]
    
    home = [-23.81717398343955, -86.120060175558, -149.97887714553212, -0.08236355383661133, 179.51579057395574, -0.037654463030787406]


    #read from file 'command.txt'
    with open('command.txt', 'r') as f:
        command = f.read()
    

    if command == 'free':
        robot.set_freedrive(True)
    elif command == 'pick-up':
        robot.move_joints(pick_up, acceleration=20, velocity=20)
    elif command == 'home':
        robot.move_joints(home, acceleration=20, velocity=20)


        

    
    #test(robot)
    app(robot, thresholdPercent=0.035, pallet_type='full')