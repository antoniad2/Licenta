import logging
from abc import ABC, abstractmethod
from Interfaces.robot_interface.robots.CRX10.fanucpy.robot import Robot as fanucRobot
from Interfaces.robot_interface.robots.UR5.urx.urrobot import URRobot as urx
from Interfaces.robot_interface.robots.UR5.urx.robotiq_two_finger_gripper import Robotiq_Two_Finger_Gripper
import pandas as pd
import math


class Robot(ABC):
    """
    This is the base class for all robots. It is an abstract class, meaning it cannot be instantiated directly.
    Instead, it should be subclassed, and at least the methods move_to_coords, move_joints, get_coords,
    and get_joints should be overridden.
    """

    def __new__(cls, data):
        """
        The constructor of the Robot class. It initializes the robot based on the type specified in the data.
        If the type is 'CRX10', it initializes a Fanuc robot. If the type is 'UR5', it initializes a UR5 robot.

        Args:
            data (str): The data containing the type of the robot.
            filename (str): The filename where the robot data is stored.
        """
        if cls is Robot:
            if data == 'CRX10':
                return super(Robot, cls).__new__(CRX10)
            elif data == 'UR5':
                return super(Robot, cls).__new__(UR5)
        return super(Robot, cls).__new__(cls)

    def __init__(self, data):
        pass

    @abstractmethod
    def move_to_coords(self, coords, velocity, acceleration, cnt_val=0, linear=0):
        """
        This is an abstract method that should be overridden in a subclass.
        It should move the robot to the specified coordinates.

        Args:
            coords (list): The coordinates to move the robot to.
            velocity (float): The velocity at which to move the robot.
            acceleration (float): The acceleration of the robot movement.
            cnt_val (int, optional): The continuous value for stopping. Defaults to 0.
            linear (int, optional): Whether the movement is linear or not. Defaults to 0.
        """
        pass

    @abstractmethod
    def move_joints(self, joints, velocity, acceleration, cnt_val=0, linear=0):
        """
        This is an abstract method that should be overridden in a subclass.
        It should move the robot's joints to the specified positions.

        Args:
            joints (list): The joint positions to move the robot to.
            velocity (float): The velocity at which to move the robot.
            acceleration (float): The acceleration of the robot movement.
            cnt_val (int, optional): The continuous value for stopping. Defaults to 0.
            linear (int, optional): Whether the movement is linear or not. Defaults to 0.
        """
        pass

    @abstractmethod
    def move_xs(self, command, pose_list, acceleration, velocity, radius):
        pass

    @abstractmethod
    def get_coords(self):
        """
        This is an abstract method that should be overridden in a subclass.
        It should return the current coordinates of the robot.
        """
        pass

    @abstractmethod
    def get_joints(self):
        """
        This is an abstract method that should be overridden in a subclass.
        It should return the current joint positions of the robot.
        """
        pass

    @abstractmethod
    def freedrive(self, enable=True, timeout=5000):
        pass

    @abstractmethod
    def connect(self):
        pass

    @abstractmethod
    def disconnect(self):
        pass

    @abstractmethod
    def gripper_action(self, action):
        pass

    @abstractmethod
    def gripper_open(self):
        pass

    @abstractmethod
    def gripper_close(self):
        pass
    
    @abstractmethod
    def convertCameraToRobot(self, x, y, z, angle):
        pass


class CRX10(Robot):
    """
    This is the CRX10 class, a subclass of Robot. It overrides the abstract methods of the Robot class.
    """

    def __init__(self, data):
        """
        The constructor of the UR5 class. It calls the constructor of the superclass (Robot) to initialize the robot.

        Args:
            data (str): The data containing the type of the robot.
            filename (str): The filename where the robot data is stored.
        """
        super().__init__(data)  # ASTA POATE FI SCOASA DAR DA WARNING / NU INFLUENTEAZA CU NIMIC PT CA NU E IMPLEMENTATA IN CLASA PARINTE
        self._robot = fanucRobot(robot_model="Fanuc",
                                 host="192.168.0.111",
                                 port=18735,
                                 ee_DO_type="RDO",
                                 ee_DO_num=7)

    def move_to_coords(self, coords, velocity, acceleration, cnt_val=50, linear=0):
        """
        This method moves the CRX10 robot to the specified coordinates.

        Args:
            coords (list): The coordinates to move the robot to.
            velocity (float): The velocity at which to move the robot.
            acceleration (float): The acceleration of the robot movement.
            cnt_val (int, optional): The continuous value for stopping. Defaults to 0.
            linear (int, optional): Whether the movement is linear or not. Defaults to 0.
        """
        #print("Jpos of current position:", self._robot.get_curjpos())
        self._robot.move("pose", vals=coords, velocity=velocity, acceleration=acceleration, cnt_val=cnt_val,
                         linear=linear)

    def move_joints(self, coords, velocity, acceleration, cnt_val=50, linear=0):
        """
        This method moves the joints of the CRX10 robot to the specified positions.

        Args:
            coords (list): The joint positions to move the robot to.
            velocity (float): The velocity at which to move the robot.
            acceleration (float): The acceleration of the robot movement.
            cnt_val (int, optional): The continuous value for stopping. Defaults to 0.
            linear (int, optional): Whether the movement is linear or not. Defaults to 0.
        """
        #print("Jpos of current position:", self._robot.get_curjpos())
        self._robot.move("joint", vals=coords, velocity=velocity, acceleration=acceleration, cnt_val=cnt_val,
                         linear=linear)

    def get_coords(self):
        """
        This method returns the current coordinates of the CRX10 robot.
        """
        #print(f"Current pose:", self._robot.get_curpos())
        return self._robot.get_curpos()

    def get_joints(self):
        """
        This method returns the current joint positions of the CRX10 robot.
        """
        #print(f"Current joints:", self._robot.get_curjpos())
        return self._robot.get_curjpos()

    def connect(self):
        self._robot.connect()

    def disconnect(self):
        self._robot.disconnect()

    def gripper_action(self, action):
        pass

    def gripper_close(self):
        pass

    def gripper_open(self):
        pass
    
    def freedrive(self, enable=True, timeout=60):
        pass

    def convertCameraToRobot(self, xC, yC, angle):
        a = -0.006085198660343979
        b = 1.8581266655422601
        c = 1.9186277618485514
        d = -0.039582054309345534
        tx = 324.3707478254586
        ty = -708.2134371732161

        xR = a * xC + b * yC + tx
        yR = c * xC + d * yC + ty
                
        return xR, yR, 270 - angle

class UR5(Robot):
    """
    This is the UR5 class, a subclass of Robot. It overrides the abstract methods of the Robot class.
    """

    def __init__(self, data):
        """
        The constructor of the UR5 class. It calls the constructor of the superclass (Robot) to initialize the robot.

        Args:
            data (str): The data containing the type of the robot.
            filename (str): The filename where the robot data is stored.
        """
        super().__init__(data)  # ASTA POATE FI SCOASA DAR DA WARNING / NU INFLUENTEAZA CU NIMIC PT CA NU E IMPLEMENTATA IN CLASA PARINTE
        logging.basicConfig(level=logging.WARN)
        self._robot = urx('192.168.0.132')
        self._robot.set_payload(1.2, (0, 0, 0))
        self._robot.set_tcp((0, 0, 0.151, 0, 0, 0))
        self._robot.stopj()
        #self.gripper = Robotiq_Two_Finger_Gripper(self._robot)
        #self.gripper.gripper_action(True)

    def move_to_coords(self, coords, velocity, acceleration, cnt_val=0, linear=0):
        """
        This method moves the UR5 robot to the specified coordinates.

        Args:
            coords (list): The coordinates to move the robot to.
            velocity (float): The velocity at which to move the robot.
            acceleration (float): The acceleration of the robot movement.
            cnt_val (int, optional): The continuous value for stopping. Defaults to 0.
            linear (int, optional): Whether the movement is linear or not. Defaults to 0.
        """
        velocity = velocity / 100
        acceleration = acceleration / 100
        coords = [coords[0] / 1000, coords[1] / 1000, coords[2] / 1000, math.radians(coords[3]),
                   math.radians(coords[4]), math.radians(coords[5])]

        #coords = [coords[0]/1000, coords[1]/1000, coords[2]/1000, coords[3], coords[4], coords[5]]
        self._robot.movel(tpose=coords, vel=velocity, acc=acceleration)

    def move_joints(self, joints, velocity, acceleration, cnt_val=0, linear=0):
        """
        This method moves the joints of the UR5 robot to the specified positions.

        Args:
            joints (list): The joint positions to move the robot to.
            velocity (float): The velocity at which to move the robot.
            acceleration (float): The acceleration of the robot movement.
            cnt_val (int, optional): The continuous value for stopping. Defaults to 0.
            linear (int, optional): Whether the movesment is linear or not. Defaults to 0.
        """
        velocity = velocity / 100
        acceleration = acceleration / 100
        joints = [math.radians(joints[0]), math.radians(joints[1]), math.radians(joints[2]), math.radians(joints[3]),
                  math.radians(joints[4]), math.radians(joints[5])]
        #joints = [joints[0], joints[1], joints[2], joints[3], joints[4], joints[5]]
        self._robot.movej(joints=joints, vel=velocity, acc=acceleration)

    def move_xs(self, command, pose_list, acceleration, velocity, radius):
        velocity = velocity / 100
        acceleration = acceleration / 100
        radius = radius / 100
        coords=[]
        newposes=[]

        if command == 'movel':
            for coords in pose_list:
                coords = [coords[0] / 1000, coords[1] / 1000, coords[2] / 1000, math.radians(coords[3]),
                    math.radians(coords[4]), math.radians(coords[5])]
                newposes.append(coords)
        elif command == 'movej':
            for joints in pose_list:
                joints = [math.radians(joints[0]), math.radians(joints[1]), math.radians(joints[2]), math.radians(joints[3]),
                  math.radians(joints[4]), math.radians(joints[5])]
                newposes.append(joints)

        self._robot.movexs(command=command, pose_list=newposes, acc=acceleration, vel=velocity, radius=radius)


    def get_coords(self):
        """
        This method returns the current coordinates of the UR5 robot.
        """
        coords = self._robot.getl()
        coords = [coords[0] * 1000, coords[1] * 1000, coords[2] * 1000, math.degrees(coords[3]),
                  math.degrees(coords[4]), math.degrees(coords[5])]
        #print(f"Current pose:", coords)
        return coords

    def get_joints(self):
        """
        This method returns the current joint positions of the UR5 robot.
        """
        coords = self._robot.getj()
        coords = [math.degrees(coords[0]), math.degrees(coords[1]), math.degrees(coords[2]), math.degrees(coords[3]),
                  math.degrees(coords[4]), math.degrees(coords[5])]
        #print(f"Current joints:", coords)
        return coords

    def freedrive(self, enable=True, timeout=60):
        if enable:
            self._robot.set_freedrive(timeout)
        else:
            self._robot.set_freedrive(False)

    def gripper_action(self, action):
        self.gripper.gripper_action(action)

    def gripper_close(self):
        self._robot.set_digital_out(0, False)

    def gripper_open(self):
        self._robot.set_digital_out(0, True)

    def connect(self):
        pass

    def disconnect(self):
        self._robot.close()
        
    def convertCameraToRobot(self, xC, yC, angle):
        a = -0.05324761290130534
        b = 1.9175380730856597
        c = 1.9067066688332064
        d = 0.025175112481834398
        tx = -1061.9398393492365
        ty = -797.2170478784193

        xR = a * xC + b * yC + tx
        yR = c * xC + d * yC + ty
            
        return xR, yR, 180 - angle
