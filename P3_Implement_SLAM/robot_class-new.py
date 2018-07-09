from math import *
import random


### ------------------------------------- ###
# Below, is the robot class
#
# This robot lives in 2D, x-y space, and its motion is
# pointed in a random direction, initially.
# It moves in a straight line until it comes close to a wall 
# at which point it stops.
#
# For measurements, it  senses the x- and y-distance
# to landmarks. This is different from range and bearing as
# commonly studied in the literature, but this makes it much
# easier to implement the essentials of SLAM without
# cluttered math.
#
class robot:
    
    # --------
    # init:
    #   creates a robot with the specified parameters and initializes
    #   the location (self.x, self.y) to the center of the world
    #
    def __init__(self, 
                 world_size = [(0.0,100.0), (0.0,100.0)], 
                 measurement_range = [(0.0,30.0), (0.0,30.0)],
                 motion_noise = [1.0, 1.0], 
                 measurement_noise = [1.0, 1.0],
                 position = [world_size / 2.0,  world_size / 2.0]):
        self.world_size = world_size
        self.measurement_range = measurement_range
        self.position = position
        self.motion_noise = motion_noise
        self.measurement_noise = measurement_noise
        self.landmarks = []
        self.num_landmarks = 0
        self.dof = len(position) # degrees of freedom
    
    
    # returns a positive, random float
    def rand(self):
        return random.random() * 2.0 - 1.0
    
    
    # --------
    # move: attempts to move robot by dx, dy. If outside world
    #       boundary, then the move does nothing and instead returns failure
    #
    def move(self, dx):
        
        pos = []
        for i in range(self.dof):
            pos.append(position[i] + dx[i] + self.rand() * self.motion_noise[i])
        #x = self.x + dx + self.rand() * self.motion_noise
        #y = self.y + dy + self.rand() * self.motion_noise
        chk = [world_size[i][0] < pos[i] < world_size[i][1] for i in range(self.dof)]
        #if x < 0.0 or x > self.world_size or y < 0.0 or y > self.world_size:
        if False in chk:
            return False
        else:
            #self.x = x
            #self.y = y
            self.position = pos
            return True


    # --------
    # sense: returns x- and y- distances to landmarks within visibility range
    #        because not all landmarks may be in this range, the list of measurements
    #        is of variable length. Set measurement_range to -1 if you want all
    #        landmarks to be visible at all times
    #
    
    ## TODO: paste your complete the sense function, here
    ## make sure the indentation of the code is correct
    def sense(self):
        ''' This function does not take in any parameters, instead it references internal variables
            (such as self.landamrks) to measure the distance between the robot and any landmarks
            that the robot can see (that are within its measurement range).
            This function returns a list of landmark indices, and the measured distances (dx, dy)
            between the robot's position and said landmarks.
            This function should account for measurement_noise and measurement_range.
            One item in the returned list should be in the form: [landmark_index, dx, dy].
            '''
           
        measurements = []
        
        ## TODO: iterate through all of the landmarks in a world
        
        ## TODO: For each landmark
        ## 1. compute dx and dy, the distances between the robot and the landmark
        ## 2. account for measurement noise by *adding* a noise component to dx and dy
        ##    - The noise component should be a random value between [-1.0, 1.0)*measurement_noise
        ##    - Feel free to use the function self.rand() to help calculate this noise component
        ## 3. If either of the distances, dx or dy, fall outside of the internal var, measurement_range
        ##    then we cannot record them; if they do fall in the range, then add them to the measurements list
        ##    as list.append([index, dx, dy]), this format is important for data creation done later
        for i, pos in enumerate(self.landmarks):
            dx = [pos[i] - self.position[i] + self.rand() * self.measurement_noise[i] for i in range(self.dof)]
            #dx = (x - self.x) + self.rand() * self.measurement_noise
            #dy = (y - self.y) + self.rand() * self.measurement_noise
            chk = [self.measurement_range[i][0] < dx[i] < self.measurement_range[i][1] for i in range(self.dof)]
            #if (self.measurement_range-abs(dx)) > 0.0 and (self.measurement_range-abs(dy)) > 0.0:
            if False not in chk:
                #measurements.append([i, dx, dy])
                measurements.append([i, *dx])
        ## TODO: return the final, complete list of measurements
        return measurements


    # --------
    # make_landmarks:
    # make random landmarks located in the world
    #
    def make_landmarks(self, num_landmarks):
        self.landmarks = []
        for j in range(num_landmarks):
            #self.landmarks.append([round(random.random() * self.world_size),
            #                       round(random.random() * self.world_size)])
            self.landmarks.append([round(random.uniform(*self.world_size[i])) for i in range(self.dof)])
            
        self.num_landmarks = num_landmarks


    # called when print(robot) is called; prints the robot's location
    def __repr__(self):
        #return 'Robot: [x=%.5f y=%.5f]'  % (self.x, self.y)
        return 'Robot: {}'.format('{%.5f} '.join(self.position))



####### END robot class #######