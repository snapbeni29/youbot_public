# -*- coding: utf-8 -*-
"""
The aim of this code is to show small examples of controlling the displacement of the robot in V-REP. 

(C) Copyright Renaud Detry 2013, Mathieu Baijot 2017, Norman Marlier 2019.
Distributed under the GNU General Public License.
(See http://www.gnu.org/copyleft/gpl.html)
"""
# VREP
import sim as vrep

# Useful import
import time
import numpy as np
import sys

from cleanup_vrep import cleanup_vrep
from vrchk import vrchk
from youbot_init import youbot_init
from youbot_drive import youbot_drive
from youbot_hokuyo_init import youbot_hokuyo_init
from youbot_hokuyo import youbot_hokuyo
from youbot_xyz_sensor import youbot_xyz_sensor
from beacon import beacon_init, youbot_beacon
from utils_sim import angdiff
import matplotlib.pyplot as plt
import grid_map as gm

# Test the python implementation of a youbot
# Initiate the connection to the simulator.
print('Program started')
# Use the following line if you had to recompile remoteApi
# vrep = remApi('remoteApi', 'extApi.h')
# vrep = remApi('remoteApi')

# Close the connection in case if a residual connection exists
vrep.simxFinish(-1)
clientID = vrep.simxStart('127.0.0.1',  19997, True, True, 2000, 5)

# The time step the simulator is using (your code should run close to it).
timestep = .05

# Synchronous mode
returnCode = vrep.simxSynchronous(clientID, True)

# If you get an error like:
#   Remote API function call returned with error code: 64.
# Explanation: simxStart was not yet called.
# Make sure your code is within a function!
# You cannot call V-REP from a script.
if clientID < 0:
    sys.exit('Failed connecting to remote API server. Exiting.')

print('Connection ' + str(clientID) + ' to remote API server open')

# Make sure we close the connection whenever the script is interrupted.
# cleanup_vrep(vrep, id)

# This will only work in "continuous remote API server service".
# See http://www.v-rep.eu/helpFiles/en/remoteApiServerSide.htm
vrep.simxStartSimulation(clientID, vrep.simx_opmode_blocking)

# Send a Trigger to the simulator: this will run a time step for the physics engine
# because of the synchronous mode. Run several iterations to stabilize the simulation
for i in range(int(1./timestep)):
    vrep.simxSynchronousTrigger(clientID)
    vrep.simxGetPingTime(clientID)

# Retrieve all handles, mostly the Hokuyo.
h = youbot_init(vrep, clientID)
h = youbot_hokuyo_init(vrep, h)
beacons_handle = beacon_init(vrep, clientID)

# Send a Trigger to the simulator: this will run a time step for the physics engine
# because of the synchronous mode. Run several iterations to stabilize the simulation
for i in range(int(1./timestep)):
    vrep.simxSynchronousTrigger(clientID)
    vrep.simxGetPingTime(clientID)



##############################################################################
#                                                                            #
#                          INITIAL CONDITIONS                                #
#                                                                            #
##############################################################################
# Define all the variables which will be used through the whole simulation.
# Important: Set their initial values.

# Get the position of the beacons in the world coordinate frame (x, y)
beacons_world_pos = np.zeros((len(beacons_handle), 3))
for i, beacon in enumerate(beacons_handle):   
    res, beacons_world_pos[i] = vrep.simxGetObjectPosition(clientID, beacon, -1,
                                                           vrep.simx_opmode_buffer)

# Parameters for controlling the youBot's wheels: at each iteration,
# those values will be set for the wheels.
# They are adapted at each iteration by the code.
forwBackVel = 0  # Move straight ahead.
rightVel = 0  # Go sideways.
rotateRightVel = 0  # Rotate.

# First state of state machine
fsm = 'rotateRight'
print('Switching to state: ', fsm)

# Get the initial position
res, youbotPos = vrep.simxGetObjectPosition(clientID, h['ref'], -1, vrep.simx_opmode_buffer)
# Set the speed of the wheels to 0.
h = youbot_drive(vrep, h, forwBackVel, rightVel, rotateRightVel)

# Send a Trigger to the simulator: this will run a time step for the physic engine
# because of the synchronous mode. Run several iterations to stabilize the simulation
for i in range(int(1./timestep)):
    vrep.simxSynchronousTrigger(clientID)
    vrep.simxGetPingTime(clientID)

# Start the demo. 
while True:
    try:
        # Check the connection with the simulator
        if vrep.simxGetConnectionId(clientID) == -1:
            sys.exit('Lost connection to remote API.')

        # Get the position and the orientation of the robot.
        res, youbotPos = vrep.simxGetObjectPosition(clientID, h['ref'], -1, vrep.simx_opmode_buffer)
        vrchk(vrep, res, True) # Check the return value from the previous V-REP call (res) and exit in case of error.
        res, youbotEuler = vrep.simxGetObjectOrientation(clientID, h['ref'], -1, vrep.simx_opmode_buffer)
        vrchk(vrep, res, True)

        # Get the distance from the beacons
        # Change the flag to True to constraint the range of the beacons
        beacon_dist = youbot_beacon(vrep, clientID, beacons_handle, h, flag=False)


        if fsm == 'exploring':
            fsm_exp = 'scanning'
            while True:
                
                if fsm_exp == 'scanning':
                    # Get data from the hokuyo - return empty if data is not captured
                    scanned_points, contacts = youbot_hokuyo(vrep, h, vrep.simx_opmode_buffer)

                    # get youbot rotation
                    res, youbotEuler = vrep.simxGetObjectOrientation(clientID, h['ref'], -1, vrep.simx_opmode_buffer)
                    vrchk(vrep, res)

                    # get youbot position
                    res, youbotPos = vrep.simxGetObjectPosition(clientID, h['ref'], -1, vrep.simx_opmode_buffer)
                    vrchk(vrep, res)

                    # build the grid
                    start_time = time.time()
                    tmp_grid, free_cells = gm.local_grid_map(youbotPos, youbotEuler, scanned_points, contacts)
                    end_time = time.time()
                    print('grid map generated in : (s) ', end_time - start_time)

                    # MERGE GRID_MAPS !!

                    ## Uncomment the following lines to plot the grid map
                    #plt.imshow(np.flip(tmp_grid, axis=0)) # flip it to see it in the right angle
                    #plt.colorbar()
                    #plt.show()
                    
                    fsm_exp = 'planning'
                
                elif fsm_exp == 'planning':
                    unknown_boundaries = gm.get_unknown_boundaries(tmp_grid, free_cells)
                    most_uncertain_cell = gm.get_most_uncertain_cell(grid_map, unknown_boundaries, 5)
                    orthonormal_pos, grid_pos = gm.get_youbot_position(youbotPos, grid_size, house_size)
                    # TODO: Compute the shortest path to the most uncertain cell 


        # Apply the state machine.
        elif fsm == 'forward':

            # Make the robot drive with a constant speed (very simple controller, likely to overshoot). 
            # The speed is - 1 m/s, the sign indicating the direction to follow. Please note that the robot has
            # limitations and cannot reach an infinite speed. 
            forwBackVel = -1

            # Stop when the robot is close to y = - 6.5. The tolerance has been determined by experiments: if it is too
            # small, the condition will never be met (the robot position is updated every 50 ms); if it is too large,
            # then the robot is not close enough to the position (which may be a problem if it has to pick an object,
            # for example). 
            if abs(youbotPos[0] + 6.5) < .02:
                forwBackVel = 0  # Stop the robot.
                fsm = 'backward'
                print('Switching to state: ', fsm)


        elif fsm == 'backward':
            # A speed which is a function of the distance to the destination can also be used. This is useful to avoid
            # overshooting: with this controller, the speed decreases when the robot approaches the goal. 
            # Here, the goal is to reach y = -4.5. 
            forwBackVel = - 2 * (youbotPos[1] + 4.5)
            # distance to goal influences the maximum speed

            # Stop when the robot is close to y = 4.5.
            if abs(youbotPos[1] + 4.5) < .01:
                forwBackVel = 0  # Stop the robot.
                fsm = 'right'
                print('Switching to state: ', fsm)
        elif fsm == 'right':
            # Move sideways, again with a proportional controller (goal: x = - 4.5). 
            rightVel = - 2 * (youbotPos[0] + 4.5)

            # Stop at x = - 4.5
            if abs(youbotPos[0] + 4.5) < .01:
                rightVel = 0  # Stop the robot.
                fsm = 'rotateRight'
                print('Switching to state: ', fsm)

        elif fsm == 'rotateRight':
            # Rotate until the robot has an angle of -pi/2 (measured with respect to the world's reference frame). 
            # Again, use a proportional controller. In case of overshoot, the angle difference will change sign, 
            # and the robot will correctly find its way back (e.g.: the angular speed is positive, the robot overshoots, 
            # the anguler speed becomes negative). 
            # youbotEuler(3) is the rotation around the vertical axis.              
            rotateRightVel = angdiff(youbotEuler[2], (-np.pi/2))
            res, youbotEuler = vrep.simxGetObjectOrientation(clientID, h['ref'], -1, vrep.simx_opmode_buffer)
            print(youbotEuler)
            fsm = 'finished'
            continue
            # Stop when the robot is at an angle close to -pi/2.
            if abs(angdiff(youbotEuler[2], (-np.pi/2))) < .002:
                rotateRightVel = 0
                res, youbotEuler = vrep.simxGetObjectOrientation(clientID, h['ref'], -1, vrep.simx_opmode_buffer)
                print(youbotEuler)
                # fsm = 'finished'
                print('Switching to state: ', fsm)

        elif fsm == 'finished':
            rotateRightVel = 0
            print('Finish')
            #time.sleep(3)
            break
        else:
            sys.exit('Unknown state ' + fsm)

        # Update wheel velocities.
        h = youbot_drive(vrep, h, forwBackVel, rightVel, rotateRightVel)

        res, youbotEuler = vrep.simxGetObjectOrientation(clientID, h['ref'], -1, vrep.simx_opmode_buffer)
        print(youbotEuler)

        # What happens if you do not update the velocities?
        # The simulator always considers the last speed you gave it,
        # until you set a new velocity.

        # Send a Trigger to the simulator: this will run a time step for the physic engine
        # because of the synchronous mode.
        vrep.simxSynchronousTrigger(clientID)
        vrep.simxGetPingTime(clientID)
    except KeyboardInterrupt:
        cleanup_vrep(vrep, clientID)
        sys.exit('Stop simulation')

cleanup_vrep(vrep, clientID)
print('Simulation has stopped')