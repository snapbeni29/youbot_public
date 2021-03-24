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
import pathfinder as pf
from scipy import ndimage


GRID_WIDTH = 50



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
fsm = 'exploring'
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

start = time.time()
forward_counter = 0
positions = []
diff = []
pos, grid_pos = gm.get_youbot_position(youbotPos)
global_map = -np.ones((GRID_WIDTH, GRID_WIDTH))
boundaries_to_visit = []
route = []
next_waypoint = None
dist_to_waypoint = 999
ten_last_waypoints = []
known_borders = set()
global_dilated = global_map
# Start the demo. 
while True:
    try:
        # Check the connection with the simulator
        if vrep.simxGetConnectionId(clientID) == -1:
            sys.exit('Lost connection to remote API.')

        # Get the position and the orientation of the robot.
        res, youbotPos = vrep.simxGetObjectPosition(clientID, h['ref'], -1, vrep.simx_opmode_buffer)
        vrchk(vrep, res, True) # Check the return value from the previous V-REP call (res) and exit in case of error.
        if len(diff) == 0:
            diff.append(0)
        else:
            diff.append(youbotPos[0] - positions[-1])
        positions.append(youbotPos[0])
        res, youbotEuler = vrep.simxGetObjectOrientation(clientID, h['ref'], -1, vrep.simx_opmode_buffer)
        vrchk(vrep, res, True)

        # Get the distance from the beacons
        # Change the flag to True to constraint the range of the beacons
        beacon_dist = youbot_beacon(vrep, clientID, beacons_handle, h, flag=False)


        if fsm == 'exploring':
            fsm_exp = 'scanning'
            while True:
                # get youbot rotation
                res, youbotEuler = vrep.simxGetObjectOrientation(clientID, h['ref'], -1, vrep.simx_opmode_buffer)
                vrchk(vrep, res)

                # get youbot position
                res, youbotPos = vrep.simxGetObjectPosition(clientID, h['ref'], -1, vrep.simx_opmode_buffer)
                vrchk(vrep, res)

                # get position (x, y) and grid_position (row, col)
                # WARNING : row corresponds on the y axis on the plot
                # and col corresponds to the x axis on the plot
                pos, grid_pos = gm.get_youbot_position(youbotPos)
                
                if fsm_exp == 'scanning':
                    # Get data from the hokuyo - return empty if data is not captured
                    scanned_points, contacts = youbot_hokuyo(vrep, h, vrep.simx_opmode_buffer)


                    # build the grid
                    start_time = time.time()
                    tmp_grid, free_cells = gm.local_grid_map(youbotPos, youbotEuler, scanned_points, contacts)
                    end_time = time.time()
                    #print('grid map generated in : (s) ', end_time - start_time)


                    tmp_dilated = np.zeros([GRID_WIDTH, GRID_WIDTH])
                    obstacles_mask = (tmp_grid == 1)
                    tmp_dilated[obstacles_mask] = 1
                    tmp_dilated = ndimage.morphology.grey_dilation(
                        tmp_dilated, size=(5, 5))
                    unexplored_mask = (tmp_grid == -1)
                    obstacles_mask = (tmp_dilated == 1)
                    tmp_dilated[unexplored_mask] = -1
                    tmp_dilated[obstacles_mask] = 1


                    new_borders = set()
                    for cell in free_cells:
                        border = gm.get_unexplored_neighbour(cell, tmp_dilated)
                        if border != None:
                            new_borders.add(border)
                    
                    for border in new_borders:
                        if global_dilated[border[0]][border[1]] < -0.9:
                            known_borders.add(border)
                    
                    to_remove = []
                    for border in known_borders:
                        if tmp_grid[border[0]][border[1]] > -0.9:
                            to_remove.append(border)
                    for item in to_remove:
                        known_borders.remove(item)
                    
                    # known_borders.remove(grid_pos)

                    
                    # adding new knowledge to the global map
                    global_map = np.maximum.reduce([global_map, tmp_grid])
                    print('position in the grid:', grid_pos)
                    global_map[grid_pos[0]][grid_pos[1]] = 0

                    # create a grid where walls are dilated
                    global_dilated = np.zeros([GRID_WIDTH, GRID_WIDTH])
                    obstacles_mask = (global_map == 1)
                    global_dilated[obstacles_mask] = 1
                    global_dilated = ndimage.morphology.grey_dilation(
                        global_dilated, size=(5, 5))
                    unexplored_mask = (global_map == -1)
                    obstacles_mask = (global_dilated == 1)
                    global_dilated[unexplored_mask] = -1
                    global_dilated[obstacles_mask] = 1


                    # plot the "real" grid
                    plt.imshow(np.flip(global_map, axis=0))
                    plt.colorbar()
                    plt.show()
                    
                    print('switching to state analyze_grid_map')
                    fsm_exp = 'analyze_grid_map'
                
                elif fsm_exp == 'analyze_grid_map':
                    # find the nearest "uncertain" cell
                    target_cell = pf.get_nearest_cell(known_borders, grid_pos)
                    print('target cell: ', target_cell)
                    # compute the path to this cell
                    route = pf.astar(global_dilated, grid_pos, (target_cell[0], target_cell[1]))
                    # if robot is stuck in walls (because of the dilation),
                    # go backward to the last waypoint which was in a free cell
                    if route == False:
                        print('route is False')
                        route = []
                        for wp_x, wp_y in ten_last_waypoints[::-1]:
                            route.append((wp_x, wp_y))
                            if global_dilated[wp_x][wp_y] < .9:
                                break
                    else:
                        print('route is not false :', route)
                    # plot dilated grid
                    plt.imshow(np.flip(global_dilated, axis=0)) # flip it to see it in the right angle
                    plt.colorbar()
                    plt.show()

                    # reverse the route
                    route = route[::-1]
                    print(route)
                    fsm_exp = 'get_next_waypoint'
                elif fsm_exp == 'get_next_waypoint':
                    # if the route is empty, we are arrived to the destination
                    # we have to scan again
                    if len(route) == 0:
                        print('switch to state scanning')
                        fsm_exp = 'scanning'
                    else:
                        # otherwise, we have to find the next waypoint to go to
                        route, next_waypoint = pf.next_waypoint(route, grid_pos, global_dilated)
                        dist_to_waypoint = 999
                        # if detecting walls between robot and first position of the path,
                        # it's probably because we are close to a wall
                        if next_waypoint == None:
                            # go point by point
                            print('popping the first point')
                            next_waypoint = route.pop(0)
                        print('switch to state rotating')
                        fsm_exp = 'rotating'
                elif fsm_exp == 'rotating':
                    # When we know where is the next waypoint, we first rotate
                    # to have the youbot in the youbot's north
                    # get the real position of the center of the waypoint's cell
                    x, y = gm.cell_to_pos(next_waypoint)
                    # compute the angles
                    desired_angle = np.arctan2(y - pos[1], x - pos[0]) + np.pi/2
                    # print(desired_angle)
                    forwBackVel = 0
                    rightVel = 0
                    rotateRightVel = angdiff(youbotEuler[2], desired_angle)
                    if abs(angdiff(youbotEuler[2], desired_angle)) < 0.02:
                        # rotate the youbot until the desired angle is reached
                        rotateRightVel = 0
                        res, youbotEuler = vrep.simxGetObjectOrientation(clientID, h['ref'], -1, vrep.simx_opmode_buffer)
                        fsm_exp = 'forward'
                        print('switching to state forward')
                elif fsm_exp == 'forward':
                    # once the robot is turned in the right direction,
                    # go forward until the destination is reached
                    forwBackVel = -1
                    # compute the distance between the youbot and the waypoint
                    dist = pf.heuristic(pos, gm.cell_to_pos(next_waypoint))
                    # print(next_waypoint)
                    # print(dist)
                    # print(dist_to_waypoint)

                    # if the youbot is really close to the waypoint
                    # or the distance between them is not decreasing anymore, stop the youbot
                    if dist < .4 or dist > dist_to_waypoint:
                        forwBackVel = 0  # Stop the robot.
                        # the waypoint is reached, add it to the last 10 waypoints list
                        if global_dilated[grid_pos[0]][grid_pos[1]] < 0.9:
                            ten_last_waypoints.append(next_waypoint)
                            # keep only the last 10 waypoints
                            if len(ten_last_waypoints) > 10:
                                ten_last_waypoints.pop(0)
                        # if the route is not finished, go to next waypoint
                        if len(route) != 0:
                            print('route (go to next waypoint)')
                            print(route)
                            fsm_exp = 'get_next_waypoint'
                        else:
                            # if the route is finished, start scanning again
                            print("empty route, go to scanning")
                            global_map[next_waypoint[0]][next_waypoint[1]] = 0
                            fsm_exp = 'scanning'
                    # update the last distance to the waypoint
                    if dist < dist_to_waypoint:
                        dist_to_waypoint = dist

                # move the youbot
                h = youbot_drive(vrep, h, forwBackVel, rightVel, rotateRightVel)

                vrep.simxSynchronousTrigger(clientID)
                vrep.simxGetPingTime(clientID)
                

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
            if abs(youbotPos[0] + 2) < .02:
                forwBackVel = 0  # Stop the robot.
                fsm = 'finished'
                end = time.time()
                print('time: ', end - start)
                print('positions:')
                print(positions)
                print('differences:')
                print(diff)
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
            print('Finish')
            time.sleep(3)
            break
        else:
            sys.exit('Unknown state ' + fsm)

        # Update wheel velocities.
        h = youbot_drive(vrep, h, forwBackVel, rightVel, rotateRightVel)

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
