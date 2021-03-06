# -*- coding: utf-8 -*-
import numpy as np
from bresenham import bresenham
import math

GRID_WIDTH = 50

def get_youbot_position(position, grid_size=GRID_WIDTH, house_size=15):
    # compute the position of the youbot w.r.t. the orthonormal of the house 
    # (translated to not have the origin in the middle of the house)
    o_x = position[0] + (house_size/2)
    o_y = position[1] + (house_size/2)

    # transform the orthonormal position of the youbot to a grid position (row, col)
    o_x_grid = int((o_x / house_size) * grid_size)
    o_y_grid = int((o_y / house_size) * grid_size)
    # WARNING : o_y_grid and o_x_grid are inversed
    # because y corresponds to the row in the plot
    # and x corresponds to the column
    return (o_x, o_y), (o_y_grid, o_x_grid)

def local_grid_map(position, rotation, scanned_points, contacts, grid_size=GRID_WIDTH, house_size=15):
    
    # stack the data returned by the 2 hokuyo sensors
    contacts = np.hstack((contacts[1], contacts[0])).T
    scanned_points_x = np.hstack((scanned_points[3], scanned_points[0])).T
    scanned_points_y = np.hstack((scanned_points[4], scanned_points[1])).T
    
    # compute the positions of the points returned by the hokuyo sensors w.r.t. the orthonormal of the house
    x = scanned_points_x * np.cos(-rotation[2]) + scanned_points_y * np.sin(-rotation[2])
    y = -scanned_points_x * np.sin(-rotation[2]) + scanned_points_y * np.cos(-rotation[2])

    orthonormal_pos, grid_pos = get_youbot_position(position, grid_size, house_size)
    o_x = orthonormal_pos[0]
    o_y = orthonormal_pos[1]
    o_y_grid = grid_pos[0]
    o_x_grid = grid_pos[1]
    
    # translate the points returned by the hokuyo sensors too
    x = x + o_x
    y = y + o_y

    # initialize the grid with -1 values
    # the grid cells can take the following values:
    #   -1: unexplored
    #    0: free cell
    #    1: obstacle
    #    2: youbot
    grid_map = -np.ones((grid_size, grid_size))

    # transform all points returned by the hokuyos to grid positions
    j = ((x / house_size) * grid_size).astype(int)
    i = ((y / house_size) * grid_size).astype(int)

    # do the same for those which sensed contact
    contact_j = j[contacts].T.tolist()[0]
    contact_i = i[contacts].T.tolist()[0]

    j = j.T.tolist()[0]
    i = i.T.tolist()[0]

    free_cells = set()

    # apply the bresenham algorithm and update free cells in the grid
    for a, b in zip(j, i):
        line = list(bresenham(o_x_grid, o_y_grid, a, b))
        free_cells.update(line)
        cells = np.transpose(line)
        for cell_x, cell_y in zip(cells[0], cells[1]):
            grid_map[cell_y, cell_x] = 0
    # update obstacles position in the grid
    grid_map[contact_i, contact_j] = 1

    free_cells = np.argwhere(grid_map == 0)
    
    ## Uncomment to see where the youbot is
    #grid_map[o_y_grid, o_x_grid] = 2
    
    return grid_map, free_cells

def remove_free_cells(cells, grid_map):
    to_remove = []
    for x, y in cells:
        if grid_map[x][y] == 0:
            to_remove.append((x, y))
    for item in to_remove:
        cells.remove(item)
    return cells

def cell_to_pos(cell, grid_size=GRID_WIDTH, house_size=15):
    # Returns the "real" position of the center of the cell
    cell_width = house_size/grid_size
    y = cell[0] * cell_width + cell_width/2
    x = cell[1] * cell_width + cell_width/2
    return (x, y)

def findBorders(map):
    limit = len(map)
    borders_coordinates = []
    for i in range(len(map)):
        for j in range(len(map[i])):
            if map[i][j] == -1:
                if i+1 < limit and map[i+1][j] == 0:
                    borders_coordinates.append((j, i))
                elif i+1 < limit and j+1 < limit and map[i+1][j+1] == 0:
                    borders_coordinates.append((j, i))
                elif i+1 < limit and j-1 >= 0 and map[i+1][j-1] == 0:
                    borders_coordinates.append((j, i))
                elif i-1 >= 0 and map[i-1][j] == 0:
                    borders_coordinates.append((j, i))
                elif i-1 >= 0 and j+1 < limit and map[i-1][j+1] == 0:
                    borders_coordinates.append((j, i))
                elif i-1 >= 0 and j-1 >= 0 and map[i-1][j-1] == 0:
                    borders_coordinates.append((j, i))
                elif j+1 < limit and map[i][j+1] == 0:
                    borders_coordinates.append((j, i))
                elif j-1 >= 0 and map[i][j-1] == 0:
                    borders_coordinates.append((j, i))
    """for e in borders_coordinates:
        map[e[1]][e[0]] = 2
    plt.imshow(np.flip(map, axis=0))
    plt.colorbar()
    plt.show()"""
    return borders_coordinates

def has_unexplored_neighbour(cell, grid):
    min_x = cell[0]-1 if cell[0] > 0 else 0
    max_x = cell[0]+2 if cell[0] < len(grid)-1 else len(grid)
    min_y = cell[1]-1 if cell[1] > 0 else 0
    max_y = cell[1]+2 if cell[1] < len(grid)-1 else len(grid)
    neighbours = grid[min_x:max_x, min_y:max_y]
    _min = np.min(neighbours)
    return _min < 0.9
        



def get_unexplored_neighbour(cell, grid):
    min_x = cell[0]-1 if cell[0] > 0 else 0
    max_x = cell[0]+2 if cell[0] < len(grid)-1 else len(grid)
    min_y = cell[1]-1 if cell[1] > 0 else 0
    max_y = cell[1]+2 if cell[1] < len(grid)-1 else len(grid)
    neighbours = grid[min_x:max_x, min_y:max_y]
    _min = np.argmin(neighbours)
    row = math.floor(_min/3)
    col = _min % 3
    return (cell[0]-1 + row, cell[1]-1 + col)  if neighbours[row][col] < 0.9 else None
        


def findClosestBorder(map, youbotX, youbotY):
    # print(free_cells)
    borders_coordinates = findBorders(map)
    xy_closestBorderCoordinate = []
    minDist = 999
    for e in borders_coordinates:
        dist = euclideanDistance(youbotX, youbotY, e[0], e[1])
        if dist < minDist:
            minDist = dist
            xy_closestBorderCoordinate = [e[0], e[1]]
    #print("Unknown boundaries : ")
    # print(unknown_boundaries)
    map[xy_closestBorderCoordinate[1]][xy_closestBorderCoordinate[0]] = 2
    plt.imshow(np.flip(map, axis=0))
    plt.colorbar()
    plt.show()
    return minDist, xy_closestBorderCoordinate




"""
def get_unknown_boundaries(grid_map, free_cells):
    unknown_boundaries = []
    for x, y in free_cells:
        is_boundary = False
        for j in range(y-1, y+2):
            for i in range(x-1, x+2):
                if grid_map[i][j] < -0.1:
                    unknown_boundaries.append((i, j))
                    is_boundary = True
                if is_boundary:
                    break
            if is_boundary:
                break
    return unknown_boundaries

def get_most_uncertain_cell(grid_map, cells, window_half_size):
    most_uncertain_cell = None
    biggest_uncertainty = 0
    for x, y in cells:
        cell_uncertainty = 0
        for j in range(y - window_half_size, y + window_half_size + 1):
            for i in range(x - window_half_size, x + window_half_size + 1):
                if grid_map[i][j] < -0.1:
                    cell_uncertainty += 1
        cell_uncertainty /= ((window_half_size * 2) + 1) ** 2
        if biggest_uncertainty < cell_uncertainty:
            biggest_uncertainty = cell_uncertainty
            most_uncertain_cell = (x, y)
    return most_uncertain_cell
"""