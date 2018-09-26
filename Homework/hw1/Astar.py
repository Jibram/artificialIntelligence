# Loop through all mazes first and call function that solves them based on the read width and height
import sys
import numpy as np
import heapq as heap


# Linear solution using a queue/heap might be optimal


def manhattan_distance(x, y, height, width):
    return (height - 1) - x + (width - 1) - y


def path(best_previous, current):
    solution = [current]
    while current in best_previous:
        current = best_previous[current]
        solution.append(current)
    return solution


def solve_maze(maze_matrix, w):

    height = maze_matrix.shape[0]
    width = maze_matrix.shape[1]

    traveled = []
    hedge = []
    best_previous = {}
    dist_matrix = np.zeros((height, width))
    value_matrix = np.zeros((height, width))
    dist_matrix[:][:] = sys.maxsize
    value_matrix[:][:] = sys.maxsize
    dist_matrix[0][0] = 0
    value_matrix[0][0] = manhattan_distance(0,0, height, width)
    start = (0, 0)
    heap.heappush(hedge, (value_matrix[0][0], start))
    while bool(hedge):
        current = heap.heappop(hedge)[1]
        w.write(str(current[0]) + ' ' + str(current[1]) + '\n')
        if current[0] == height - 1 and current[1] == width - 1:
            return path(best_previous, current)

        traveled.append(current)
        #Up
        if (current[0] > 0) and (maze_matrix[current[0] - 1][current[1]] == 0):
            up = (current[0] - 1, current[1])
            for i in range(1):

                if up in traveled:
                    continue

                tent_dist = dist_matrix[current[0]][current[1]] + 1
                if tent_dist >= dist_matrix[up[0]][up[1]]:
                    continue

                best_previous[up] = current
                dist_matrix[up[0]][up[1]] = tent_dist
                value_matrix[up[0]][up[1]] = tent_dist + manhattan_distance(up[0],up[1],height, width)

                if up not in hedge:
                    heap.heappush(hedge, (value_matrix[up[0]][up[1]], up))

        #Down
        if current[0] < height - 1 and (maze_matrix[current[0] + 1][current[1]] == 0):

            down = (current[0] + 1, current[1])
            for i in range(1):
                if down in traveled:
                    continue

                tent_dist = dist_matrix[current[0]][current[1]] + 1
                if tent_dist >= dist_matrix[down[0]][down[1]]:
                    continue

                best_previous[down] = current
                dist_matrix[down[0]][down[1]] = tent_dist
                value_matrix[down[0]][down[1]] = tent_dist + manhattan_distance(down[0],down[1],height, width)

                if down not in hedge:
                    heap.heappush(hedge, (value_matrix[down[0]][down[1]], down))
        #Left
        if current[1] > 0 and (maze_matrix[current[0]][current[1] - 1] == 0):
            left = (current[0], current[1] - 1)
            for i in range(1):

                if left in traveled:
                    continue

                tent_dist = dist_matrix[current[0]][current[1]] + 1
                if tent_dist >= dist_matrix[left[0]][left[1]]:
                    continue

                best_previous[left] = current
                dist_matrix[left[0]][left[1]] = tent_dist
                value_matrix[left[0]][left[1]] = tent_dist + manhattan_distance(left[0], left[1], height, width)

                if left not in hedge:
                    heap.heappush(hedge, (value_matrix[left[0]][left[1]], left))

        #Right
        if current[1] < width - 1 and (maze_matrix[current[0]][current[1] + 1] == 0):
            right = (current[0], current[1] + 1)
            for i in range(1):
                if right in traveled:
                    continue

                #hedge add should be here maybe?

                tent_dist = dist_matrix[current[0]][current[1]] + 1
                if tent_dist >= dist_matrix[right[0]][right[1]]:
                    continue

                best_previous[right] = current
                dist_matrix[right[0]][right[1]] = tent_dist
                value_matrix[right[0]][right[1]] = tent_dist + manhattan_distance(right[0], right[1], height, width)

                if right not in hedge:
                    heap.heappush(hedge, (value_matrix[right[0]][right[1]], right))


def maze_to_matrix():

    f = open('input.txt', 'r')
    w = open('output.txt', 'w')
    maze_count = int(f.readline())
    for i in range(maze_count):
        dimensions = f.readline()
        dimensions = f.readline().split()
        height = int(dimensions[0])
        width = int(dimensions[1])
        maze_matrix = np.zeros((height, width))
        for j in range(height):
            row = f.readline().split(',')
            for k in range(width):
                maze_matrix[j][k] = int(row[k])
        list = solve_maze(maze_matrix, w)
        for i in range(len(list)-1, -1, -1):
            w.write(str(list[i]))
        w.write('\n')
        w.write('\n')
    return

maze_to_matrix()