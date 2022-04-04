import random
import time


class Node():
    #A node class for A* Pathfinding algorithm

    def __init__(self, parent=None, position=None):
        self.parent = parent
        self.position = position

        self.g = 0 # Distance between current node and the start node
        self.h = 0 # Distance between current node and the end node. The heuristic function
        self.f = 0 # Total cost function. This we want to minimize

    def __eq__(self, other):
        return self.position == other.position

def astar(maze, start, end):
    # This returns a list, containing the path from start to end as tuples

    # Create start and end node
    start_node = Node(None, start) # Start node with start as the position and no parent
    start_node.g = start_node.h = start_node.f = 0
    end_node = Node(None, end) # End node with end as position and no parent
    end_node.g = end_node.h = end_node.f = 0

    # Create a timer variable
    timer = 0

    # Initialize both frontier and closed list
    frontier = [] # Nodes that is potential part of the path
    closed_list = [] # Nodes that have been looked upon

    # Add the start node
    frontier.append(start_node)

    # Loop until you find the end. i.e when there are no more nodes in frontier
    while len(frontier) > 0:
        # Add the amount of time since the last iteration
        timer += time.process_time()
        # Get the current node
        current_node = frontier[0] # First node in the frontier list
        current_index = 0 # Set the index to the index of the first item in a list

        # Loop through the frontier list
        for index, item in enumerate(frontier):
            # If the total cost of the current looked upon node, is higher than a node in the frontier
            # then set the current looked upon node to the node with a lower total cost
            if item.f < current_node.f:
                current_node = item
                current_index = index

        # Pop current off frontier, add to closed list
        frontier.pop(current_index)
        closed_list.append(current_node)

        # If the current node is equal to end node, then the goal have been found
        if current_node == end_node:
            path = [] # This is the list that contains the path from end point to start. So this needs to be reversed
            current = current_node
            while current is not None:
                path.append(current.position) # Append the current node position to the path list
                current = current.parent # Now set the current node, to the parent of the previous appended node
            return path[::-1]  # Return the reversed path

        # Generate children
        children = [] # This list will contain the children for this iteration of the while loop
        for neighbors in [(0, -1), (0, 1), (-1, 0), (1, 0), (-1, -1), (-1, 1), (1, -1), (1, 1)]:  # Adjacent squares

            # Get the position of the neighbor node for this iteration of the for loop
            # Take the x position of the current node and add it to the current iteration of the new_position x position
            # Take the y position of the current node and add it to the current iteration of the new_position y position
            neighbor_position = (current_node.position[0] + neighbors[0], current_node.position[1] + neighbors[1])

            # Check whether the neighbor node is inside the maze
            if neighbor_position[0] > (len(maze) - 1) or neighbor_position[0] < 0 or neighbor_position[1] > (
                    len(maze[len(maze) - 1]) - 1) or neighbor_position[1] < 0:
                continue # Skip this iteration of the for loop

            # Check if the neighbor node is walkable
            if maze[neighbor_position[0]][neighbor_position[1]] != 0:
                continue # Skip this iteration of the for loop

            # Create new node with the position of the neighbor and the parent are the current looked upon note
            new_node = Node(current_node, neighbor_position)

            # Append the neighbor node to the children list
            children.append(new_node)

        # Loop through all children
        for child in children:

            # If the child is in the closed list, then just skip this iteration of the for loop
            for closed_child in closed_list:
                if child == closed_child:
                    continue

            # Calculate the f, g, and h values
            child.g = current_node.g + 1
            child.h = ((child.position[0] - end_node.position[0]) ** 2) + (
                        (child.position[1] - end_node.position[1]) ** 2)
            child.f = child.g + child.h

            # If the child is in the frontier list, and the child is further in the maze, the skip the iteration
            for open_node in frontier:
                if child == open_node and child.g > open_node.g:
                    continue

            # Add the child to the open list
            frontier.append(child)

        # If the function have run more than 10 seconds, then terminate the algorithm
        if timer > 10:
            return False

# function for random generate a maze
def create_maze(width, height):
    maze = []
    for i in range(0, height):
        line = []
        for j in range(0, width):
            if i != 0 and j != 0 and i != height-1 and j != width-1:
                line.append(random.randint(0, 1))
            else:
                line.append(0)
        maze.append(line)
    return maze


def main():
    # Create a maze
    #maze = [[0, 0, 0, 0, 1, 0, 0, 0, 0, 0],
    #        [0, 0, 0, 0, 1, 0, 0, 0, 0, 0],
    #        [0, 0, 0, 0, 1, 0, 0, 0, 0, 0],
    #        [0, 0, 0, 0, 1, 0, 0, 0, 0, 0],
    #        [0, 0, 0, 0, 1, 0, 0, 0, 0, 0],
    #        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    #        [0, 0, 0, 0, 1, 0, 0, 0, 0, 0],
    #        [0, 0, 0, 0, 1, 0, 0, 0, 0, 0],
    #        [0, 0, 0, 0, 1, 0, 0, 0, 0, 0],
    #        [0, 0, 0, 0, 1, 0, 0, 0, 0, 0]]

    maze = create_maze(10,10)

    # Print maze
    for line in maze:
        print(*line)

    # Create start and end point
    start = (0, 0)
    end = (9, 9)

    path = astar(maze, start, end)
    print(path)

if __name__ == '__main__':
    main()

