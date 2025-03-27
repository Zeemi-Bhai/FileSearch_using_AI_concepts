
#Importing necessary libraries

import os
import random
import matplotlib.pyplot as plt
from collections import deque
import math
import networkx as nx

# Making class, as it will allow us to keep track of file in easy manner

class FileNode:
    def __init__(self, name, path, is_dir=False):
        self.name = name # file name
        self.path = path # path
        self.is_dir = is_dir # is dir or file?
        self.children = [] # sub directories or files

#Function , when called will build the whole TREE for files - root to leaf ( leaf mean who dont have further sub directories or files )

def build_file_tree(directory):
    """Builds a tree structure representing the directory."""
    root = FileNode(os.path.basename(directory), directory, is_dir=True)
    queue = deque([(directory, root)])
    
    while queue:
        current_dir, parent_node = queue.popleft()
        
        try:
            for entry in os.scandir(current_dir):
                node = FileNode(entry.name, entry.path, entry.is_dir())
                parent_node.children.append(node)
                if entry.is_dir():
                    queue.append((entry.path, node))
        except PermissionError:
            continue  # Skip directories that cannot be accessed
    
    return root

# This one function will enable us to make plot of our searches using matplotlib.

def visualize_search(visited_nodes, depths, search_type):
    """Visualizes the search process using matplotlib, including depth changes."""
    plt.figure(figsize=(10, 5))
    plt.scatter(range(len(visited_nodes)), depths, c='blue', label="Depth", marker='o')
    plt.plot(range(len(visited_nodes)), depths, linestyle='dashed', alpha=0.6)
    
    plt.xlabel("Step")
    plt.ylabel("Depth")
    plt.title(f"{search_type} Search Visualization")
    plt.legend()
    plt.grid(True, linestyle="--", alpha=0.7)
    
    plt.show()

# Finding Files by Breadth First Search
#First search for in folder files , then exloring the sub directories.

def bfs_search(root, target):
    """Performs BFS search in the file tree while tracking depth."""
    queue = deque([(root, 0)])  # (Node, Depth)
    visited_nodes, depths = [], []

    while queue:
        node, depth = queue.popleft()
        visited_nodes.append(node.name)
        depths.append(depth)

        if target in node.name:
            visualize_search(visited_nodes, depths, "BFS")
            return node.path

        for child in node.children:
            queue.append((child, depth + 1))

    visualize_search(visited_nodes, depths, "BFS")
    return None

# Finding Files by Depth First Search
#First search for in sib directoies, then exloring the files in same directory.

def dfs_search(root, target):
    """Performs DFS search while tracking depth correctly."""
    stack = [(root, 0)]  # (Node, Depth)
    visited_nodes, depths = [], []

    while stack:
        node, depth = stack.pop()
        visited_nodes.append(node.name)
        depths.append(depth)

        if target in node.name:
            visualize_search(visited_nodes, depths, "DFS")
            return node.path

        # Reverse children before pushing to stack to maintain correct DFS order
        for child in reversed(node.children):
            stack.append((child, depth + 1))

    visualize_search(visited_nodes, depths, "DFS")
    return None

# Exploring Based on Directories level defined. 
def iddfs_search(root, target, max_depth):
    """Performs Iterative Deepening DFS while visualizing search order correctly."""
    visited_nodes = []
    depths = []

    for depth_limit in range(max_depth + 1):
        level_nodes = []  # Track nodes at this depth limit
        level_depths = []
        result = dls(root, target, depth_limit, 0, level_nodes, level_depths)

        visited_nodes.extend(level_nodes)
        depths.extend(level_depths)

        if result:
            visualize_search(visited_nodes, depths, "IDDFS")
            # format_found_path(root, node.path)
            return result  # Stop once the target is found

    visualize_search(visited_nodes, depths, "IDDFS")
    return None  # If target was not found

# Its assisting the IDDFS to implement , as iddfs is the implementation of this , but level by level.

def dls(node, target, depth_limit, current_depth, visited_nodes, depths):
    """Performs Depth-Limited Search (DLS) properly for IDDFS."""
    if current_depth > depth_limit:
        return None  # Stop searching deeper

    visited_nodes.append(node.name)
    depths.append(current_depth)

    if target in node.name:
        return node.path  # Return path if found

    if current_depth < depth_limit:  # Only expand if within depth limit
        for child in node.children:
            result = dls(child, target, depth_limit, current_depth + 1, visited_nodes, depths)
            if result:
                return result  # Return early if found

    return None


# Hill Climbing Approach , but not suggested , as it always aiming for the maxima - so will stuck at the end of the first sub direcoties leaf node

def hill_climbing(root, target):
    """Performs Hill Climbing search while tracking depth."""
    current_node = root
    visited_nodes, depths = [], []
    depth = 0

    while current_node.children:
        visited_nodes.append(current_node.name)
        depths.append(depth)

        current_node.children.sort(key=lambda x: x.name)  # Sorting as heuristic
        next_node = current_node.children[0]
        
        if target in next_node.name:
            visited_nodes.append(next_node.name)
            depths.append(depth + 1)
            visualize_search(visited_nodes, depths, "Hill Climbing")
            return next_node.path
        
        current_node = next_node
        depth += 1

    visualize_search(visited_nodes, depths, "Hill Climbing")
    return None

# Implementaion of Genetics Algo - For this problem it can be complete or cannot be depending upon the situtaion.

def genetic_algorithm(root, target, generations=5):
    """Performs a Genetic Algorithm-style search with selection, crossover, and mutation."""
    if not root.children:
        return None  # Handle empty directories

    population = [random.choice(root.children) for _ in range(min(5, len(root.children)))]
    visited_nodes = []
    depths = []

    for gen in range(generations):
        new_population = []

        for node in population:
            visited_nodes.append(node.name)
            depths.append(gen)  # Using generation number as depth

            if target in node.name:
                visualize_search(visited_nodes, depths, "Genetic Algorithm")
                return node.path

            # Sort children based on similarity to the target (heuristic selection)
            sorted_children = sorted(node.children, key=lambda x: abs(len(x.name) - len(target)))

            # Pick the top 2 closest matches for next generation
            new_population.extend(sorted_children[:2] if len(sorted_children) >= 2 else sorted_children)

            # Crossover: Mix features of two parents
            if len(population) > 1:
                parent1, parent2 = random.sample(population, 2)
                crossover_node = parent1 if len(parent1.name) < len(parent2.name) else parent2
                if crossover_node.children:
                    new_population.append(random.choice(crossover_node.children))

        # Prevent getting stuck: If all nodes are the same, restart with new random choices
        if len(set(node.name for node in population)) == 1:
            population = [random.choice(root.children) for _ in range(min(5, len(root.children)))]
        else:
            population = new_population if new_population else population  # Update population

    visualize_search(visited_nodes, depths, "Genetic Algorithm")
    return None



def visualize_tree(root):
    G = nx.DiGraph()
    
    def add_edges(node):
        for child in node.children:
            G.add_edge(node.name, child.name)
            add_edges(child)
    
    add_edges(root)
    plt.figure(figsize=(12, 6))
    pos = nx.spring_layout(G)
    nx.draw(G, pos, with_labels=True, node_color='lightblue', edge_color='gray', node_size=3000, font_size=10)
    plt.show()

def minmax_search(node, target, depth, maximizing_player, alpha, beta):
    """
    Minimax algorithm with Alpha-Beta Pruning for file search.
    """
    if depth == 3 or not node.children:
        return node.path if target in node.name else None
    
    best_eval = None
    
    if maximizing_player:
        for child in node.children:
            eval = minmax_search(child, target, depth + 1, False, alpha, beta)
            if eval:
                best_eval = eval
            alpha = max(alpha, alpha if best_eval is None else float('-inf'))
            if beta <= alpha:
                break  
    else:
        for child in node.children:
            eval = minmax_search(child, target, depth + 1, True, alpha, beta)
            if eval:
                best_eval = eval
            beta = min(beta, beta if best_eval is None else float('inf'))
            if beta <= alpha:
                break  
    
    return best_eval

def search_file_with_minmax(directory, target):
    root = build_file_tree(directory)
    visualize_tree(root)
    return minmax_search(root, target, 0, True, -math.inf, math.inf)


# Print File Tree in good looking format with icons

def print_file_tree(node, indent=0):
    """Recursively prints the file tree structure in a beautiful format."""
    prefix = "   " * indent + ("📂 " if node.is_dir else "📄 ")
    print(prefix + node.name)
    
    for child in node.children:
        print_file_tree(child, indent + 1)


# Calling the functions for searching - and actuall program execution.

def main():
    current_directory = os.getcwd()
    root = build_file_tree(current_directory)
    print("File Tree Structure:")
    print_file_tree(root)
    
    search_query = input("Enter filename to search: ")
    
    print("BFS Search Result:", bfs_search(root, search_query) or "Not found")
    print("DFS Search Result:", dfs_search(root, search_query) or "Not found")
    print("IDDFS Search Result:", iddfs_search(root, search_query, max_depth=5) or "Not found")
    print("Hill Climbing Result:", hill_climbing(root, search_query) or "Not found")
    print("Genetic Algorithm Result:", genetic_algorithm(root, search_query) or "Not found")
    print("Min-Max Search Result:", search_file_with_minmax(current_directory, search_query) or "Not found")
    

if __name__ == "__main__":
    main()


