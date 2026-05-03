import heapq
from typing import Dict, List, Optional, Tuple, Set

def manhattan_distance(a: Tuple[int, int], b: Tuple[int, int]) -> int:
    """Dummy heuristic – replace with actual distance measure if needed."""
    return 0

def a_star_product(start_node: int, goal_node: int,
                   parent_of: Dict[int, int],
                   children_of: Dict[int, List[int]],
                   heuristic: callable) -> Tuple[Optional[List[int]], int]:
    """
    A* search on the product category tree (bidirectional movement allowed).
    """
    # Priority queue: (f_score, counter, node)
    counter = 0
    frontier = [(heuristic(start_node, goal_node), counter, start_node)]
    heapq.heapify(frontier)
    
    # g_score: best cost so far from start to node
    g_score = {start_node: 0}
    came_from = {start_node: None}
    visited = set()
    expanded_count = 0
    
    while frontier:
        f, _, current = heapq.heappop(frontier)
        
        if current in visited:
            continue
        visited.add(current)
        expanded_count += 1
        
        if current == goal_node:
            # Reconstruct path
            path = []
            node = current
            while node is not None:
                path.append(node)
                node = came_from[node]
            path.reverse()
            return path, expanded_count
        
        # Expand neighbors: parent (if exists) + children
        neighbors = []
        if current in parent_of:
            neighbors.append(parent_of[current])
        neighbors.extend(children_of.get(current, []))
        
        for neighbor in neighbors:
            if neighbor in visited:
                continue
            tentative_g = g_score[current] + 1  # uniform step cost
            if neighbor not in g_score or tentative_g < g_score[neighbor]:
                came_from[neighbor] = current
                g_score[neighbor] = tentative_g
                f_score = tentative_g + heuristic(neighbor, goal_node)
                counter += 1
                heapq.heappush(frontier, (f_score, counter, neighbor))
    
    return None, expanded_count