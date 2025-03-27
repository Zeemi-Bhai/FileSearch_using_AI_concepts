# File Search Algorithm Implementation

## Introduction
This project implements various search algorithms to efficiently locate a target file within a directory structure. It supports the following search techniques:

- **Breadth-First Search (BFS)**
- **Depth-First Search (DFS)**
- **Iterative Deepening DFS (IDDFS)**
- **Hill Climbing**
- **Genetic Algorithm**
- **Min-Max Search with Alpha-Beta Pruning**

Additionally, the project includes tree visualization using `networkx` and `matplotlib.pyplot`.

## Features
- Constructs a tree representation of a directory.
- Uses different search algorithms for file retrieval.
- Visualizes the directory structure as a graph.
- Handles permission errors and cyclic symbolic links gracefully.

## Installation
Clone this repository:
```sh
git clone https://github.com/your-username/file-search-algorithms.git
cd file-search-algorithms
```

Install the required dependencies:
```sh
pip install -r requirements.txt
```

## Usage
Run the script and specify the directory and file name:
```sh
python file_search.py
```
The program will:
1. Build a tree representation of the directory.
2. Print the directory structure.
3. Prompt the user for a file search query.
4. Execute various search algorithms to locate the file.
5. Visualize the directory tree.

## Dependencies
The script uses the following Python libraries:
- `os`
- `random`
- `matplotlib`
- `collections`
- `math`
- `networkx`

## Requirements
All dependencies are listed in `requirements.txt`. Install them using:
```sh
pip install -r requirements.txt
```

## Example Output
```
Enter the target file name: example.txt
BFS found: /path/to/example.txt
DFS found: /path/to/example.txt
...
```

## License
This project is licensed under the MIT License.

## Contributing
Contributions are welcome! Fork the repository and submit a pull request.

---

### `requirements.txt`
```
matplotlib
networkx
```

