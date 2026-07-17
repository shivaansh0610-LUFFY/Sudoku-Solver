def is_valid(grid, row, col, num) -> bool:
    """
    Checks if placing `num` at grid[row][col] is valid according to Sudoku rules.
    - Checks row
    - Checks column
    - Checks 3x3 box containing (row, col)
    Returns True if valid, otherwise False.
    """
    # Check row
    for c in range(9):
        if grid[row][c] == num:
            return False
            
    # Check column
    for r in range(9):
        if grid[r][col] == num:
            return False
            
    # Check 3x3 box
    box_r_start = (row // 3) * 3
    box_c_start = (col // 3) * 3
    for r in range(box_r_start, box_r_start + 3):
        for c in range(box_c_start, box_c_start + 3):
            if grid[r][c] == num:
                return False
                
    return True

def find_empty_cell(grid) -> tuple or None:
    """
    Scans the grid in row-major order.
    Returns (row, col) of the first cell containing 0, or None if the grid is fully solved.
    """
    for r in range(9):
        for c in range(9):
            if grid[r][c] == 0:
                return (r, c)
    return None

def solve(grid) -> bool:
    """
    Solves the Sudoku grid in-place using a classic recursive backtracking algorithm.
    Returns True if a solution was found and mutated in place, False otherwise.
    """
    empty = find_empty_cell(grid)
    if not empty:
        return True
        
    row, col = empty
    for num in range(1, 10):
        if is_valid(grid, row, col, num):
            grid[row][col] = num
            if solve(grid):
                return True
            # Backtrack
            grid[row][col] = 0
            
    return False
