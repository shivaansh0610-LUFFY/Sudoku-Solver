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

def is_valid_puzzle(grid) -> bool:
    """
    Performs a sanity check on the pre-filled cells of the Sudoku puzzle.
    Verifies that no already-filled cell conflicts with row, column, or 3x3 box rules.
    Returns True if the puzzle has no contradictions, False otherwise.
    
    Avoids the self-comparison bug by temporarily clearing each cell while checking.
    """
    for r in range(9):
        for c in range(9):
            num = grid[r][c]
            if num != 0:
                # Temporarily clear the cell to avoid comparing it to itself
                grid[r][c] = 0
                valid = is_valid(grid, r, c, num)
                grid[r][c] = num # Restore the cell
                if not valid:
                    return False
    return True

def pretty_print_grid(grid) -> None:
    """
    Prints a 9x9 Sudoku grid in a readable format with 3x3 block separators.
    """
    for r in range(9):
        if r % 3 == 0 and r != 0:
            print("-" * 21)
        row_str = ""
        for c in range(9):
            if c % 3 == 0 and c != 0:
                row_str += "| "
            val = grid[r][c]
            row_str += f"{val if val != 0 else '.'} "
        print(row_str.strip())
