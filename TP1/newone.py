import pygame
import random

# Define the size of the grid
GRID_SIZE = 10

# Define the number of colors in the game
NUM_COLORS = 4

# Define the symbols used to represent each color
COLORS = [(255, 255, 255), (255, 0, 0), (0, 255, 0), (0, 0, 255), (255, 255, 0)]

# Define the size of each cell
CELL_SIZE = 50

# Define the size of the window
WINDOW_SIZE = (GRID_SIZE * CELL_SIZE, GRID_SIZE * CELL_SIZE)

def create_grid(size):
    """
    Creates a new grid with the given size and fills it with random colors.
    """
    grid = [[random.randint(1, NUM_COLORS) for _ in range(size)] for _ in range(size)]
    return grid

def fill_zone(grid, row, col, color):
    """
    Fills the zone around the given cell with the given color.
    """
    if row < 0 or row >= len(grid) or col < 0 or col >= len(grid[0]):
        # Cell is out of bounds
        return
    if grid[row][col] != color:
        # Cell is not the same color as the starting cell
        return
    # Fill the current cell with the new color
    grid[row][col] = 0
    # Recursively fill the surrounding cells
    fill_zone(grid, row-1, col, color)
    fill_zone(grid, row+1, col, color)
    fill_zone(grid, row, col-1, color)
    fill_zone(grid, row, col+1, color)

def draw_grid(screen, grid):
    """
    Draws the grid on the screen.
    """
    for row in range(GRID_SIZE):
        for col in range(GRID_SIZE):
            color = COLORS[grid[row][col]]
            rect = pygame.Rect(col*CELL_SIZE, row*CELL_SIZE, CELL_SIZE, CELL_SIZE)
            pygame.draw.rect(screen, color, rect)

def play_game():
    """
    Runs the Fill Zone game.
    """
    # Initialize Pygame
    pygame.init()
    # Create the screen
    screen = pygame.display.set_mode(WINDOW_SIZE)
    # Set the caption
    pygame.display.set_caption("Fill Zone")
    # Create a new grid
    grid = create_grid(GRID_SIZE)
    # Run the game loop
    running = True
    while running:
        # Handle events
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.MOUSEBUTTONUP:
                # Get the position of the mouse click
                pos = pygame.mouse.get_pos()
                # Convert the position to grid coordinates
                col = pos[0] // CELL_SIZE
                row = pos[1] // CELL_SIZE
                # Get the starting color
                color = grid[row][col]
                # Fill the zone with the starting color
                fill_zone(grid, row, col, color)
        # Clear the screen
        screen.fill((0, 0, 0))
        # Draw the grid
        draw_grid(screen, grid)
        # Update the screen
        pygame.display.flip()
    # Quit Pygame
    pygame.quit()

if __name__ == '__main__':
    play_game()