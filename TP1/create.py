import numpy as np
import arcade
import random

"""
Array Backed Grid Shown By Sprites

Show how to use a two-dimensional list/array to back the display of a
grid on-screen.

This version makes a grid of sprites instead of numbers. Instead of
interating all the cells when the grid changes we simply just
swap the color of the selected sprite. This means this version
can handle very large grids and still have the same performance.

If Python and Arcade are installed, this example can be run from the command line with:
python -m arcade.examples.array_backed_grid_sprites_2
"""


# Set how many rows and columns we will have
ROW_COUNT = 16
COLUMN_COUNT = 15
COLORS_COUNT = 6

# This sets the WIDTH and HEIGHT of each grid location
WIDTH = 30 
HEIGHT = 30

# This sets the margin between each cell
# and on the edges of the screen.
MARGIN = 5

# Do the math to figure out our screen dimensions
SCREEN_WIDTH = (WIDTH + MARGIN) * COLUMN_COUNT + MARGIN
SCREEN_HEIGHT = (HEIGHT + MARGIN) * ROW_COUNT + MARGIN
SCREEN_TITLE = "Array Backed Grid Buffered Example"

colors = {
    0: arcade.color.WHITE,
    1: arcade.color.RED,
    2: arcade.color.BLUE,
    3: arcade.color.GREEN,
    4 : arcade.color.PINK,
    5 :arcade.color.YELLOW
}

class MyGame(arcade.Window):
    """
    Main application class.
    """

    def __init__(self, width, height, title):
        """
        Set up the application.
        """
        super().__init__(width, height, title)

        # Set the background color of the window
        self.background_color = arcade.color.BLACK

        print('HOLA')
        self.grid = []
        for row in range(ROW_COUNT):
            print('Ciclo row:', row)
            self.grid.append([])
            for column in range(COLUMN_COUNT):
                if row == ROW_COUNT - 1 :
                    if column < COLORS_COUNT:
                        color=colors[column]  
                        self.grid[row].append(color)
                        x = (MARGIN + WIDTH) * column + MARGIN + WIDTH // 2
                        y = (MARGIN + HEIGHT) * row + MARGIN + HEIGHT // 2
                        # Draw the box
                        arcade.draw_rectangle_filled(x, y, WIDTH, HEIGHT, color)  # Append a cell
                else:
                    color=colors[random.randrange(0,5)]  
                    self.grid[row].append(color)
                    x = (MARGIN + WIDTH) * column + MARGIN + WIDTH // 2
                    y = (MARGIN + HEIGHT) * row + MARGIN + HEIGHT // 2

                    # Draw the box
                    arcade.draw_rectangle_filled(x, y, WIDTH, HEIGHT, color)# Append a cell

    def on_mouse_press(self, x, y, button, modifiers):
        """
        Called when the user presses a mouse button.
        """

        # Convert the clicked mouse position into grid coordinates
        column = int(x // (WIDTH + MARGIN))
        row = int(y // (HEIGHT + MARGIN))

        if row == ROW_COUNT-1 and column < COLORS_COUNT:
            selected_color = colors[column]
            first_color = self.grid[ROW_COUNT - 2][0]
            print("First color: ", first_color)
            print("Selected color: ", selected_color)
            fill_zone(self,x,y, selected_color,first_color)

        
       
def fill_zone(self, row, column, selected_color, first_color):
        """
        Fills the zone around the given cell with the given color.
        """

        print('First color:',first_color)
        it_fill_zone(self, ROW_COUNT-2,0, first_color, selected_color)
        
def it_fill_zone(self, row,column, first_color, selected_color):
    if row < 0 or row > ROW_COUNT - 2 or column < 0 or column > COLUMN_COUNT:
        return
    
    if self.grid[row][column] == first_color:
        self.grid[row][column]=selected_color
        x = (MARGIN + WIDTH) * column + MARGIN + WIDTH // 2
        y = (MARGIN + HEIGHT) * row + MARGIN + HEIGHT // 2
        arcade.draw_rectangle_filled(x, y, WIDTH, HEIGHT, selected_color) # Append a cell
        it_fill_zone(self, row-1, column, first_color, selected_color)
        it_fill_zone(self, row+1, column, first_color, selected_color)
        it_fill_zone(self, row, column-1, first_color, selected_color)
        it_fill_zone(self, row, column+1, first_color, selected_color)


def main():
    MyGame(SCREEN_WIDTH, SCREEN_HEIGHT, SCREEN_TITLE)
    arcade.run()

if __name__ == "__main__":
    main()