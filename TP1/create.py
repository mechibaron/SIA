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

# WHITE = (255, 255, 255)
# RED = (255, 0, 0)
# GREEN = (0, 255, 0)
# BLUE = (0, 0, 255)
# PINK = (255, 200, 200)
# YELLOW = (255, 255, 0)

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

        # One dimensional list of all sprites in the two-dimensional sprite list
        self.grid_sprite_list = arcade.SpriteList()
        self.grid_sprites = []
            
        # Create a list of solid-color sprites to represent each grid location
        for row in range(ROW_COUNT):
            self.grid_sprites.append([])
            if row == ROW_COUNT - 1 :
                for column in range(6):
                    x = column * (WIDTH + MARGIN) + (WIDTH / 2 + MARGIN)
                    y = row  * (HEIGHT + MARGIN) + (HEIGHT / 2 + MARGIN)
                    sprite = arcade.SpriteSolidColor(WIDTH, HEIGHT, colors[column])
                    sprite.center_x = x
                    sprite.center_y = y
                    self.grid_sprite_list.append(sprite)
                    self.grid_sprites[row].append(sprite)
            else:
                for column in range(COLUMN_COUNT):
                    x = column * (WIDTH + MARGIN) + (WIDTH / 2 + MARGIN)
                    y = row  * (HEIGHT + MARGIN) + (HEIGHT / 2 + MARGIN)
                    sprite = arcade.SpriteSolidColor(WIDTH, HEIGHT, colors[random.randrange(0,5)])
                    sprite.center_x = x
                    sprite.center_y = y
                    self.grid_sprite_list.append(sprite)
                    self.grid_sprites[row].append(sprite)

    def on_draw(self):
        """
        Render the screen.
        """
        # We should always start by clearing the window pixels
        self.clear()

        # Batch draw the grid sprites
        self.grid_sprite_list.draw()


    def on_mouse_press(self, x, y, button, modifiers):
        """
        Called when the user presses a mouse button.
        """

        # Convert the clicked mouse position into grid coordinates
        column = int(x // (WIDTH + MARGIN))
        row = int(y // (HEIGHT + MARGIN))

        if(row == ROW_COUNT - 1 and column <= 6):
            selected_color = colors[column]
            print("Selected color: ", selected_color)

            fill_zone(self,x,y, selected_color)
            # DESDE EL BLOQUE [ROW_COUNT][COLUMN_COUNT] -> AGARRAR TODOS LOS QUE SON DEL MISMO COLOR A SU ALREDEDOR (CODIGO MECHA) Y PINTARLOS DE SELECTED_COLOR 
            
def fill_zone(self, x, y, selected_color):
        """
        Fills the zone around the given cell with the given color.
        """
        
        column = int(x // (WIDTH + MARGIN))
        row = int(y // (HEIGHT + MARGIN))

        print(f"Click coordinates: ({x}, {y}). Grid coordinates: ({row}, {column})")

        # Make sure we are on-grid. It is possible to click in the upper right
        # corner in the margin and go to a grid location that doesn't exist
        if row < 0 or row > ROW_COUNT - 1 or column < 0 or column > COLUMN_COUNT:
            # Simply return from this method since nothing needs updating
            return

        first_color = self.grid_sprites[ROW_COUNT - 2][0]
        print(first_color)
        it_fill_zone(self, 14,0, first_color, selected_color)
def it_fill_zone(self, row,column, first_color, selected_color):
    if self.grid_sprites[row][column] == first_color:
        print('son iguales')
        self.grid_sprites[row][column].color = selected_color
    # it_fill_zone(self, row-1, col, first_color, selected_color)
    # it_fill_zone(self, row+1, col, first_color, selected_color)
    # it_fill_zone(self, row, col-1, first_color, selected_color)
    # it_fill_zone(self, row, col+1, first_color, selected_color)


def main():
    MyGame(SCREEN_WIDTH, SCREEN_HEIGHT, SCREEN_TITLE)
    arcade.run()

if __name__ == "__main__":
    main()