import numpy as np
import pandas as pd
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

FIRST_ROW = 14
FIRST_COLUMN = 0

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
    0: arcade.color.WHITE, # WHITE = (255, 255, 255)
    1: arcade.color.RED, # RED = (255, 0, 0)
    2: arcade.color.BLUE, # BLUE = (0, 0, 255)
    3: arcade.color.GREEN, # GREEN = (0, 255, 0)
    4: arcade.color.PINK, # PINK = (255, 192, 203)
    5: arcade.color.YELLOW # YELLOW = (255, 255, 0)
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

        first_color = self.grid_sprites[FIRST_ROW][FIRST_COLUMN]
        # print([self.grid_sprites[FIRST_ROW][i].color for i in range(len(self.grid_sprites[FIRST_ROW]))])
        # print("First color: ",first_color.color)
        pos = FIRST_ROW * COLUMN_COUNT + FIRST_COLUMN
        print("First color: ",self.grid_sprite_list[pos].color)
        it_fill_zone(self, FIRST_ROW, FIRST_COLUMN, first_color, selected_color)

def it_fill_zone(self, row,column, first_color, selected_color):
    print(row, column)
    print("Before:\t Actual:",self.grid_sprites[row][column].color, " First: ",first_color.color) #hay un problema aca que toma el color actual siempre igual a first color
    
    if row >= ROW_COUNT-1 or column >= COLUMN_COUNT:
        return

    if row < 0 or column < 0:
        return
    
    if self.grid_sprites[row][column] != first_color:
        return
    
    it_fill_zone(self, row-1, column, first_color, selected_color)
    it_fill_zone(self, row+1, column, first_color, selected_color)
    it_fill_zone(self, row, column-1, first_color, selected_color)
    it_fill_zone(self, row, column+1, first_color, selected_color) 

    print("Selected: ", selected_color)
    if self.grid_sprites[row][column] == first_color:
        self.grid_sprites[row][column].color = selected_color
        x = column * (WIDTH + MARGIN) + (WIDTH / 2 + MARGIN)
        # y = row  * (HEIGHT + MARGIN) + (HEIGHT / 2 + MARGIN)
        # sprite = arcade.SpriteSolidColor(WIDTH, HEIGHT, arcade.color.RED)
        # sprite.center_x = x
        # sprite.center_y = y
        # self.grid_sprites[row][column].color = sprite.color
    print("After: ",self.grid_sprites[row][column].color)
    



def main():
    MyGame(SCREEN_WIDTH, SCREEN_HEIGHT, SCREEN_TITLE)
    arcade.run()

if __name__ == "__main__":
    main()