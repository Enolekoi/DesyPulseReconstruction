from modules import preprocessing as pre
import logging
import curses

'''
Variables and settings
'''
# Logger Settings
logging.basicConfig(
        level=logging.INFO,
        style="{",
        format="{asctime} - {name} - {funcName} - {levelname}: {message}", datefmt='%d-%m-%Y %H:%M:%S',
        handlers=[
            logging.StreamHandler()
            ]
)
logger = logging.getLogger(__name__)

def wait_for_key(key='q'):
    def main(stdscr):
        # initialize curses
        curses.cbreak()     # disable line buffering
        curses.noecho()     # disable input echoing
        stdscr.keypad(True) # Enable keyboard for special keys
        
        # Instructions for the user
        stdscr.addstr(0, 0, f"Press '{key}' to continue...")
        stdscr.refresh

        while True:
            # Wait for a key press
            pressed_key = stdscr.getch() # get the pressed key
            if chr(pressed_key) == key:
                stdscr.addstr(1, 0, f"Resuming...")
                print(f"'{key}' pressed! Resuming script.")
                stdscr.refresh
                break
    curses.wrapper(main)

GRID_SIZE = 512

pre.prepareDirectoriesForPreprocessing(
        dataset_directory="/mnt/data/desy/dataset/dataset_01/",
        experimental_blacklist_path="./additional/experimental_matrix_blacklist.csv",
        grid_size= GRID_SIZE
        )

# wait_for_key('y')

pre.pre(
    dataset_directory="/mnt/data/desy/dataset/dataset_01/",
    grid_size= GRID_SIZE
    )
