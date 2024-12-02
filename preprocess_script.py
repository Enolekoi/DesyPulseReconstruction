from modules import preprocessing as pre
import logging
import keyboard

'''
Variables and settings
'''
# Logger Settings
logging.basicConfig(
        level=logging.INFO,
        style="{",
        format="{asctime} - {name} - {funcName} - {levelname}: {message}",
        datefmt='%d-%m-%Y %H:%M:%S',
        handlers=[
            logging.StreamHandler()
            ]
)
logger = logging.getLogger(__name__)

def wait_for_key(key='q'):
    print(f"Press '{key}' to continue...")
    while True:
        if keyboard.is_pressed(key):
        print(f"'{key}' pressed! Resuming script.")
        break

GRID_SIZE = 256

pre.prepareDirectoriesForPreprocessing(
        dataset_directory="/mnt/data/desy/dataset/dataset_01/",
        experimental_blacklist_path="./additional/experimental_matrix_blacklist.csv",
        grid_size= GRID_SIZE
        )

wait_for_key('y')

pre.pre(
    dataset_directory="/mnt/data/desy/dataset/dataset_01/",
    grid= GRID_SIZE
    )
