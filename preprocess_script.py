from modules import preprocessing as pre
import logging

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

GRID_SIZE = 256

pre.prepare(
        dataset_directory="/mnt/data/desy/dataset/dataset_01/",
        experimental_blacklist_path="./additional/experimental_matrix_blacklist.csv",
        grid_size= GRID_SIZE
        )
