from modules import preprocessing as pre
import logging
import curses

'''
Variables and settings
'''
log_file = "/mnt/data/desy/dataset/dataset_01/preprocessing.log"
blacklist_path = "./additional/experimental_matrix_blacklist.csv"
dataset_directory ="/mnt/data/desy/dataset/dataset_01/"

# Logger Settings
logging.basicConfig(
        level=logging.INFO,
        style="{",
        format="{asctime} - {name} - {funcName} - {levelname}: {message}", datefmt='%d-%m-%Y %H:%M:%S',
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler()
            ]
)
logger = logging.getLogger(__name__)

GRID_SIZE = 512

pre.prepareDirectoriesForPreprocessing(
        dataset_directory=dataset_directory,
        experimental_blacklist_path=blacklist_path,
        grid_size= GRID_SIZE
        )

pre.preprocessDataset(
    dataset_directory=dataset_directory,
    grid_size= GRID_SIZE
    )

missing_es = pre.checkForMissingFile(
    parent_dir='/mnt/data/desy/dataset/dataset_01/preproc/simulated/',
    required_filename='Es.dat'
    )

missing_as = pre.checkForMissingFile(
    parent_dir='/mnt/data/desy/dataset/dataset_01/preproc/simulated/',
    required_filename='as_gn00.dat'
    )

print(f"Directories where 'Es.dat' files are missing:\n{missing_es}")
print(f"Directories where 'as_gn00.dat' files are missing:\n{missing_as}")
