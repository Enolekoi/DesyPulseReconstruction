from modules import preprocessing as pre

pre.prepare(
        dataset_directory="/mnt/data/desy/dataset/dataset_01/",
        experimental_blacklist_path="./additional/experimental_matrix_blacklist.csv"
        )
