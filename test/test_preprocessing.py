from modules import preprocessing
from modules import visualize as vis

shg_path    = './test/experimental_data/original/spectrogram_1959.txt'
output_path = './test/experimental_data/preprocessed/as.dat'
plot_path = './test/experimental_data/preprocessed/preproc.png'
N = 512

full_output_path = preprocessing.preprocess(shg_path, output_path, N)
vis.comparePreproccesSHGMatrix(
        raw_filepath=shg_path,
        preproc_filepath=output_path,
        save_path=plot_path)
