from modules import preprocessing
from modules import visualize

shg_path    = './test/experimental_data/original/spectrogram_1959.txt'
output_path = './test/experimental_data/preprocessed/'
N = 256

full_output_path = preprocessing.preprocess(shg_path, output_path, 256)

visualize.plotSpectrogram(full_output_path)
# visualize.plotSpectrogram('./additional/samples/as_gn00.dat')
