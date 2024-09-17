import matplotlib.pyplot as plt

def plotSpectrogram(spectrogram):
    
    # Plot the spectrogram
    plt.figure(figsize=(10, 6))
    # plt.imshow(spectrogram, aspect='auto', origin='lower', extent=[t.min(), t.max(), f.min(), f.max()], cmap='viridis')
    plt.imshow(spectrogram, aspect='auto', origin='lower', cmap='viridis')

    # Add labels and title
    plt.colorbar(label='Intensity')  # or the appropriate unit for your data
    plt.ylabel("Wavelength [nm]")
    plt.xlabel("Time [fs]")
    plt.title("Spectrogram")

    # Show the plot
    plt.show()
