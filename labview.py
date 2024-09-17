import matplotlib.pyplot as plt

def plotSpectrogram(spectrogram):
    print("1")
    # Plot the spectrogram
    plt.figure(figsize=(10, 6))
    print("2")
    # plt.imshow(spectrogram, aspect='auto', origin='lower', extent=[t.min(), t.max(), f.min(), f.max()], cmap='viridis')
    plt.imshow(spectrogram, aspect='auto', origin='lower', cmap='viridis')

    print("3")
    # Add labels and title
    plt.colorbar(label='Intensity')  # or the appropriate unit for your data
    plt.ylabel("Wavelength [nm]")
    plt.xlabel("Time [fs]")
    plt.title("Spectrogram")

    print("4")
    # Show the plot
    plt.show()

    print("5")
    return True
