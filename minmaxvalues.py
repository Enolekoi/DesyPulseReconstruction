import os

def process_header(lines):
    """
    Extract the header information from the lines of the file.
    If the first line has fewer than 5 elements, the header is 2 lines.
    Otherwise, it's just the first line.
    """
    header_line_1 = lines[0].split()
    
    if len(header_line_1) < 5:
        header_line_2 = lines[1].split()
        header = header_line_1 + header_line_2
    else:
        header = header_line_1

    if len(header) < 5:
        print("ERROR")
    
    # Convert elements to float for numeric calculations
    header = [float(x) for x in header]
    
    return header

def process_files(root_directory):
    max_timestep = float('-inf')
    max_wavelength = float('-inf')
    min_wavelength = float('inf')
    
    for subdirectory in os.listdir(root_directory):
        subdirectory_path = os.path.join(root_directory, subdirectory)
        if os.path.isdir(subdirectory_path):
            file_path = os.path.join(subdirectory_path, 'as_gn00.dat')
            if os.path.exists(file_path):
                with open(file_path, 'r') as f:
                    # Read the first two lines
                    lines = [f.readline(), f.readline()]
                    
                    # Extract the header
                    header = process_header(lines)
                    
                    # Extract the required values
                    timestep = header[2]  # Third element
                    number_wavelength = header[1]  # Second element
                    wavelength_step = header[3]  # Fourth element
                    center_wavelenght = header[4]  # Fifth element

                    # Update the maximum timestep
                    if timestep > max_timestep:
                        max_timestep = timestep
                    
                    # Calculate wavelength related values
                    wavelength_plus = (number_wavelength // 2) * wavelength_step + center_wavelenght
                    wavelength_minus = (number_wavelength // 2) * wavelength_step - center_wavelenght
                    
                    # Update maximum and minimum wavelengths
                    if wavelength_plus > max_wavelength:
                        max_wavelength = wavelength_plus
                    if wavelength_minus < min_wavelength:
                        min_wavelength = wavelength_minus
                            
    # Print the results
    print(f"Highest Time Step: {max_timestep}")
    print(f"Max Wavelength: {max_wavelength}")
    print(f"Min Wavelength: {min_wavelength}")

# Example usage:
root_dir = "/mnt/data/desy/frog_simulated/grid_256_v3/"
process_files(root_dir)
