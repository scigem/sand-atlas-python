import numpy
import tifffile
import nrrd

def load_data(filename):
    """
    Load data from a file based on its extension.
    Parameters:
    filename (str): The path to the file to be loaded.
    Returns:
    data: The data loaded from the file. The type of data returned depends on the file extension:
        - For '.tif' or '.tiff' files, returns a memmap or an array from tifffile.
        - For '.raw' files, returns a memmap from numpy.
        - For '.npz' files, returns an array from numpy.
        - For '.nrrd' files, returns the data and header from nrrd.
    """
    extension = filename.split('.')[-1]
    if ( extension.lower() == 'tif' ) or ( extension.lower() == 'tiff'):
        try:
            data = tifffile.memmap(filename)
        except:
            data = tifffile.imread(filename)
    elif ( extension.lower() == 'raw' ):
        data = numpy.memmap(filename)
    elif ( extension.lower() == 'npz' ):
        data = numpy.load(filename, allow_pickle=True)['arr_0']
    elif ( extension.lower() == 'nrrd' ):
        data, header = nrrd.read(filename)
    else:
        raise ValueError('Unsupported file extension')
    
    return data

def save_data(data, filename):
    """
    Save data to a file with the specified filename and extension.

    Parameters:
    data (array-like): The data to be saved.
    filename (str): The name of the file to save the data to. The extension of the filename determines the format in which the data will be saved.

    Supported file extensions:
    - 'tif' or 'tiff': Save data as a TIFF file using tifffile.imsave.
    - 'raw': Save data as a raw binary file using data.tofile.
    - 'npz': Save data as a NumPy compressed file using numpy.savez.
    - 'nrrd': Save data as an NRRD file using nrrd.write.

    Raises:
    ValueError: If the file extension is not supported.
    """
    extension = filename.split('.')[-1]
    if ( extension.lower() == 'tif' ) or ( extension.lower() == 'tiff'):
        tifffile.imsave(filename, data)
    elif ( extension.lower() == 'raw' ):
        data.tofile(filename)
    elif ( extension.lower() == 'npz' ):
        numpy.savez(filename, data)
    elif ( extension.lower() == 'nrrd' ):
        nrrd.write(filename, data)
    else:
        raise ValueError('Unsupported file extension')

def convert(input_filename, output_filename):
    """
    Converts data from an input file and saves it to an output file.

    Args:
        input_filename (str): The path to the input file containing the data to be converted.
        output_filename (str): The path to the output file where the converted data will be saved.

    Returns:
        None
    """
    data = load_data(input_filename)
    save_data(data, output_filename)

