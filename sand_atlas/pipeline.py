import numpy
from skimage.measure import label, regionprops
from skimage.morphology import closing

def label_binary_data(binary_data):

    cube = numpy.ones((5, 5, 5), dtype='uint8')
    closed = closing(binary_data, cube)

    labelled = label(closed)

    return labelled

    # tifffile.imwrite(outname + '.tif', labelled, compression ='zlib')