"""
Functions for loading datasets from .mat files

"""
import scipy.io as sio
import os


def to_mat(features, labels, dst, val_features=None, val_labels=None):
    """
    Convert hyperspectral patches and labels to a mat file on disk

    Parameters
    ----------
    features : array-like
        4D or iterable of 3D arrays containing image data
    labels : array-like
        1D array of labels
    dst : str
        path to new location
    val_features: array-like
        Optional 4D or iterable of 3D arrays containing validation image data
    val_labels : array-like
        Optional 1D array of validation labels
    """
    if features.dtype != 'float32':
        raise TypeError('Image data type must be uint32')
    if val_features is not None:
        if val_features.dtype != 'float32':
            raise TypeError('Image data type must be uint32')
    # add suffix
    if not dst.endswith('.mat'):
        dst = dst+".mat"
    
    if (val_features is not None) and (val_labels is not None):
        mdic = {"features" : features,
                "labels" : labels,
                "val_features" : val_features,
                "val_labels" : val_labels
            }
    else:
        mdic = {"features" : features,
                "labels" : labels,
            }
        
    sio.savemat(dst, mdic, format="5", do_compression=True)
    #note: setting compression to True has apparently caused issues for some users in the past. If saved files wont re-load, try setting it to false.
    
    return None


def open_mat(src):
    """
    Read mat file of hyperspectral patches
    
    Parameters
    ----------
    src : str
        filepath to read
        
    Returns
    ---------
    features : array_like
               4D array of features (first dimension is the feature index, remaining dimensions are the feature dimensions).
    labels : array_like
              1D array of feature labels.
    val_features : array_like
              Optional array of validation features, in the same format as the features array.
    val_labels : array_like
                 Optional array of validation labels, in the same format as the labels array.
    """
    if not os.path.exists(src):
        raise ValueError('mat File Not Found')
        
    mat_data = sio.loadmat(src) #loads data from the matlab file
    
    features = mat_data["features"] #extracts the data from the mat_data dictionary
    labels = mat_data["labels"][0]
    
    if "val_features" in mat_data:
        
        val_features = mat_data["val_features"]
        val_labels = mat_data["val_labels"][0]
        
        return features, labels, val_features, val_labels
    
    else:
        
        return features, labels
    

def to_mat_image(image, labels, dst, image_name="image", labels_name="labels", dst_labels=None):
    """
    Convert hyperspectral image and labels to a mat file on disk

    Parameters
    ----------
    image : array-like
            3D array containing image data (first two dimensions spatial, remaining dimension is band number).
    labels : array-like
             2D array of class labels associated with the image.
    dst : str
          Path to location at which the data should be stored.
    image_name : str
                 The key to be associated with the image data in the .mat file.
    labels_name : str
                  The key to be associated with the image labels in the .mat file.
    dst_labels : str
                 A second filepath at which which labels should be stored. If this is provided, labels will not be stored at the image filepath.
    """
    
    if image.dtype != 'float32':
        raise TypeError('Image data type must be uint32')
    # add suffix
    if not dst.endswith('.mat'):
        dst = dst+".mat"
    if dst_labels:
        if not dst_labels.endswith('.mat'):
            dst_labels = dst_labels+".mat"
    
    if not dst_labels: #saves image and labels in same file, if second path not provided
        mdic = {image_name : image,
                labels_name : labels,
            }
        sio.savemat(dst, mdic, format="5", do_compression=True)
        
    else: #saves image and labels in separate files, if a labels path is provided
        mdic = {image_name : image}
        sio.savemat(dst, mdic, format="5", do_compression=True)
        
        labels_mdic = {labels_name : labels}
        sio.savemat(dst_labels, labels_mdic, format="5", do_compression=True)
    
    return None


def open_mat_image(src, image_name="image",labels_name="labels",src_labels=None):
    """
    Read mat file(s) containing a hyperspectral image and associated labels
    
    Parameters
    ----------
    src : str
          Filepath to read image and label data from.
    image_name : str
                 The key associated with the image data in the .mat file.
    labels_name : str
                  The key associated with the image labels in the .mat file.
    src_labels : str
                 A second filepath from which labels should be loaded from. If this is provided, no attempt to load labels from the image filepath will be made.
        
    Returns
    ---------
    image : array_like
               3D array of spectral data (first two dimensions are spatial, remaining dimension is band number).
    labels : array_like
              2D array of class labels associated with the image.
    """
    
    if not os.path.exists(src):
        raise ValueError('mat File Not Found')
        
    mat_data = sio.loadmat(src) #loads data from the matlab file
    
    image = mat_data[image_name] #extracts image data
    if not src_labels: #extracts label data, according to the filepaths provided
        labels = mat_data[labels_name]
    else:
        mat_label_data = sio.loadmat(src_labels)
        labels = mat_label_data[labels_name]
   
    return image, labels
    