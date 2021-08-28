import numpy as np

def to_one_hot(labels,val_labels=None,null_class=0):
    '''Function that converts class labels to one-hot vectors.
    
    Parameters
    ----------
    labels : array_like
             1-dimensional array of class labels to be converted.
    val_labels : array_like
                 (Optional) 1-dimensional array of validation class labels to be converted.
    
    returns
    --------
    one_hots : array_like
               An array of one-hot vectors associated with the labels provided.
    label_dict : dictionary
                 A dictionary that matches class types (the keys) to their associated one-hot vectors (the keys).'''
    
    label_dict = {} #creates dictionary to convert class labels to one hot vectors
    classes = get_classes(labels,as_array=True)
    classes.remove(null_class)
    for i, _class in enumerate(classes):
        one_hot = np.zeros(len(classes))
        one_hot[i] = 1
        label_dict[_class] = one_hot
    
    one_hots = []
    for label in labels: #converts all numberical class labels to the corresponding one-hot vector
        one_hots.append(label_dict[label])
    one_hots = np.array(one_hots)
    
    if val_labels:
        val_one_hots = []
        for label in val_labels: #converts all numberical class labels to the corresponding one-hot vector
            val_one_hots.append(label_dict[label])
        val_one_hots = np.array(val_one_hots)
        
        return one_hots, val_one_hots, label_dict
    
    else:
        
        return one_hots, label_dict
    
    '''Add try except statement to tell user if labels present in validation set that arent in training set'''


def from_one_hot(labels,label_dict=None):
    pass

def get_classes(labels, as_array=False):
    '''Function that returns a list of all the class types present in the labels set.
    
    Parameters
    ----------
    labels : array_like
             1-dimensional array of class labels, or
             a 2-dimensional array of class labels associated with a hyperspectral image.
    
    returns
    --------
    classes : dictionary or array_like
              A dictionary with class types as keys and class label counts as items, or
              a 1-dimensional, sorted array of the class types present in the image labels.'''
    
    from collections import Counter
    
    if labels.ndim == 1:
        total_counter = Counter(labels)
    
    elif labels.ndim == 2:
        total_counter = Counter(labels[0])
        for i in range(1,len(labels)): #for loop to record all labels present in the training set, converting them from one-hot vectors to integers
            total_counter += Counter(labels[i])
    
    else:
        raise ValueError("Labels array must be 1-dimensional or 2-dimensional")
    
    if as_array:
        classes = [] #array to store a list of class types present in the training set
        for class_type in total_counter: #for loop to record all class types present in the training set
            classes.append(class_type)
        classes = sorted(classes) #sorts the class types for ease of reading when they are later displayed to the user

        return classes
    
    else:
        return total_counter