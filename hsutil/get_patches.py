import numpy as np
import random

def get(data,labels,split=0.25,patch_size=5,purity=1,null_class=0):
    '''Function to split data into training and validation patches.
    
    Parameters
    ----------
    split : float
            The fraction of total data that should be used for validation.
    patch_size : integer
                 The dimension of the square patches to be extracted, in pixels. Should be positive, odd integer.
    purity : float
             The fraction of the extracted patch that should be of the same class type.
    null_class : str, integer, float, other
                 The class label in the dataset that corresponds to un-labelled data.
    
    returns
    ------
    patches : dictionary
              A dictionary containing arrays of training patches, training patch labels, training patch coordinates, validation patches, validation patch labels, 
              validation patch coordinates and a 2d-array representing a map of all patches accross the image.
    '''
    
    def _valid_patch(index, labels, patches, patch_size, purity, use):
        '''Function to check that a new patch is valid. This is done by ensuring the selected patch is sufficiently pure (i.e., the total fraction of the patch that
        is the same class must be over the threshold purity) and ensuring no overlap between training and validation
        patches'''
        
        valid = True #variable to track invalid conditions
            
        if labels[index[0], index[1]] == 0: #excludes un-labelled data from potential patches
            valid = False
        
        if valid: #lengthier checks are carried out if the basic criteria are met
            
            classes = [] #stores a list of classes within the patch
            patch_map_classes = [] #stores a list of all other patch types present within the patch
            radius = int((patch_size-1)/2)
            for i in range(-radius,radius+1): #loops to record all classes and patch types present within the new proposed patch
                for j in range(-radius,radius+1):
                    classes.append(labels[index[0]+i, index[1]+j])
                    patch_map_classes.append(patches["patch_map"][index[0]+i, index[1]+j])

            base_class = classes[0] #stores the class of the patch index
            
            if classes.count(base_class)/(patch_size**2) < purity:
                valid = False
                        
            #other procedures can be added here
            
            if (use == "train") and (2 in patch_map_classes): #checks that no pixels in the proposed patch are already occupied by a patch of the opposite type (training or validation)
                valid = False
            elif (use == "valid") and (1 in patch_map_classes):
                valid = False
            
        return valid #returns a boolean signalling if the proposed patch is valid
    
    
    def _save_patch(index, data, labels, patches, patch_size, use):
        '''Function to store new patch data in appropriate arrays.'''
        
        def mark_patch(index, patches, patch_size, use):
            '''Function that marks the new patch on the patch map: Parameters are as specified for parent function.'''
            
            if use == "train": #determines the marker used to mark the new patch on the patch map
                M = 1
            elif use == "valid":
                M = 2
            
            radius = int((patch_size-1)/2)
            for i in range(-radius,radius+1): #for loops to mark the new patch on the patch map
                for j in range(-radius,radius+1):
                    if patches["patch_map"][index[0]+i, index[1]+j] != M: #statement to avoid marking a position with the same patch type more than once
                        patches["patch_map"][index[0]+i, index[1]+j] += M
                        
            return patches
        
        i = index[0] #defines the patch index as i and j, for ease of reading
        j = index[1]
        
        radius = int((patch_size-1)/2)
        if use == "train": #appends the new patch data to the relevant arrays
            patches["train_patches"].append(data[i-radius:i+radius+1,j-radius:j+radius+1,:])
            patches["train_labels"].append(labels[i,j])
            patches["train_indices"].append(index)
            patches = mark_patch(index, patches, patch_size, use)
        else:
            patches["valid_patches"].append(data[i-radius:i+radius+1,j-radius:j+radius+1,:])
            patches["valid_labels"].append(labels[i,j])
            patches["valid_indices"].append(index)
            patches = mark_patch(index, patches, patch_size, use)
        
        return patches
    
    
    def _initiate_patches(labels, patches, patch_size, purity, null_class):
        '''Function that attempts to initiate a training patch in every class region of the image. If this can't be done successfully, a warning is displayed to alert
        the user that not all class types will be present in the training set.'''
        
        def _get_class_indices(labels):
    
            classes = [] #list to store class types
            indices = [] #list to store indices associated with each class type
            
            radius = int((patch_size-1)/2)
            for i in range(radius,labels.shape[0]-radius): #'''fix to only scan within valid patch areas (i.e. only indices where patches can actually fit)'''
                for j in range(radius,labels.shape[1]-radius):
                    if labels[i][j] in classes:
                        indices[classes.index(labels[i][j])].append([i,j])
                    else:
                        classes.append(labels[i][j])
                        indices.append([[i,j]])
            
            return classes, indices
        
        
        classes, indices = _get_class_indices(labels)
        
        for i in range(len(indices)): #shuffles the order of indices from each class
            random.shuffle(indices[i])
        
        saved = [] #list to store all of the saved patch indices 
        found_all_classes = True #variable to determine if at least one patch of every class type has been saved
        
        for i, index_list in enumerate(indices):
            found_patch = True #variable to determine if a patch has been found in the selected class type
            
            if classes[i] != null_class:
                found_patch = False
                
                for index in index_list: #attempts to find a patch in the selected class area
                    if _valid_patch(index, labels, patches, patch_size, purity, "train"):
                        patches = _save_patch(index, data, labels, patches, patch_size, "train")
                        found_patch = True
                        saved.append(index)
                        break
            
            if found_patch == False:
                found_all_classes = False
            
        if found_all_classes == False:
            print("Warning: not all classes are present in the training set. Try using smaller patches or decreasing the patch purity.")
        
        return saved, patches
    
    
    patch_size = int(round((patch_size-1)/2)*2 + 1) #rounds the requested patch_size to the nearest odd number
    if patch_size < 1: #ensures the patch size is at least 1
        patch_size = 1
    
    #initiates a dictionary to store all relevant patch information
    patches = {
        "train_patches" : [], #stores the training patches
        "train_labels" : [], #stores the training labels
        "train_indices" : [], #stores the indices of the training patches
        "valid_patches" : [],
        "valid_labels" : [],
        "valid_indices" : [],
        "patch_map" : np.zeros(labels.shape) #stores an area map of all of the training and validation patches
    }
    
    n_valid = round(split*100) #calculates the percentage of validation and training patches to generate
    n_train = 100 - n_valid
    
    radius = int((patch_size-1)/2)
    indices = []
    for i in range(radius,labels.shape[0]-radius):
        for j in range(radius,labels.shape[1]-radius):
            indices.append([i,j])
    random.shuffle(indices)
    train_patch_rejects = []
    valid_patch_rejects = []
    
    saved, patches = _initiate_patches(labels, patches, patch_size, purity, null_class)
    
    for index in saved: #removes indices from the indices list that were populated with patches during initialisation
        indices.remove(index)
    
    count = len(saved) #initiates a variable to store the number of patches that have been placed
    for index in indices: #for loop that generates the patches in batches, with each batch containing the requested fraction of validation patches
        
        if count < n_train:
            if _valid_patch(index, labels, patches, patch_size, purity, "train"):
                patches = _save_patch(index, data, labels, patches, patch_size, "train")
                count += 1
            else:
                train_patch_rejects.append(index)
        
        elif count >= n_train:
            if _valid_patch(index, labels, patches, patch_size, purity, "valid"):
                patches = _save_patch(index, data, labels, patches, patch_size, "valid")
                count += 1
            else:
                valid_patch_rejects.append(index)
        
        if count >= 100:
            count = 0
    
    count = 0 #code block that attempts validation patch placement in all locations invalid for training patches
    for index in train_patch_rejects:
        if _valid_patch(index, labels, patches, patch_size, purity, "valid"):
            patches = _save_patch(index, data, labels, patches, patch_size, "valid")
            count += 1
            
    lim = int(count*((1-split)/split)) #calculates the limit on additional training patch placement (in the next code block) to ensure the correct split ratio
    
    count = 0 #code block that attempts training patch placement in all locations invalid for validation patches
    for index in valid_patch_rejects:
        if _valid_patch(index, labels, patches, patch_size, purity, "train") and (count < lim):
            patches = _save_patch(index, data, labels, patches, patch_size, "train")
            count += 1
        
    for array in patches: #converts all lists in the patches dictionary to numpy arrays
        patches[array] = np.array(patches[array])
    
    return patches