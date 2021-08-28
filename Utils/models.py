class _Model():
    
    '''Primary model class that contains functions common to all deep learning models.'''
    
    def __init__(self,patch_size,depth,n_classes,batch_size,learning_rate):
        
        self.patch_size = patch_size
        self.depth = depth
        self.n_classes = n_classes
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        
    def fit(self, train_data, valid_data, train_labels=None, valid_labels=None, epochs=20, verbose=0):
        
        '''Function to train a model.
        
        Parameters
        ----------
        train_data : array_like or tensorflow dataset
                     4-dimensional array of patches (first dimension is the patch index, remaining dimensions are the patch itself) that the model is trained on,
                     or a tensorflow dataset, with each element containing a patch and associated integer class label.
        valid_data : array_like or tensorflow dataset
                     4-dimensional array of patches (first dimension is the patch index, remaining dimensions are the patch itself) that the model is validated on,
                     or a tensorflow dataset, with each element containing a patch and associated integer class label.
        train_labels : array_like
                     2-dimensional array of one-hot vector labels (first dimension is the label index, second dimension is the one-hot vector) associated with the training patches.
                     Should only be specified when train_data and valid_data are array-like.
        valid_labels : array_like
                     2-dimensional array of one-hot vector labels (first dimension is the label index, second dimension is the one-hot vector) associated with the validation patches.
                     Should only be specified when train_data and valid_data are array-like.
        epochs : integer
                The number of epochs to run during model training.
        verbose : integer
                Integer in range 0-2 that determines the feedback printed to the console, during the training process.
        '''
        
        from keras.callbacks import Callback
        from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
        import numpy as np
        import gc
        import time
        
        class _Metrics(Callback):
            '''Class to create a metrics object that calculates and stores metrics during model training (required by Keras to access model metrics during training)'''
        
            def __init__(self, parent_model, train_data, val_data):
                super().__init__()
                self.training_data = train_data
                self.validation_data = val_data
                self.batch_size = parent_model.batch_size #records the batch size of the model being trained
            
            def on_train_begin(self, logs={}):
                '''Initiates lists to store metrics during training'''
                
                self.train_accuracy_scores = []
                self.train_f1_scores = []
                self.train_recall_scores = []
                self.train_precision_scores = []
                
                self.val_accuracy_scores = []
                self.val_f1_scores = []
                self.val_recall_scores = []
                self.val_precision_scores = []
                
                self.times = []
                self.checkpoint = time.monotonic()
                
            def on_epoch_end(self, epoch, logs={}):
                '''Calculates metrics, on each epoch, and stores them during training'''
                
                self.times.append(time.monotonic()-self.checkpoint)
                
                gc.collect()
                
                train_predict = (np.asarray(self.model.predict(self.training_data[0],batch_size=self.batch_size))).round() #calculates training data predictions
                train_targ = self.training_data[1]
                
                val_predict = (np.asarray(self.model.predict(self.validation_data[0],batch_size=self.batch_size))).round() #calculates validation data predictions
                val_targ = self.validation_data[1]
                
                train_accuracy = accuracy_score(train_targ, train_predict) #calculates training accuracy score
                train_f1 = f1_score(train_targ, train_predict, average=None) #calculates training f1 score (for each class)
                train_recall = recall_score(train_targ, train_predict, average="macro") #calculates training recall score
                train_precision = precision_score(train_targ, train_predict, average="macro") #calculates training precision score
                
                val_accuracy = accuracy_score(val_targ, val_predict)
                val_f1 = f1_score(val_targ, val_predict, average=None)
                val_recall = recall_score(val_targ, val_predict, average="macro")
                val_precision = precision_score(val_targ, val_predict, average="macro")
                
                self.train_accuracy_scores.append(train_accuracy) #stores the metrics in the appropriate list
                self.train_f1_scores.append(train_f1)
                self.train_recall_scores.append(train_recall)
                self.train_precision_scores.append(train_precision)
                
                self.val_accuracy_scores.append(val_accuracy)
                self.val_f1_scores.append(val_f1)
                self.val_recall_scores.append(val_recall)
                self.val_precision_scores.append(val_precision)
                
                self.checkpoint = time.monotonic()
        
        
        n_train_patches = int(np.floor(len(train_data)/self.batch_size)*self.batch_size) #rounds the number of training and validation patches to the nearest multiple of the batch size
        n_valid_patches = int(np.floor(len(valid_data)/self.batch_size)*self.batch_size)
        
        t_steps = round(n_train_patches/self.batch_size) #calculates the number of training and validation batches present in the dataset
        v_steps = round(n_valid_patches/self.batch_size)
        
        train_data = train_data[:n_train_patches] #trims the data arrays to the appropriate lengths, such that the data can be trained on in an integer number of batches
        train_labels = train_labels[:n_train_patches] #Note: a small amount of data is lost during this process
        valid_data = valid_data[:n_valid_patches]
        valid_labels = valid_labels[:n_valid_patches]
        
        self.metrics = _Metrics(self,[train_data,train_labels],[valid_data,valid_labels]) #initiates a metrics object for storing model metrics
        
        self.model.fit(
                x=train_data,
                y=train_labels,
                batch_size=self.batch_size,
                steps_per_epoch=t_steps,
                validation_data=(valid_data,valid_labels),
                validation_steps=v_steps,
                epochs=epochs,
                verbose=verbose,
                callbacks=[self.metrics]
                ) #trains the model
        
    
    def predict(self, data, batch_size, label_dict, verbose=0):
        
        '''Function to predict on a dataset using a model.
        
        Parameters
        ----------
        data : array_like
                     3-dimensional array of spectral data (first two dimensions are the spatial dimensions, remaining dimension is the spectral dimension) to be predicted on.
        batch_size : integer
                     The number of samples in each batch the model predicts on. Note: large batch sizes (> 500) can require significant GPU memory.
        label_dict : dictionary
                     Dictionary to associate class labels with one-hot vectors, later used to convert one-hot vectors to class labels. The keys are the dataset class labels and the items are the associated one-hot vectors.
        verbose : integer
                    Integer that determines the amount of feedback written to the console. 0: no feedback. 1: appropriate warnings displayed. 2: full feedback on prediction process.
        
        Returns
        -------
        predictions : array_like
                      Either: 2-dimensional array of prediction labels (dimensions are equal to the spatial dimensions of the input data array) (if an image was previously passed in as data),
                      Or: 1-dimensional array of prediction labels (length equal to the number of patches) (if an array of patches was previously passed in as data).
        '''
        
        import numpy as np
        
        shape = data.shape #stores the dimensions of the input data
        
        if data.ndim == 3: #interprets the data as raw spectral imagery, if the number of dimensions is 3
            
            radius = int((self.patch_size-1)/2) #calculates the "radius" of a patch (how many pixels the patch extends from its central pixel)
            data = np.pad(data, ((radius,radius),(radius,radius),(0,0)), mode="edge") #pads the input data using the "edge" method, to prevent dimensionality reduction during sliding window operation
            predictions = np.zeros(shape[0]*shape[1]) #initiates a 1-dimensional array to store the model predictions
            
            patch_arr = [] #list to store patches extracted from the input data
            for i in range(radius,shape[0]+radius): #for loops to iterate over the input data and extract patches, in a sliding window operation
                for j in range(radius,shape[1]+radius):
                    patch_arr.append(data[i-radius:i+radius+1,j-radius:j+radius+1,:])
                    
        elif data.ndim == 4: #interprets the data as a list of patches
        
            patch_arr = data
            predictions = np.zeros(shape[0]) #initiates a 1-dimensional array to store the model predictions
        
        inv_label_dict = {np.argmax(v): k for k, v in label_dict.items()} #creates a dictionary to convert one-hot vectors to class labels (essentially label_dict inverted, with integers representing one-hot vectors)
        
        #block of code to construct new models for prediction (this must be done if the requested prediction batch size is different than the training batch size)
        weights = self.model.get_weights() #loads the current model weights
        b_pred_model = self._build_model(batch_size) #generates a batch prediction model (a model that predicts in batches)
        b_pred_model.set_weights(weights) #sets the weights of the prediction model to that of the trained model
        b_pred_model.compile() #compiles the batch prediction model
        s_pred_model = self._build_model(1) #generates a single prediction model (a model that predicts one item at a time)
        s_pred_model.set_weights(weights)
        s_pred_model.compile()
                
        remainder = len(patch_arr)%batch_size #calculates the remainder of the number of patches divided by the batch size - these extra patches will need to be predicted individually
        num_batches = len(patch_arr)//batch_size #calculates the number of batches patch_arr should be divided into
        
        if (num_batches == 0) and (verbose > 0): #warns the user if the batch size is too large
            print("Warning: requested batch size is larger than the number of datapoints. All predictions will therefore be made individually, which could take significant time.")
        
        b_patch_arr = np.array(patch_arr[:len(patch_arr)-remainder]) #slices patch_arr to extract patches to be included in batch prediction and stores them in b_patch_arr
        b_patch_arr = b_patch_arr.reshape((num_batches,batch_size,self.patch_size,self.patch_size,shape[2])) #reshapes b_patch_arr into batches of patches
        s_patch_arr = np.array(patch_arr[len(patch_arr)-remainder:]) #slices patch_arr to extract patches to be predicted individually and stores them in s_patch_arr
        
        for n, batch in enumerate(b_patch_arr): #for loop to run batch predictions
            
            b_predictions = b_pred_model.predict(batch, batch_size=batch_size) #predicts a batch
            for i, prediction in enumerate(b_predictions): #loop to convert one-hot vector predictions to class labels, using the inv_label_dict, and store them in the predictions array
                predictions[n*batch_size + i] = inv_label_dict[np.argmax(prediction)]
            
            if (n%10 == 0) and (verbose > 1): #statement to allow the user to track progress
                print("Predicting batch "+str(n)+" out of "+str(num_batches)+".")
              
        for i, patch in enumerate(s_patch_arr): #for loop to run single predictions
            
            prediction = s_pred_model.predict(np.array([patch])) #predicts a patch
            predictions[(n+1)*batch_size + i] = inv_label_dict[np.argmax(prediction)] #converts one-hot vector prediction to class label, using the inv_label_dict, and stores it in the predictions array
            
        if data.ndim == 3:
            
            predictions = np.reshape(predictions,(shape[0],shape[1])) #reshapes the predictions array to 2 dimensions equal to the spatial dimensions of the input data
            
        return predictions
            

class SSRN(_Model):
    
    '''Class to generate a spectral spatial residual network object.
    
    Parameters
    ----------
    patch_size : integer
                 Size of the data patches the model should accept. Should be an odd number greater than or equal to 3.
    depth : integer
            The number of spectral and/or spatial blocks to include in the network.
    n_classes: integer
               The number of class types the model should expect to recieve in the data.
    batch_size : integer
                 The number of samples per batch that the model is trained on.
    
    Returns
    -------
    model : keras model object
    '''
    
    def __init__(self,patch_size,depth,n_classes,batch_size,learning_rate):
        super().__init__(patch_size,depth,n_classes,batch_size,learning_rate)
        
        self.model = self._build_model()
        
    
    def _build_model(self):
        '''Function to construct and compile the neural network.'''
        
        import tensorflow as tf
        import keras
        from keras.layers import BatchNormalization, Conv3D, Activation, Input, AveragePooling3D, Add, Dense, Flatten, Lambda
        from keras.layers.core import Reshape
        
        
        def edge_pad(X, padding=(1,1,1)):
            '''Function to perform edge padding on a keras tensor.'''
            
            for layer in range(padding[0]): #for loop to add padding in the x-dimension
                X = tf.concat((tf.expand_dims(X[:, 0, ...], 1), X), axis=1)
                X = tf.concat((X, tf.expand_dims(X[:, -1, ...], 1)), axis=1)
            
            for layer in range(padding[1]): #for loop to add padding in the y-dimension
                X = tf.concat((tf.expand_dims(X[:, :, 0, ...], 2), X), axis=2)
                X = tf.concat((X, tf.expand_dims(X[:, :, -1, ...], 2)), axis=2)
            
            for layer in range(padding[2]): #for loop to add padding in the z-dimension
                X = tf.concat((tf.expand_dims(X[:, :, :, 0, ...], 3), X), axis=3)
                X = tf.concat((X, tf.expand_dims(X[:, :, :, -1, ...], 3)), axis=3)
                
            return X
        
    
        def spectral_res_block(X):
            '''Function to add a spectral residual block to the neural network sequence.'''
            
            X_shortcut = X #stores the initial network sequence, so a residual connection can be formed later on
            
            X = Lambda(lambda X: edge_pad(X, (0,0,3)))(X) #edge padding in the z-direction
            X = Conv3D(filters=24, kernel_size=(1,1,7), padding="valid", kernel_initializer = 'glorot_uniform')(X) #first convolutional batchnorm layer of the spectral residual block
            X = BatchNormalization(axis=3)(X)
            X = Activation('relu')(X)
            
            X = Lambda(lambda X: edge_pad(X, (0,0,3)))(X) #edge padding in the z-direction
            X = Conv3D(filters=24, kernel_size=(1,1,7), padding="valid", kernel_initializer = 'glorot_uniform')(X) #second convolutional batchnorm layer of the spectral residual block
            
            X = Add()([X, X_shortcut]) #addition layer to add in the residual connection to the network sequence
            X = BatchNormalization(axis=3)(X)
            X = Activation('relu')(X)
            
            #Architectural notes: The addition of the residual connection could be carried out after batchnormalisation.
            
            return X
        
        
        def spatial_res_block(X):
            '''Function to add a spatial residual block to the neural network sequence'''
            
            X_shortcut = X #stores the initial network sequence, so a residual connection can be formed later on
            
            X = Lambda(lambda X: edge_pad(X, (1,1,0)))(X) #edge padding in the x and y directions
            X = Conv3D(filters=24, kernel_size=(3,3,24), padding="valid", kernel_initializer = 'glorot_uniform')(X)  #first convolutional batchnorm layer of the spatial residual block
            X = Reshape((X.shape[1],X.shape[2],X.shape[4],1))(X)
            X = BatchNormalization(axis=3)(X)
            X = Activation('relu')(X)
            
            X = Lambda(lambda X: edge_pad(X, (1,1,0)))(X) #edge padding in the x and y directions
            X = Conv3D(filters=24, kernel_size=(3,3,24), padding="valid", kernel_initializer = 'glorot_uniform')(X) #second convolutional batchnorm layer of the spatial residual block
            X = Reshape((X.shape[1],X.shape[2],X.shape[4],1))(X)
            
            X = Add()([X, X_shortcut]) #addition layer to add in the residual connection to the network sequence
            X = BatchNormalization(axis=3)(X)
            X = Activation('relu')(X)
            
            #Architectural notes: The reshape layers could be positioned after batchnormalisation. The addition of the residual connection could be carried out after batchnormalisation.
            
            return X
        
        
        X_input = Input(batch_size=self.batch_size,shape=(self.patch_size,self.patch_size,200,1)) #initialises an input to the neural network of dimensions patch_size x patch_size x 200, with a single data channel
        X = Conv3D(filters=24, kernel_size=(1,1,7), strides=(1,1,2), padding="valid", kernel_initializer = 'glorot_uniform')(X_input) #initial convolution layer preceding the spectral residual blocks
        X = BatchNormalization(axis=3)(X)
        X = Activation('relu')(X)
        
        for block in range(self.depth): #for loop to add the desired number of spectral residual blocks
            
            X = spectral_res_block(X)
            
        X = Conv3D(filters=128, kernel_size=(1,1,97), padding="valid", kernel_initializer = 'glorot_uniform')(X) #first post-spectral residual block convolution layer preceding the spectral residual blocks
        X = Reshape((X.shape[1],X.shape[2],X.shape[4],1))(X)
        X = BatchNormalization(axis=3)(X)
        X = Activation('relu')(X)
        
        if self.patch_size == 3:
            X = Lambda(lambda X: edge_pad(X, (1,1,0)))(X) #edge padding in the x and y directions to ensure tensor is not smaller than filter
            
        X = Conv3D(filters=24, kernel_size=(3,3,128), padding="valid", kernel_initializer = 'glorot_uniform')(X) #second post-spectral residual block convolution layer preceding the spectral residual blocks
        X = Reshape((X.shape[1],X.shape[2],X.shape[4],1))(X)
        X = BatchNormalization(axis=3)(X)
        X = Activation('relu')(X)
        
        #Architectural notes: The reshape layers could be positioned after batchnormalisation.
            
        for block in range(self.depth): #for loop to add the desired number of spectral residual blocks
            
            X = spatial_res_block(X)
        
        if self.patch_size == 3: #statements to ensure correct pool sizes are used for average pooling
            X = AveragePooling3D(pool_size=(3,3,1), padding='valid')(X)
        else:
            X = AveragePooling3D(pool_size=(self.patch_size-2,self.patch_size-2,1), padding='valid')(X)
        #average pooling layer preceding the residual blocks to reduce dimensions
        X = Flatten()(X)
        X = Dense(self.n_classes, kernel_initializer = 'glorot_uniform')(X) #final dense layer used for classification
        X = Activation('softmax')(X)
        
        model = keras.Model(inputs=X_input, outputs=X, name="SSRN") #initiates a keras model object, with the network sequence as constructed above
        
        optimizer = tf.keras.optimizers.Adam(learning_rate=self.learning_rate)
        model.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics="accuracy") #compiles the keras model for training
        
        return model
    
    
class Spectral(_Model):
    
    '''Class to generate a spectral residual network object.
    
    Parameters
    ----------
    patch_size : integer
                 Size of the data patches the model should accept. Should be an odd number greater than or equal to 3.
    depth : integer
            The number of spectral and/or spatial blocks to include in the network.
    n_classes: integer
               The number of class types the model should expect to recieve in the data.
    batch_size : integer
                 The number of samples per batch that the model is trained on.
                 
    Returns
    -------
    model : keras model object
    '''
    
    def __init__(self,patch_size,depth,n_classes,batch_size,learning_rate):
        super().__init__(patch_size,depth,n_classes,batch_size,learning_rate)
        
        self.model = self._build_model()
        
    
    def _build_model(self):
        
        import tensorflow as tf
        import keras
        from keras.layers import BatchNormalization, Conv3D, Activation, Input, AveragePooling3D, Add, Dense, Flatten, Lambda
        from keras.layers.core import Reshape
        
        
        def edge_pad(X, padding=(1,1,1)):
            '''Function to perform edge padding on a keras tensor.'''
            
            for layer in range(padding[0]): #for loop to add padding in the x-dimension
                X = tf.concat((tf.expand_dims(X[:, 0, ...], 1), X), axis=1)
                X = tf.concat((X, tf.expand_dims(X[:, -1, ...], 1)), axis=1)
            
            for layer in range(padding[1]): #for loop to add padding in the y-dimension
                X = tf.concat((tf.expand_dims(X[:, :, 0, ...], 2), X), axis=2)
                X = tf.concat((X, tf.expand_dims(X[:, :, -1, ...], 2)), axis=2)
            
            for layer in range(padding[2]): #for loop to add padding in the z-dimension
                X = tf.concat((tf.expand_dims(X[:, :, :, 0, ...], 3), X), axis=3)
                X = tf.concat((X, tf.expand_dims(X[:, :, :, -1, ...], 3)), axis=3)
                
            return X
        
        
        def spectral_res_block(X):
            '''Function to add a spectral residual block to the neural network sequence'''
            
            X_shortcut = X #stores the initial network sequence, so a residual connection can be formed later on
            
            X = Lambda(lambda X: edge_pad(X, (0,0,3)))(X) #edge padding in the z-direction
            X = Conv3D(filters=24, kernel_size=(1,1,7), padding="valid", kernel_initializer = 'glorot_uniform')(X) #first convolutional batchnorm layer of the spectral residual block
            X = BatchNormalization(axis=3)(X)
            X = Activation('relu')(X)
            
            X = Lambda(lambda X: edge_pad(X, (0,0,3)))(X) #edge padding in the z-direction
            X = Conv3D(filters=24, kernel_size=(1,1,7), padding="valid", kernel_initializer = 'glorot_uniform')(X) #second convolutional batchnorm layer of the spectral residual block
            
            X = Add()([X, X_shortcut]) #addition layer to add in the residual connection to the network sequence
            X = BatchNormalization(axis=3)(X)
            X = Activation('relu')(X)
            
            #Architectural notes: The addition of the residual connection could be carried out after batchnormalisation.
            
            return X
        
        
        X_input = Input(batch_size=self.batch_size,shape=(self.patch_size,self.patch_size,200,1)) #initialises an input to the neural network of dimensions patch_size x patch_size x 200, with a single data channel
        X = AveragePooling3D(pool_size=(self.patch_size,self.patch_size,1),padding="valid")(X_input) #average pooling to reduce the patch to a size of 1x1x200
        X = Conv3D(filters=24, kernel_size=(1,1,7), strides=(1,1,2), padding="valid", kernel_initializer = 'glorot_uniform')(X) #initial convolution layer preceding the spectral residual blocks
        X = BatchNormalization(axis=3)(X)
        X = Activation('relu')(X)
        
        for block in range(self.depth): #for loop to add the desired number of spectral residual blocks
            
            X = spectral_res_block(X)
            
        X = Conv3D(filters=128, kernel_size=(1,1,97), padding="valid", kernel_initializer = 'glorot_uniform')(X) #post-spectral residual block convolution layer
        X = Reshape((X.shape[1],X.shape[2],X.shape[4],1))(X)
        X = BatchNormalization(axis=3)(X)
        X = Activation('relu')(X)
            
        X = Flatten()(X)
        X = Dense(self.n_classes, kernel_initializer = 'glorot_uniform')(X) #final dense layer used for classification
        X = Activation('softmax')(X)
        
        model = keras.Model(inputs=X_input, outputs=X, name="Spectral") #initiates a keras model object, with the network sequence as constructed above
        
        optimizer = tf.keras.optimizers.Adam(learning_rate=self.learning_rate)
        model.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics="accuracy") #compiles the keras model for training
        
        return model
    