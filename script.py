from hsutil import label_utils, models, get_patches, plot, mat_io
import numpy as np

PATCH_SIZE=5

print("Loading Data")
data, labels, label_names = mat_io.open_mat_image("datasets/satellite.mat")
#data = np.pad(data, ((0,0),(0,0),(0,97)), mode="constant")

print("Getting Patches")
patches = get_patches.get(data,labels,patch_size=PATCH_SIZE,purity=0.75)
classes = label_utils.get_classes(labels)

print("Building Model")
model = models.SSRN_sat(PATCH_SIZE,6,9,10,0.0001)

print("Converting Labels")
train_labels, val_labels, label_dict = label_utils.to_one_hot(patches["train_labels"],patches["valid_labels"])

print("Training Model")
model.fit(patches["train_patches"],patches["valid_patches"],train_labels,val_labels,epochs=1,verbose=2,early_stopping=True,patience=10)

print("Predicting on Data")
predictions = model.predict(data,batch_size=100)

print("Converting Labels")
predictions = label_utils.from_one_hot(predictions, label_dict=label_dict) #label dict argument here

print("Plotting Results")
plot.patch_plot(patches)
plot.class_plot(predictions,class_labels=label_names)

'''
print("Loading Model")
model = models.load_model("Models/model",patch_size=5,depth=3,n_classes=9,batch_size=10,learning_rate=0.00001)

predictions = model.predict(data,batch_size=100)

predictions = label_utils.from_one_hot(predictions)

plot.patch_plot(patches)
plot.class_plot(predictions)

#data, labels = load_data(data_path='../datasets/UP/PaviaU.mat', data_name='PaviaU',label_path='../datasets/UP/PaviaU_gt.mat', label_name='PaviaU_gt')
'''

#data, labels = mat_io.open_mat_image("datasets/Indian Pines/Indian_pines_corrected.mat", image_name="indian_pines_corrected", labels_name="indian_pines_gt", src_labels="datasets/Indian Pines/Indian_pines_gt.mat")