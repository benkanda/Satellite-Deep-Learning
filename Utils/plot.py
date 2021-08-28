import numpy as np
import matplotlib.pyplot as plt


def patch_plot(patches,figure=1):
    
    x = np.linspace(0,patches['patch_map'].shape[0],patches['patch_map'].shape[0])
    y = np.linspace(0,patches['patch_map'].shape[1],patches['patch_map'].shape[1])
    
    plt.figure(figure)
    plt.pcolormesh(y,x,patches["patch_map"])
    plt.title("Patch Map for Patch Size "+str(patches['train_patches'][0].shape[0]))
    ax = plt.gca()
    ax.set_aspect("equal")
    plt.show()
    
    return None


def class_plot(labels, class_labels=None, title=None, figure=1):
    
    import load_data
    
    classes = load_data.get_classes(labels)
    maximum = max(classes)
    minimum = min(classes)
    n_classes = len(classes)
    if class_labels:
        for i, label in enumerate(classes):
            if label != 0:
                classes[i] = class_labels[label]
    
    cmap = plt.get_cmap("tab20",n_classes-1)
    cmap.set_under("black")
    im = plt.imshow(labels, cmap=cmap, origin="lower", vmin=1,vmax=maximum)
    cbar = plt.colorbar(im, orientation="vertical", ticks=np.linspace(minimum+0.5,maximum+0.5,n_classes+1))
    cbar.set_ticklabels(classes)
    plt.show()
    
def metric_plot(metrics):
    
    fig_m, ax_m = plt.subplots(ncols=2,nrows=2, figsize=(12,10))
    
    n_epochs = len(metrics.train_accuracy_scores)
    epochs = np.linspace(1,n_epochs,n_epochs)
    
    ax_m[0][0].plot(epochs,metrics.train_accuracy_scores,label="training")
    ax_m[0][0].plot(epochs,metrics.val_accuracy_scores,label="validation")
    ax_m[0][0].set_ylim((0,1))
    ax_m[0][0].set_title("Accuracy")
    ax_m[0][0].set_xlabel("Epochs")
    ax_m[0][0].set_ylabel("Accuracy")
    ax_m[0][0].legend(bbox_to_anchor=(1.4,1)) 
    
    f1_scores = np.array(metrics.val_f1_scores)
    f1_scores = np.transpose(f1_scores)
    for i, class_scores in enumerate(f1_scores):
        ax_m[0][1].plot(epochs,class_scores)
        ax_m[0][1].set_ylim((0,1))
    ax_m[0][1].set_title("F1 Scores")
    ax_m[0][1].set_xlabel("Epochs")
    ax_m[0][1].set_ylabel("F1 Score")
    ax_m[0][1].legend(bbox_to_anchor=(1.05,1))
    
    ax_m[1][0].plot(epochs,metrics.train_recall_scores,label="training")
    ax_m[1][0].plot(epochs,metrics.val_recall_scores,label="validation")
    ax_m[1][0].set_ylim((0,1))
    ax_m[1][0].set_title("Recall")
    ax_m[1][0].set_xlabel("Epochs")
    ax_m[1][0].set_ylabel("Recall")
    ax_m[1][0].legend(bbox_to_anchor=(1.05,1))
    
    ax_m[1][1].plot(epochs,metrics.train_precision_scores,label="training")
    ax_m[1][1].plot(epochs,metrics.val_precision_scores,label="validation")
    ax_m[1][1].set_ylim((0,1))
    ax_m[1][1].set_title("Precision")
    ax_m[1][1].set_xlabel("Epochs")
    ax_m[1][1].set_ylabel("Precision")
    ax_m[1][1].legend(bbox_to_anchor=(1.05,1))
    
    
    fig_m.tight_layout()
    plt.show()
    
def evaluate_model(labels,patches,predictions,metrics,label_dict,class_labels=None):
    
    import load_data
    
    max_class = np.amax(labels)
    
    fig_c, ax_c = plt.subplots(ncols=2,nrows=2, figsize=(12,10))
    
    classes = load_data.get_classes(labels)
    maximum = max(classes)
    minimum = min(classes)
    n_classes = len(classes)
    if class_labels:
        for i, label in enumerate(classes):
            if label != 0:
                classes[i] = class_labels[label]
    cmap = plt.get_cmap("tab20",n_classes-1)
    cmap.set_under("black")
    labels_im = ax_c[0][0].imshow(labels, cmap=cmap, origin="lower", vmin=1,vmax=max_class, interpolation="none") #np.ma.masked_values(labels,0)
    labels_cbar = fig_c.colorbar(labels_im, orientation="vertical", ticks=np.linspace(minimum+0.5,maximum+0.5,n_classes+1), ax=ax_c[0][0])
    labels_cbar.set_ticklabels(classes)
    ax_c[0][0].set_title("Ground Truth Labels")
    
    train_mask = np.copy(labels)
    train_mask[patches["patch_map"] == 2] = 0
    train_mask[patches["patch_map"] == 0] = 0
    classes = load_data.get_classes(train_mask)
    maximum = max(classes)
    minimum = min(classes)
    n_classes = len(classes)
    if class_labels:
        for i, label in enumerate(classes):
            if label != 0:
                classes[i] = class_labels[label]
    cmap = plt.get_cmap("tab20",n_classes-1)
    cmap.set_under("black")
    train_im = ax_c[0][1].imshow(train_mask, cmap=cmap, origin="lower", vmin=1,vmax=max_class, interpolation="none")
    train_cbar = fig_c.colorbar(train_im, orientation="vertical", ticks=np.linspace(minimum+0.5,maximum+0.5,n_classes+1), ax=ax_c[0][1])
    train_cbar.set_ticklabels(classes)
    ax_c[0][1].set_title("Training Patches")
    
    valid_mask = np.copy(labels)
    valid_mask[patches["patch_map"] == 1] = 0
    valid_mask[patches["patch_map"] == 0] = 0
    classes = load_data.get_classes(valid_mask)
    maximum = max(classes)
    minimum = min(classes)
    n_classes = len(classes)
    if class_labels:
        for i, label in enumerate(classes):
            if label != 0:
                classes[i] = class_labels[label]
    cmap = plt.get_cmap("tab20",n_classes-2)
    cmap.set_under("black")
    valid_im = ax_c[1][0].imshow(valid_mask, cmap=cmap, origin="lower",vmin=1,vmax=max_class, interpolation="none")
    valid_cbar = fig_c.colorbar(valid_im, orientation="vertical", ticks=np.linspace(minimum+0.5,maximum+0.5,n_classes), ax=ax_c[1][0])
    valid_cbar.set_ticklabels(classes)
    ax_c[1][0].set_title("Validation Patches")
    
    classes = load_data.get_classes(predictions)
    maximum = max(classes)
    minimum = min(classes)
    n_classes = len(classes)
    if class_labels:
        for i, label in enumerate(classes):
            if label != 0:
                classes[i] = class_labels[label]
    pred_im = ax_c[1][1].imshow(predictions, cmap=plt.get_cmap("tab20", n_classes), origin="lower", interpolation="none")
    pred_cbar = fig_c.colorbar(pred_im, orientation="vertical", ticks=np.linspace(minimum+0.5,maximum+0.5,n_classes+1), ax=ax_c[1][1])
    pred_cbar.set_ticklabels(classes)
    ax_c[1][1].set_title("Model Predictions")
    
    fig_m, ax_m = plt.subplots(ncols=2,nrows=2, figsize=(12,10))
    
    n_epochs = len(metrics.train_accuracy_scores)
    epochs = np.linspace(1,n_epochs,n_epochs)
   
    f1_classes = list(label_dict.keys())
    if class_labels:
        for i, label in enumerate(f1_classes):
            if label != 0:
                f1_classes[i] = class_labels[label]
    
    ax_m[0][0].plot(epochs,metrics.train_accuracy_scores,label="training")
    ax_m[0][0].plot(epochs,metrics.val_accuracy_scores,label="validation")
    ax_m[0][0].set_ylim((0,1))
    ax_m[0][0].set_title("Accuracy")
    ax_m[0][0].set_xlabel("Epochs")
    ax_m[0][0].set_ylabel("Accuracy")
    ax_m[0][0].legend(bbox_to_anchor=(1.4,1)) 
    
    f1_scores = np.array(metrics.val_f1_scores)
    f1_scores = np.transpose(f1_scores)
    for i, class_scores in enumerate(f1_scores):
        ax_m[0][1].plot(epochs,class_scores,label=f1_classes[i])
        ax_m[0][1].set_ylim((0,1))
    ax_m[0][1].set_title("F1 Scores")
    ax_m[0][1].set_xlabel("Epochs")
    ax_m[0][1].set_ylabel("F1 Score")
    ax_m[0][1].legend(bbox_to_anchor=(1.05,1))
    
    ax_m[1][0].plot(epochs,metrics.train_recall_scores,label="training")
    ax_m[1][0].plot(epochs,metrics.val_recall_scores,label="validation")
    ax_m[1][0].set_ylim((0,1))
    ax_m[1][0].set_title("Recall")
    ax_m[1][0].set_xlabel("Epochs")
    ax_m[1][0].set_ylabel("Recall")
    ax_m[1][0].legend(bbox_to_anchor=(1.05,1))
    
    ax_m[1][1].plot(epochs,metrics.train_precision_scores,label="training")
    ax_m[1][1].plot(epochs,metrics.val_precision_scores,label="validation")
    ax_m[1][1].set_ylim((0,1))
    ax_m[1][1].set_title("Precision")
    ax_m[1][1].set_xlabel("Epochs")
    ax_m[1][1].set_ylabel("Precision")
    ax_m[1][1].legend(bbox_to_anchor=(1.05,1))
    
    
    fig_c.tight_layout()
    fig_m.tight_layout()
    plt.show()