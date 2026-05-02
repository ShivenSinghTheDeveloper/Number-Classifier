import numpy as np
import matplotlib.pyplot as plt

def show_samples(images, labels, n = 8):
    n = min(n, len(images))
    cols = min(n, 8)
    rows = int(np.ceil(n / cols))
    fig, axis = plt.subplots(rows, cols, figsize = (2 * cols, 2 * rows))
    axis = np.array(axis).reshape(-1)
    for i in range(rows * cols):
        ax = axis[i]
    
        #Situation: What if there some images left? 
        if i < n:
            ax.imshow(images[i], cmap="gray")
            ax.set_title(f"Label: {labels[i]}")
        #Hide axis lines
        ax.axis("off")
        
    #Display everything on samples
    plt.tight_layout()
    plt.show()

#Make the show error functions
def show_errors(x_test, y_test, y_pred, n=12):
    wrong = np.where(y_pred!=y_test)[0]
    if (len(wrong) == 0):
        print("There are no mistakes")
        return
    n = min(n, len(wrong))
    cols = 4
    rows = int(np.ceil(n/cols))  
    fig, axis = plt.subplots(rows, cols, figsize = (10, 2.5 * rows))
    axis = np.array(axis).reshape(-1)
    for i in range(rows * cols):
        ax = axis[i]
        if i < n:
            idx = wrong[i]
            img = x_test[idx].reshape(8,8)
            ax.imshow(img, cmap="gray")
            ax.set_title(f"Pred: {y_pred[idx]} | True: {y_test[idx]}")
        ax.axis("off")
    plt.tight_layout()
    plt.show()
    