import os

class CFG:
    wandb_key = ""
    seed = 43
    data_dir = "/content/drive/MyDrive/dataset/fodi"
  
    # input for experiment
    model = "vgg"
    batch_size = 128
    learning_rate = 3e-4
    epochs = 5
    img_size = 224
    
    fold = 5
    n_split = 5
    num_workers = int(os.cpu_count()/2)

