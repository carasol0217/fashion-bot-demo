# config.py

class Config:

    DATA_DIR = 'sampled_data'
    MODEL_SAVE_PATH = 'fashion_model.pth'
    
    # model Parameters
    NUM_CLASSES = 6
    BATCH_SIZE = 64
    NUM_EPOCHS = 10
    LEARNING_RATE = 0.001

    # image transformations
    IMAGE_SIZE = (128, 128)
    MEAN = [0.485, 0.456, 0.406]
    STD = [0.229, 0.224, 0.225]
