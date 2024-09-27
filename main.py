from train.train import train_model
from config.config import Config  
from train.evaluate import evaluate_model
if __name__ == "__main__":
 
    train_model(
        root_dir=Config.DATA_DIR, 
        num_classes=Config.NUM_CLASSES, 
        num_epochs=Config.NUM_EPOCHS, 
        batch_size=Config.BATCH_SIZE
    )

    # evaluate_model(
    #     root_dir=Config.DATA_DIR, 
    #     num_classes=Config.NUM_CLASSES, 
    #     batch_size=Config.BATCH_SIZE
    # )