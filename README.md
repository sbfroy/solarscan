## How to Train on fe.uia.no

1. **Zip project** and upload it.
2. **Unzip project** in Jupyter using the following command:
    !unzip solarscan.zip -d solarscan
3. **Install requirements** with:
    !pip install -r solarscan/solarscan/requirements.txt

# ####################################################################

1. **Find the best checkpoints**
ls -lt solarscan/solarscan/src/tmp/checkpoints

2. **Save tp .pth** 
num_classes = len(config.CLASS_NAMES)
checkpoint_path = "solarscan/solarscan/src/tmp/checkpoints/###PASTE BEST.ckpt"
model = model.SOLARSCANMODEL.load_from_checkpoint(checkpoint_path, num_classes=num_classes, 
    learning_rate=config.LEARNING_RATE, 
    patience=config.LR_PATIENCE, 
    factor=config.LR_FACTOR)
torch.save(model.state_dict(), 'SOLARSCANMODEL_weights_RESNET18.pth')