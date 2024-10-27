## How to Train on fe.uia.no

1. **Zip project** and upload it.
2. **Unzip project** in Jupyter using the following command:
    !unzip solarscan.zip -d solarscan
3. **Install requirements** with:
    !pip install -r solarscan/solarscan/requirements.txt

# ####################################################################

1. **Find the best checkpoints**
!ls solarscan/solarscan/src/tmp/checkpoints

2. **Save tp .pth** 
num_classes = len(image_datasets['train'].classes)
checkpoint_path = "solarscan/solarscan/src/checkpoints/###PASTE BEST.ckpt"
model = SOLARSCANMODEL.load_from_checkpoint(checkpoint_path, num_classes=num_classes)
torch.save(model.state_dict(), 'SOLARSCANMODEL_weights.pth')