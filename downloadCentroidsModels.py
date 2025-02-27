import gdown

# For market1501 model
url = "https://drive.google.com/uc?id=1ZFywKEytpyNocUQd2APh2XqTe8X0HMom"
output = "./reid/centroids-reid/models/market1501_resnet50_256_128_epoch_120.ckpt"
gdown.download(url, output, quiet=False)

# For dukemtmc model
url = "https://drive.google.com/uc?id=1w9yzdP_5oJppGIM4gs3cETyLujanoHK8"
output = "./reid/centroids-reid/models/dukemtmcreid_resnet50_256_128_epoch_120.ckpt"
gdown.download(url, output, quiet=False)