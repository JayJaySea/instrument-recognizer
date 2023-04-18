import click
from pathlib import Path
import librosa
import torch
from torch import load
from torchvision import transforms
import torch.nn.functional as F
import matplotlib.pyplot as plt
from PIL import Image
import numpy as np
from os import system
from model import SimpleNet, DenseNet

#transforms.Resize((64, 64)),
@click.command()
@click.option('--play', '-p', is_flag=True , help="Play track after recognition")
@click.option('--model', '-m', default="simple",
      help="Model used for audio recognition. Possible options: dense, simple, resnet")
@click.argument("track")
def main(play, model, track):
    if play:
        system("play " + track + " &> /dev/null &")

    classify_track(track, model)

def classify_track(track, model):
    trans = init_transforms(model)

    try:
        model = torch.load(f"./models/{model}")
    except:
        print("No such model. Possible options: dense, simple, resnet")
        exit(1)

    model.eval()
    labels = ["drum", "guitar", "piano", "unknown", "violin"]
    my_file = Path(track)

    if not my_file.exists():
        print("No such track. Please provide valid path.")
        exit(1)

    try: 
        audio, sr = librosa.load(track, sr=None)
    except:
        print("File isn't valid track.")
        exit(1)

    spectrogram = librosa.feature.melspectrogram(y=audio, sr=sr)
    log_spectrogram = librosa.power_to_db(spectrogram, ref=np.max)
    librosa.display.specshow(log_spectrogram, sr=sr, x_axis="time", y_axis="mel")
    path = "/tmp/spect"
    plt.gcf().savefig(path, dpi=50)

    img = Image.open(path + ".png").convert("RGB") 

    img = trans(img)
    img = torch.unsqueeze(img, 0)


    prediction = model(img)
    prediction = F.softmax(prediction, dim=-1)
    prediction = prediction.argmax()
    print(labels[prediction]) 

def init_transforms(model):
    compose = [
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            )
        ]
    if model == "simple" or model == "dense":
        compose.append(transforms.Resize((64, 64), antialias=None))

    return transforms.Compose(compose)

if __name__ == "__main__":
    main()
