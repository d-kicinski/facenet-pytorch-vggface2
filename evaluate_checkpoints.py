import dataclasses
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping, Any, List

import numpy as np
import argparse
import os
import gc
import torch
import torchvision.transforms as transforms
from torch.nn.modules.distance import PairwiseDistance
from torch.utils.tensorboard import SummaryWriter
from torch import nn
from datasets.LFWDataset import LFWDataset
from validate_on_LFW import evaluate_lfw
from tqdm import tqdm
from models.inceptionresnetv2 import InceptionResnetV2Triplet
from models.mobilenetv2 import MobileNetV2Triplet
from models.resnet import (
    Resnet18Triplet,
    Resnet34Triplet,
    Resnet50Triplet,
    Resnet101Triplet,
    Resnet152Triplet
)


parser = argparse.ArgumentParser(description="Training a FaceNet facial recognition model using Triplet Loss.")
parser.add_argument('--lfw', type=str, required=True,
                    help="(REQUIRED) Absolute path to the labeled faces in the wild dataset folder"
                    )
parser.add_argument('--model_architecture', type=str, default="resnet18", choices=["resnet18",
                                                                            "resnet34", "resnet50", "resnet101", "resnet152", "inceptionresnetv2", "mobilenetv2"],
                    help="The required model architecture for training: ('resnet18','resnet34', 'resnet50', 'resnet101', 'resnet152', 'inceptionresnetv2', 'mobilenetv2'), (default: 'resnet18')"
                    )
parser.add_argument('--embedding_dimension', default=256, type=int,
                    help="Dimension of the embedding vector (default: 512)"
                    )
parser.add_argument('--lfw_batch_size', default=64, type=int,
                    help="Batch size for LFW dataset (default: 320)"
                    )
parser.add_argument('--resume_path', default="model_training_checkpoints",  type=Path,
                    help='path to latest model checkpoint: (model_training_checkpoints/model_resnet18_epoch_1.pt file) (default: None)'
                    )
parser.add_argument('--num_workers', default=4, type=int,
                    help="Number of workers for data loaders (default: 2)"
                    )
parser.add_argument('--margin', default=0.2, type=float,
                    help='margin for triplet loss (default: 0.2)'
                    )
parser.add_argument('--image_size', default=224, type=int,
                    help='Input image size (default: 224 (224x224), must be 299x299 for Inception-ResNet-V2)'
                    )
args = parser.parse_args()


def set_model_architecture(model_architecture, pretrained, embedding_dimension):
    if model_architecture == "resnet18":
        model = Resnet18Triplet(
            embedding_dimension=embedding_dimension,
            pretrained=pretrained
        )
    elif model_architecture == "resnet34":
        model = Resnet34Triplet(
            embedding_dimension=embedding_dimension,
            pretrained=pretrained
        )
    elif model_architecture == "resnet50":
        model = Resnet50Triplet(
            embedding_dimension=embedding_dimension,
            pretrained=pretrained
        )
    elif model_architecture == "resnet101":
        model = Resnet101Triplet(
            embedding_dimension=embedding_dimension,
            pretrained=pretrained
        )
    elif model_architecture == "resnet152":
        model = Resnet152Triplet(
            embedding_dimension=embedding_dimension,
            pretrained=pretrained
        )
    elif model_architecture == "inceptionresnetv2":
        model = InceptionResnetV2Triplet(
            embedding_dimension=embedding_dimension,
            pretrained=pretrained
        )
    elif model_architecture == "mobilenetv2":
        model = MobileNetV2Triplet(
            embedding_dimension=embedding_dimension,
            pretrained=pretrained
        )
    print("Using {} model architecture.".format(model_architecture))

    return model


def validate_lfw(model, lfw_dataloader, model_architecture, epoch, epochs, log=False):
    with torch.no_grad():
        l2_distance = PairwiseDistance(p=2)
        distances, labels = [], []

        print("Validating on LFW! ...")
        progress_bar = enumerate(tqdm(lfw_dataloader))

        for batch_index, (data_a, data_b, label) in progress_bar:
            data_a = data_a.cpu()
            data_b = data_b.cpu()

            output_a, output_b = model(data_a), model(data_b)
            distance = l2_distance.forward(output_a, output_b)  # Euclidean distance

            distances.append(distance.cpu().detach().numpy())
            labels.append(label.cpu().detach().numpy())

        labels = np.array([sublabel for label in labels for sublabel in label])
        distances = np.array([subdist for distance in distances for subdist in distance])

        true_positive_rate, false_positive_rate, precision, recall, accuracy, roc_auc, best_distances, \
        tar, far = evaluate_lfw(
            distances=distances,
            labels=labels,
            far_target=1e-3
        )

        if log:
            # Print statistics and add to log
            print("Accuracy on LFW: {:.4f}+-{:.4f}\tPrecision {:.4f}+-{:.4f}\tRecall {:.4f}+-{:.4f}\t"
                  "ROC Area Under Curve: {:.4f}\tBest distance threshold: {:.2f}+-{:.2f}\t"
                  "TAR: {:.4f}+-{:.4f} @ FAR: {:.4f}".format(
                        np.mean(accuracy),
                        np.std(accuracy),
                        np.mean(precision),
                        np.std(precision),
                        np.mean(recall),
                        np.std(recall),
                        roc_auc,
                        np.mean(best_distances),
                        np.std(best_distances),
                        np.mean(tar),
                        np.std(tar),
                        np.mean(far)
                    )
            )
            with open('logs/lfw_{}_log_triplet.txt'.format(model_architecture), 'a') as f:
                val_list = [
                    epoch,
                    np.mean(accuracy),
                    np.std(accuracy),
                    np.mean(precision),
                    np.std(precision),
                    np.mean(recall),
                    np.std(recall),
                    roc_auc,
                    np.mean(best_distances),
                    np.std(best_distances),
                    np.mean(tar)
                ]
                log = '\t'.join(str(value) for value in val_list)
                f.writelines(log + '\n')

    return EvaluationMetrics(
        accuracy=float(np.mean(accuracy)),
        precision=float(np.mean(precision)),
        recall=float(np.mean(recall)),
        roc_auc=roc_auc,
        tar=float(np.mean(tar)),
        far=float(np.mean(far)),
        distance=float(np.mean(best_distances)),
    )


@dataclass
class EvaluationMetrics:
    accuracy: float
    precision: float
    recall: float
    roc_auc: float
    tar: float
    far: float
    distance: float


def forward_pass(imgs, model, batch_size):
    imgs = imgs.cpu()
    embeddings = model(imgs)

    # Split the embeddings into Anchor, Positive, and Negative embeddings
    anc_embeddings = embeddings[:batch_size]
    pos_embeddings = embeddings[batch_size: batch_size * 2]
    neg_embeddings = embeddings[batch_size * 2:]

    # Free some memory
    del imgs, embeddings
    gc.collect()

    return anc_embeddings, pos_embeddings, neg_embeddings, model


class Tensorboard:
    def __init__(self, log_path: Path):
        self._writer = SummaryWriter(str(log_path))

    def add_dict(self, dictionary: Mapping[str, Any], global_step: int):
        for key, value in dictionary.items():
            self._writer.add_scalar(key, value, global_step=global_step)

    def add_scalar(self, name: str, value: float, global_step: int):
        self._writer.add_scalar(name, value, global_step=global_step)


def main():
    lfw_dataroot = args.lfw
    model_architecture = args.model_architecture
    embedding_dimension = args.embedding_dimension
    lfw_batch_size = args.lfw_batch_size
    resume_path = args.resume_path
    num_workers = args.num_workers
    image_size = args.image_size

    lfw_transforms = transforms.Compose([
        transforms.Resize(size=image_size),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.6068, 0.4517, 0.3800],
            std=[0.2492, 0.2173, 0.2082]
        )
    ])

    lfw_dataloader = torch.utils.data.DataLoader(
        dataset=LFWDataset(
            dir=lfw_dataroot,
            pairs_path='datasets/LFW_pairs.txt',
            transform=lfw_transforms
        ),
        batch_size=lfw_batch_size,
        num_workers=num_workers,
        shuffle=False
    )

    # Instantiate model
    model: nn.Module = set_model_architecture(
        model_architecture=model_architecture,
        pretrained=False,
        embedding_dimension=embedding_dimension
    )
    model = model.cpu()

    tensorboard_dir = Path("eval/tensorboard")
    tensorboard_dir.mkdir(exist_ok=True, parents=True)
    tensorboard = Tensorboard(tensorboard_dir)

    checkpoints_dir: Path
    for file in sorted(resume_path.iterdir()):
        epoch: int = -1
        if file.is_file():
            print("Loading checkpoint {} ...".format(file))
            checkpoint = torch.load(str(file), map_location=torch.device("cpu"))
            model.load_state_dict(checkpoint["model_state_dict"])
            epoch = checkpoint["epoch"]

        model.eval()
        metrics = validate_lfw(
            model=model,
            lfw_dataloader=lfw_dataloader,
            model_architecture=model_architecture,
            epoch=epoch,
            epochs=50,
            log=True
        )
        tensorboard.add_dict(dataclasses.asdict(metrics), epoch)


if __name__ == '__main__':
    main()
