"""Entrypoint"""
import os
from datetime import datetime

import torch
import numpy as np

from dgl.dataloading import GraphDataLoader
from dgl.data.utils import split_dataset
from torch.utils.tensorboard import SummaryWriter
from torchmetrics.classification import BinaryConfusionMatrix

from utils.datasets import DedupGroupsDataset
from utils.models import *
from utils.utilities import *
from utils.config import *

torch.manual_seed(42)
os.environ['TORCH'] = torch.__version__
checkpoint_model = f"./log/SmallGraphormer/17-05-2023_23-56-21/SmallGraphormer-epoch{START_EPOCH-1}.ckpt.pth"
current_date = datetime.now().strftime("%d-%m-%Y_%H-%M-%S")
min_valid_loss = np.inf
best_model_path = ""


# creating the dataset
dataset = DedupGroupsDataset(dataset_name="groups",
                             raw_dir="../pubmed_dataset/raw",
                             dedup_groups_path="../pubmed_dataset/groupentities",
                             save_dir="../pubmed_dataset/processed")

dataset_splittings = split_dataset(dataset=dataset, frac_list=[0.6, 0.2, 0.2], shuffle=True, random_state=42)

train_loader = GraphDataLoader(dataset_splittings[0], batch_size=BATCH_SIZE)
valid_loader = GraphDataLoader(dataset_splittings[1], batch_size=BATCH_SIZE)
test_loader = GraphDataLoader(dataset_splittings[2], batch_size=BATCH_SIZE)
train_size = len(train_loader.dataset)
valid_size = len(valid_loader.dataset)
test_size = len(test_loader.dataset)

# GCN MODELS
model = GCN3(num_features=dataset.__num_abstract_features__(),
             hidden_dim=dataset.__num_abstract_features__(),
             dropout=0.2)
# model = GCN3EdgeWeight(num_features=dataset.__num_abstract_features__(),
#                        hidden_dim=dataset.__num_abstract_features__(),
#                        dropout=0.2)
# model = GCN3Names(num_features=dataset.__num_abstract_features__() + dataset.__num_names_features(),
#                   hidden_dim=dataset.__num_abstract_features__() + dataset.__num_names_features(),
#                   dropout=0.2)

# GAT MODELS
# model = GAT3(num_features=dataset.__num_abstract_features__(),
#              hidden_dim=dataset.__num_abstract_features__(),
#              num_heads=8,
#              feat_drop=0.2,
#              attn_drop=0.2,
#              dropout=0.2)
# model = GAT3LSTM(num_features=dataset.__num_abstract_features__(),
#                  hidden_dim=dataset.__num_abstract_features__(),
#                  dropout=0.2,
#                  feat_drop=0.2,
#                  attn_drop=0.2,
#                  num_layers=2,
#                  num_heads=16)
# model = GAT3NamesLSTM(num_features=dataset.__num_abstract_features__() + dataset.__num_names_features__(),
#                       hidden_dim=dataset.__num_abstract_features__() + dataset.__num_names_features__(),
#                       dropout=0.2,
#                       feat_drop=0.2,
#                       attn_drop=0.2,
#                       num_layers=2,
#                       num_heads=16)
# model = GAT3NamesEdgesLSTM(num_features=dataset.__num_abstract_features__() + dataset.__num_names_features__(),
#                            hidden_dim=dataset.__num_abstract_features__() + dataset.__num_names_features__(),
#                            dropout=0.2,
#                            feat_drop=0.2,
#                            attn_drop=0.2,
#                            num_layers=2,
#                            num_heads=16)
# model = GAT3NamesEdgesCentralityLSTM(num_features=dataset.__num_abstract_features__() + dataset.__num_names_features(),
#                         hidden_dim=dataset.__num_abstract_features__() + dataset.__num_names_features(),
#                         dropout=0.2,
#                         feat_drop=0.2,
#                         attn_drop=0.2,
#                         num_layers=2,
#                         num_heads=16)

# GRAPHORMER MODELS
# model = SmallGraphormer(num_features=dataset.__num_abstract_features__(),
#                         transformer_hidden_dim=64,
#                         dropout=0.2,
#                         attn_dropout=0.2,
#                         num_heads=8)


train_writer = SummaryWriter(log_dir=f"./log/{model.__class__.__name__}/{current_date}/training")
valid_writer = SummaryWriter(log_dir=f"./log/{model.__class__.__name__}/{current_date}/validation")
test_writer = SummaryWriter(log_dir=f"./log/{model.__class__.__name__}/{current_date}/test")
logdir = f"./log/{model.__class__.__name__}/{current_date}/"
model = model.to(DEVICE)

optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE)
loss_criterion = F.binary_cross_entropy

# log info
log_dataset_info(logdir, dataset)
log_config(logdir, model, optimizer, BATCH_ACCUM, BATCH_SIZE)

if START_EPOCH > 1:
    print("Resuming from checkpoint")
    load_checkpoint(model, optimizer, checkpoint_model)

counter = EARLY_STOPPING
for epoch in range(START_EPOCH, EPOCHS + 1):
    # TRAINING
    train_loss = 0.0
    correct = 0
    model.train()
    accum_iter = 0
    for batched_graph, labels in train_loader:
        accum_iter += 1
        batched_graph, labels = batched_graph.to(DEVICE), labels.to(DEVICE)
        nodes_features = batched_graph.ndata["features"]
        prediction = model(batched_graph, nodes_features).squeeze()
        correct += (prediction.round() == labels).sum().item()
        loss = loss_criterion(prediction, labels.float())
        train_loss += loss.item() * batched_graph.batch_size
        loss = loss / BATCH_ACCUM
        loss.backward()
        if accum_iter == BATCH_ACCUM:
            accum_iter = 0
            optimizer.step()
            optimizer.zero_grad()
    train_acc = correct / train_size * 100
    train_loss = train_loss / train_size
    train_writer.add_scalar("Loss", train_loss, epoch)
    train_writer.add_scalar("Accuracy", train_acc, epoch)
    # VALIDATION
    valid_loss = 0.0
    correct = 0
    model.eval()
    for batched_graph, labels in valid_loader:
        batched_graph, labels = batched_graph.to(DEVICE), labels.to(DEVICE)
        nodes_features = batched_graph.ndata["features"]
        prediction = model(batched_graph, nodes_features).squeeze()
        correct += (prediction.round() == labels).sum().item()
        loss = loss_criterion(prediction, labels.float())
        valid_loss += loss.item() * batched_graph.batch_size
    valid_acc = correct / valid_size * 100
    valid_loss = valid_loss / valid_size
    valid_writer.add_scalar("Loss", valid_loss, epoch)
    valid_writer.add_scalar("Accuracy", valid_acc, epoch)
    log_epoch_status(logdir, epoch, train_loss, valid_loss, train_acc, valid_acc)
    train_writer.flush()
    valid_writer.flush()
    if min_valid_loss >= valid_loss:
        counter = EARLY_STOPPING
        min_valid_loss = valid_loss
        best_model_path = save_checkpoint(model, optimizer, epoch, logdir)
    counter -= 1
    # EARLY STOPPING
    if counter <= 0:
        print("Early stopping!")
        break
train_writer.close()
valid_writer.close()

# EVALUATION
print(f"Best model location: {best_model_path}")
load_checkpoint(model, optimizer, best_model_path)
model.eval()
test_classes = []
test_predictions = []
test_labels = []
for batched_graph, labels in test_loader:
    batched_graph, labels = batched_graph.to(DEVICE), labels.to(DEVICE)
    test_classes.extend([g.number_of_nodes() for g in dgl.unbatch(batched_graph)])
    test_labels.extend(labels)
    test_predictions.extend([pred[0] for pred in model(batched_graph, batched_graph.ndata["features"]).cpu().detach().numpy().tolist()])

# prepare tensor for binary cross entropy
stats_tensor = torch.tensor([test_classes, test_predictions, test_labels])
bcm = BinaryConfusionMatrix(threshold=0.5)

confusion_matrices = {}
# global confusion matrix
conf_matrix = bcm(stats_tensor[1, :], stats_tensor[2, :])
confusion_matrices["global"] = conf_matrix
# 3 confusion matrix
conf_matrix = bcm(
    stats_tensor[:, (stats_tensor[0, :] == 3).nonzero().squeeze(1)][1, :],
    stats_tensor[:, (stats_tensor[0, :] == 3).nonzero().squeeze(1)][2, :]
)
confusion_matrices["3"] = conf_matrix
# from 4 to 10 confusion matrix
tmp_stats_tensor = stats_tensor[:, (stats_tensor[0, :] <= 10).nonzero().squeeze(1)]
conf_matrix = bcm(
    tmp_stats_tensor[:, (tmp_stats_tensor[0, :] >= 4).nonzero().squeeze(1)][1, :],
    tmp_stats_tensor[:, (tmp_stats_tensor[0, :] >= 4).nonzero().squeeze(1)][2, :]
)
confusion_matrices["4to10"] = conf_matrix
# above 10 confusion matrix
conf_matrix = bcm(
    stats_tensor[:, (stats_tensor[0, :] > 10).nonzero().squeeze(1)][1, :],
    stats_tensor[:, (stats_tensor[0, :] > 10).nonzero().squeeze(1)][2, :]
)
confusion_matrices["above10"] = conf_matrix

for key in confusion_matrices.keys():
    test_writer.add_figure(f"{key} Confusion Matrix", plot_confusion_matrix(confusion_matrices[key]))
    log_metrics(logdir, key, conf_matrix_metrics(confusion_matrices[key]))
test_writer.close()
