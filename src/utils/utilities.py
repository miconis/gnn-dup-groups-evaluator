import glob
import os.path
import matplotlib.pyplot as plt
import torch


def print_environment_info():
    print(torch.__version__)
    cuda_id = torch.cuda.current_device()
    print("Cuda Device ID:", cuda_id)
    print("Cuda Device Name:", torch.cuda.get_device_name(cuda_id))


def log_dataset_info(logdir, dataset):
    with open(logdir + "log.txt", "a") as file:
        line = f"************************\n" \
               f"Number of groups: {dataset.__len__()}\n" \
               f"Number of bert features: {dataset.__num_abstract_features__()}\n" \
               f"Number of names features: {dataset.__num_names_features__()}\n" \
               f"Number of prediction classes: {dataset.__num_classes__()}\n"
        stats = dataset.__stats__()
        for c in stats.keys():
            line += f"{c}: {stats[c]}\n"
        line += f"***********************\n"
        print(line)
        file.write(line)


def save_checkpoint(model, optimizer, epoch, logdir):
    delete_model(f"{logdir}{model.__class__.__name__}-epoch*.pth")
    model_path = f"{logdir}{model.__class__.__name__}-epoch{epoch}.ckpt.pth"

    torch.save({
        "optimizer": optimizer.state_dict(),
        "model": model.state_dict()
    }, model_path)
    return model_path


def load_checkpoint(model, optimizer, filename):
    checkpoint = torch.load(filename)
    model.load_state_dict(checkpoint['model'])
    optimizer.load_state_dict(checkpoint['optimizer'])


def remove_file(filename):
    try:
        os.remove(filename)
    except:
        return


def delete_model(pattern):
    for f in glob.glob(pattern):
        os.remove(f)


def log_config(logdir, model, optimizer, batch_accum, batch_size):
    with open(logdir + "log.txt", "a") as file:
        line = f"{str(model)}\n{str(optimizer)}\nbatch accumulation: {batch_accum}\nbatch size: {batch_size}\n"
        print(line)
        file.write(line)


def log_epoch_status(logdir, epoch, train_loss, valid_loss, train_acc, valid_acc):
    with open(logdir + "log.txt", "a") as file:
        line = f"EPOCH {epoch:03d} -> Loss[Train:{train_loss:.4f}; Valid:{valid_loss:.4f}] - Accuracy[Train:{train_acc:.4f}; Valid:{valid_acc:.4f}]"
        file.write(line + "\n")
        print(line)


def plot_confusion_matrix(conf_matrix):
    conf_matrix = conf_matrix.numpy().tolist()
    fig, ax = plt.subplots(figsize=(7.5, 7.5))
    ax.matshow(conf_matrix, cmap=plt.cm.Blues, alpha=0.3)
    for i in range(2):
        for j in range(2):
            ax.text(x=j, y=i, s=conf_matrix[i][j], va="center", ha="center")

    plt.xlabel("Predicted", fontsize=18)
    plt.ylabel("Label", fontsize=18)

    return fig


def conf_matrix_metrics(conf_matrix):
    accuracy = (conf_matrix[1, 1] + conf_matrix[0, 0]) / (conf_matrix[0, 0] + conf_matrix[0, 1] + conf_matrix[1, 0] + conf_matrix[1, 1])
    tpr = conf_matrix[1, 1] / (conf_matrix[1, 0] + conf_matrix[1, 1])  # recall
    tnr = conf_matrix[0, 0] / (conf_matrix[0, 0] + conf_matrix[0, 1])
    fnr = conf_matrix[1, 0] / (conf_matrix[1, 0] + conf_matrix[1, 1])
    fpr = conf_matrix[0, 1] / (conf_matrix[0, 0] + conf_matrix[0, 1])
    precision = conf_matrix[1, 1] / (conf_matrix[1, 1] + conf_matrix[0, 1])
    balanced_accuracy = (tpr + tnr) / 2
    f1_score = 2*conf_matrix[1, 1] / (2*conf_matrix[1, 1] + conf_matrix[0, 1] + conf_matrix[1, 0])
    return {
        "Accuracy (A)": accuracy,
        "Balanced Accuracy (BA)": balanced_accuracy,
        "True Positive Rate (TPR) - Recall": tpr,
        "True Negative Rate (TNR)": tnr,
        "False Negative Rate (FNR)": fnr,
        "False Positive Rate (FPR)": fpr,
        "Precision": precision,
        "F1 Score": f1_score
    }


def log_metrics(logdir, classname, metrics):
    line = f"*******{classname}*******\n"
    for metric in metrics.keys():
        line += f"{metric}: {metrics[metric]}\n"
    line += f"***************************\n"
    with open(logdir + "log.txt", "a") as file:
        print(line)
        file.write(line)
