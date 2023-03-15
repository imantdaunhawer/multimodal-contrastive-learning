"""
Experiment with image/text pairs.
"""

import argparse
import json
import os
import random
import uuid
import warnings

import numpy as np
import pandas as pd
import torch
from sklearn.kernel_ridge import KernelRidge
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.metrics import accuracy_score, r2_score
from sklearn.model_selection import GridSearchCV
from sklearn.neural_network import MLPClassifier, MLPRegressor
from sklearn.preprocessing import StandardScaler
from torch.nn.utils import clip_grad_norm_
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.models import resnet18

from datasets import Multimodal3DIdent
from encoders import TextEncoder2D
from utils.infinite_iterator import InfiniteIterator
from utils.losses import infonce_loss


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--datapath", type=str, required=True)
    parser.add_argument("--model-dir", type=str, default="models")
    parser.add_argument("--model-id", type=str, default=None)
    parser.add_argument("--encoding-size", type=int, default=4)
    parser.add_argument("--hidden-size", type=int, default=100)
    parser.add_argument("--tau", type=float, default=1.0)
    parser.add_argument("--lr", type=float, default=1e-5)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--train-steps", type=int, default=100001)
    parser.add_argument("--log-steps", type=int, default=1000)
    parser.add_argument("--checkpoint-steps", type=int, default=10000)
    parser.add_argument("--evaluate", action='store_true')
    parser.add_argument("--seed", type=int, default=np.random.randint(32**2-1))
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--no-cuda", action="store_true")
    parser.add_argument("--save-all-checkpoints", action="store_true")
    parser.add_argument("--load-args", action="store_true")
    args = parser.parse_args()
    return args, parser


def train_step(data, f1, f2, loss_func, optimizer, params):

    # reset grad
    if optimizer is not None:
        optimizer.zero_grad()

    # compute loss
    x1 = data['image']
    x2 = data['text']
    hz1 = f1(x1)
    hz2 = f2(x2)
    loss_value1 = loss_func(hz1, hz2)
    loss_value2 = loss_func(hz2, hz1)
    loss_value = 0.5 * (loss_value1 + loss_value2)  # symmetrized infonce loss

    # backprop
    if optimizer is not None:
        loss_value.backward()
        clip_grad_norm_(params, max_norm=2.0, norm_type=2)  # stabilizes training
        optimizer.step()

    return loss_value.item()


def val_step(data, f1, f2, loss_func):
    return train_step(data, f1, f2, loss_func, optimizer=None, params=None)


def get_data(dataset, f1, f2, loss_func, dataloader_kwargs):
    loader = DataLoader(dataset, **dataloader_kwargs)
    iterator = InfiniteIterator(loader)
    labels_image = {v: [] for v in Multimodal3DIdent.FACTORS["image"].values()}
    labels_text = {v: [] for v in Multimodal3DIdent.FACTORS["text"].values()}
    rdict = {"hz_image": [], "hz_text": [], "loss_values": [],
             "labels_image": labels_image, "labels_text": labels_text}
    i = 0
    with torch.no_grad():
        while (i < len(dataset)):  # NOTE: can yield slightly too many samples

            # load batch
            i += loader.batch_size
            data = next(iterator)  # contains images, texts, and labels

            # compute loss
            loss_value = val_step(data, f1, f2, loss_func)
            rdict["loss_values"].append([loss_value])

            # collect representations
            hz_image = f1(data["image"])
            hz_text = f2(data["text"])
            rdict["hz_image"].append(hz_image.detach().cpu().numpy())
            rdict["hz_text"].append(hz_text.detach().cpu().numpy())

            # collect image labels
            for k in rdict["labels_image"]:
                labels_k = data["z_image"][k]
                rdict["labels_image"][k].append(labels_k)

            # collect text labels
            for k in rdict["labels_text"]:
                labels_k = data["z_text"][k]
                rdict["labels_text"][k].append(labels_k)

    # concatenate each list of values along the batch dimension
    for k, v in rdict.items():
        if type(v) == list:
            rdict[k] = np.concatenate(v, axis=0)
        elif type(v) == dict:
            for k2, v2 in v.items():
                rdict[k][k2] = np.concatenate(v2, axis=0)
    return rdict


def evaluate_prediction(model, metric, X_train, y_train, X_test, y_test):
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    return metric(y_test, y_pred)


def main():

    # parse args
    args, parser = parse_args()

    # create save_dir, where the model/results are or will be saved
    if args.model_id is None:
        setattr(args, "model_id", uuid.uuid4())
    args.save_dir = os.path.join(args.model_dir, args.model_id)
    if not os.path.exists(args.save_dir):
        os.makedirs(args.save_dir)

    # optionally, reuse existing arguments from args.json (only for evaluation)
    if args.evaluate and args.load_args:
        with open(os.path.join(args.save_dir, 'args.json'), 'r') as fp:
            loaded_args = json.load(fp)
        arguments_to_load = ["encoding_size", "hidden_size"]
        for arg in arguments_to_load:
            setattr(args, arg, loaded_args[arg])
        # NOTE: Any new arguments that shall be automatically loaded for the
        # evaluation of a trained model must be added to 'arguments_to_load'.

    # print args
    print("Arguments:")
    for k, v in vars(args).items():
        print(f"\t{k}: {v}")

    # save args to disk (only for training)
    if not args.evaluate:
        with open(os.path.join(args.save_dir, 'args.json'), 'w') as fp:
            json.dump(args.__dict__, fp)

    # set all seeds
    if args.seed is not None:
        np.random.seed(args.seed)
        random.seed(args.seed)
        torch.manual_seed(args.seed)

    # set device
    if torch.cuda.is_available() and not args.no_cuda:
        device = "cuda"
    else:
        device = "cpu"
        warnings.warn("cuda is not available or --no-cuda was set.")

    # define similarity metric and loss function
    sim_metric = torch.nn.CosineSimilarity(dim=-1)
    criterion = torch.nn.CrossEntropyLoss()
    loss_func = lambda z1, z2: infonce_loss(
        z1, z2, sim_metric=sim_metric, criterion=criterion, tau=args.tau)

    # define augmentations (only normalization of the input images)
    mean_per_channel = [0.4327, 0.2689, 0.2839]  # values from 3DIdent
    std_per_channel = [0.1201, 0.1457, 0.1082]   # values from 3DIdent
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean_per_channel, std_per_channel)])

    # define kwargs
    dataset_kwargs = {"transform": transform}
    dataloader_kwargs = {
        "batch_size": args.batch_size, "shuffle": True, "drop_last": True,
        "num_workers": args.workers, "pin_memory": True}

    # define dataloaders
    train_dataset = Multimodal3DIdent(args.datapath, mode="train", **dataset_kwargs)
    vocab_filepath = train_dataset.vocab_filepath
    if args.evaluate:
        val_dataset = Multimodal3DIdent(args.datapath, mode="val",
                                        vocab_filepath=vocab_filepath,
                                        **dataset_kwargs)
        test_dataset = Multimodal3DIdent(args.datapath, mode="test",
                                         vocab_filepath=vocab_filepath,
                                         **dataset_kwargs)
    else:
        train_loader = DataLoader(train_dataset, **dataloader_kwargs)
        train_iterator = InfiniteIterator(train_loader)

    # define image encoder
    encoder_img = torch.nn.Sequential(
        resnet18(num_classes=args.hidden_size),
        torch.nn.LeakyReLU(),
        torch.nn.Linear(args.hidden_size, args.encoding_size))
    encoder_img = torch.nn.DataParallel(encoder_img)
    encoder_img.to(device)

    # define text encoder
    sequence_length = train_dataset.max_sequence_length
    encoder_txt = TextEncoder2D(
        input_size=train_dataset.vocab_size,
        output_size=args.encoding_size,
        sequence_length=sequence_length)
    encoder_txt = torch.nn.DataParallel(encoder_txt)
    encoder_txt.to(device)

    # for evaluation, always load saved encoders
    if args.evaluate:
        path_img = os.path.join(args.save_dir, "encoder_img.pt")
        path_txt = os.path.join(args.save_dir, "encoder_txt.pt")
        encoder_img.load_state_dict(torch.load(path_img, map_location=device))
        encoder_txt.load_state_dict(torch.load(path_txt, map_location=device))

    # define the optimizer
    params = list(encoder_img.parameters())+list(encoder_txt.parameters())
    optimizer = torch.optim.Adam(params, lr=args.lr)

    # training
    # --------
    if not args.evaluate:

        # training loop
        step = 1
        loss_values = []  # list to keep track of loss values
        while (step <= args.train_steps):

            # training step
            data = next(train_iterator)  # contains images, texts, and labels
            loss_value = train_step(data, encoder_img, encoder_txt, loss_func, optimizer, params)
            loss_values.append(loss_value)

            # print average loss value
            if step % args.log_steps == 1 or step == args.train_steps:
                print(f"Step: {step} \t",
                      f"Loss: {loss_value:.4f} \t",
                      f"<Loss>: {np.mean(loss_values[-args.log_steps:]):.4f} \t")

            # save models and intermediate checkpoints
            if step % args.checkpoint_steps == 1 or step == args.train_steps:
                torch.save(encoder_img.state_dict(), os.path.join(args.save_dir, "encoder_img.pt"))
                torch.save(encoder_txt.state_dict(), os.path.join(args.save_dir, "encoder_txt.pt"))
                if args.save_all_checkpoints:
                    torch.save(encoder_img.state_dict(), os.path.join(args.save_dir, "encoder_img_%d.pt" % step))
                    torch.save(encoder_txt.state_dict(), os.path.join(args.save_dir, "encoder_txt_%d.pt" % step))
            step += 1

    # evaluation
    # ----------
    if args.evaluate:

        # collect encodings and labels from the validation and test data
        val_dict = get_data(val_dataset, encoder_img, encoder_txt, loss_func, dataloader_kwargs)
        test_dict = get_data(test_dataset, encoder_img, encoder_txt, loss_func, dataloader_kwargs)

        # print average loss values
        print(f"<Val Loss>: {np.mean(val_dict['loss_values']):.4f}")
        print(f"<Test Loss>: {np.mean(test_dict['loss_values']):.4f}")

        # handle edge case when the encodings are 1-dimensional
        if args.encoding_size == 1:
            for m in ["image", "text"]:
                val_dict[f"hz_{m}"] = val_dict[f"hz_{m}"].reshape(-1, 1)
                test_dict[f"hz_{m}"] = test_dict[f"hz_{m}"].reshape(-1, 1)

        # standardize the encodings
        for m in ["image", "text"]:
            scaler = StandardScaler()
            val_dict[f"hz_{m}"] = scaler.fit_transform(val_dict[f"hz_{m}"])
            test_dict[f"hz_{m}"] = scaler.transform(test_dict[f"hz_{m}"])

        # evaluate how well each factor can be predicted from the encodings
        results = []
        for m in ["image", "text"]:
            factors_m = Multimodal3DIdent.FACTORS[m]
            discrete_factors_m = Multimodal3DIdent.DISCRETE_FACTORS[m]
            for ix, factor_name in factors_m.items():

                # select data
                train_inputs = val_dict[f"hz_{m}"]
                test_inputs = test_dict[f"hz_{m}"]
                train_labels = val_dict[f"labels_{m}"][factor_name]
                test_labels = test_dict[f"labels_{m}"][factor_name]
                data = [train_inputs, train_labels, test_inputs, test_labels]
                r2_linreg, r2_krreg, acc_logreg, acc_mlp = [np.nan] * 4

                # check if factor ix is discrete for modality m
                if ix in discrete_factors_m:
                    factor_type = "discrete"
                else:
                    factor_type = "continuous"

                # for continuous factors, do regression and compute R2 score
                if factor_type == "continuous":
                    # linear regression
                    linreg = LinearRegression(n_jobs=-1)
                    r2_linreg = evaluate_prediction(linreg, r2_score, *data)
                    # nonlinear regression
                    gskrreg = GridSearchCV(
                        KernelRidge(kernel='rbf', gamma=0.1),
                        param_grid={"alpha": [1e0, 0.1, 1e-2, 1e-3],
                                    "gamma": np.logspace(-2, 2, 4)},
                        cv=3, n_jobs=-1)
                    r2_krreg = evaluate_prediction(gskrreg, r2_score, *data)
                    # NOTE: MLP is a lightweight alternative
                    # r2_krreg = evaluate_prediction(
                    #     MLPRegressor(max_iter=1000), r2_score, *data)

                # for discrete factors, do classification and compute accuracy
                if factor_type == "discrete":
                    # logistic classification
                    logreg = LogisticRegression(n_jobs=-1, max_iter=1000)
                    acc_logreg = evaluate_prediction(logreg, accuracy_score, *data)
                    # nonlinear classification
                    mlpreg = MLPClassifier(max_iter=1000)
                    acc_mlp = evaluate_prediction(mlpreg, accuracy_score, *data)

                # append results
                results.append([ix, m, factor_name, factor_type,
                                r2_linreg, r2_krreg, acc_logreg, acc_mlp])

        # convert evaluation results into tabular form
        columns = ["ix", "modality", "factor_name", "factor_type",
                   "r2_linreg", "r2_krreg", "acc_logreg", "acc_mlp"]
        df_results = pd.DataFrame(results, columns=columns)
        df_results.to_csv(os.path.join(args.save_dir, "results.csv"))
        print(df_results.to_string())


if __name__ == "__main__":
    main()
