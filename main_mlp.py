"""
Numerical simulation for the multimodal setup.

This code builds on the following projects:
- https://github.com/brendel-group/cl-ica
- https://github.com/ysharma1126/ssl_identifiability
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
from scipy.stats import wishart
from sklearn import kernel_ridge, linear_model
from sklearn.metrics import r2_score
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler
from torch.nn.utils import clip_grad_norm_

import encoders
from utils.invertible_network_utils import construct_invertible_mlp
from utils.latent_spaces import LatentSpace, NRealSpace, ProductLatentSpace
from utils.losses import LpSimCLRLoss


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-dir", type=str, default="models")
    parser.add_argument("--model-id", type=str, default=None)
    parser.add_argument("--encoding-size", type=int, default=5)
    parser.add_argument("--content-n", type=int, default=5)
    parser.add_argument("--style-n", type=int, default=5)
    parser.add_argument("--modality-n", type=int, default=5)
    parser.add_argument("--style-change-prob", type=float, default=1.0)
    parser.add_argument("--statistical-dependence", action='store_true')
    parser.add_argument("--content-dependent-style", action='store_true')
    parser.add_argument("--c-param", type=float, default=1.0)
    parser.add_argument("--m-param", type=float, default=1.0)
    parser.add_argument("--n-mixing-layer", type=int, default=3)
    parser.add_argument("--shared-mixing", action='store_true')
    parser.add_argument("--shared-encoder", action='store_true')
    parser.add_argument("--seed", type=int, default=np.random.randint(32**2-1))
    parser.add_argument("--batch-size", type=int, default=6144)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--train-steps", type=int, default=300001)
    parser.add_argument("--log-steps", type=int, default=1000)
    parser.add_argument("--evaluate", action='store_true')
    parser.add_argument("--num-eval-batches", type=int, default=5)
    parser.add_argument("--permuted-content", action="store_true")
    parser.add_argument("--mlp-eval", action="store_true")
    parser.add_argument("--no-cuda", action="store_true")
    parser.add_argument("--load-args", action="store_true")
    args = parser.parse_args()

    return args, parser


def train_step(data, h1, h2, loss_func, optimizer, params):

    # reset grad
    if optimizer is not None:
        optimizer.zero_grad()

    # compute symmetrized loss
    z1, z2, z1_, z2_ = data
    hz1 = h1(z1)
    hz1_ = h1(z1_)
    hz2 = h2(z2)
    hz2_ = h2(z2_)
    total_loss_value1, _, _ = loss_func(z1, z2, z1_, hz1, hz2, hz1_)
    total_loss_value2, _, _ = loss_func(z2, z1, z2_, hz2, hz1, hz2_)
    total_loss_value = 0.5 * (total_loss_value1 + total_loss_value2)

    # backprop
    if optimizer is not None:
        total_loss_value.backward()
        clip_grad_norm_(params, max_norm=2.0, norm_type=2)  # stabilizes training
        optimizer.step()

    return total_loss_value.item()


def val_step(data, h1, h2, loss_func):
    return train_step(data, h1, h2, loss_func, optimizer=None, params=None)


def generate_data(latent_space, h1, h2, device, num_batches=1, batch_size=4096,
                  loss_func=None, permuted_content=False):

    rdict = {k: [] for k in
             ["c", "s", "s~", "m1", "m2", "c'", "hz1", "hz2", "loss_values"]}
    with torch.no_grad():
        for _ in range(num_batches):

            # sample batch of latents
            z1, z2 = latent_space.sample_z1_and_z2(batch_size, device)

            # compute representations
            hz1 = h1(z1)
            hz2 = h2(z2)
            if permuted_content:
                nc = latent_space.content_n
                z1_intervened = z1.clone()
                z2_intervened = z2.clone()
                perm = torch.randperm(len(z1))
                z1_intervened[:, :nc] = z1_intervened[perm, :nc]
                z2_intervened[:, :nc] = z2_intervened[perm, :nc]
                hz1 = h1(z1_intervened)
                hz2 = h2(z2_intervened)
                c_perm = z1_intervened[:, 0:nc]

            # compute loss
            if loss_func is not None:
                z1_, z2_ = latent_space.sample_z1_and_z2(batch_size, device)
                data = [z1, z2, z1_, z2_]
                loss_value = val_step(data, h1, h2, loss_func)
                rdict["loss_values"].append([loss_value])

            # partition latents into content, style, modality-specific factors
            c, s1, m1 = latent_space.zi_to_csmi(z1)
            _, s2, m2 = latent_space.zi_to_csmi(z2)  # NOTE: same content c

            # collect labels and representations
            rdict["c"].append(c.detach().cpu().numpy())
            rdict["s"].append(s1.detach().cpu().numpy())
            rdict["s~"].append(s2.detach().cpu().numpy())
            rdict["m1"].append(m1.detach().cpu().numpy())
            rdict["m2"].append(m2.detach().cpu().numpy())
            rdict["hz1"].append(hz1.detach().cpu().numpy())
            rdict["hz2"].append(hz2.detach().cpu().numpy())
            if permuted_content:
                rdict["c'"].append(c_perm.detach().cpu().numpy())

    # concatenate each list of values along the batch dimension
    for k, v in rdict.items():
        if len(v) > 0:
            v = np.concatenate(v, axis=0)
        rdict[k] = np.array(v)
    return rdict


def evaluate_prediction(model, metric, X_train, y_train, X_test, y_test):
    # handle edge cases when inputs or labels are zero-dimensional
    if any([0 in x.shape for x in [X_train, y_train, X_test, y_test]]):
        return np.nan
    assert X_train.shape[1] == X_test.shape[1]
    assert y_train.shape[1] == y_test.shape[1]
    # handle edge cases when the inputs are one-dimensional
    if X_train.shape[1] == 1:
        X_train = X_train.reshape(-1, 1)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    return metric(y_test, y_pred)


def main():

    # parse args
    args, _ = parse_args()

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
        arguments_to_load = [
            "style_change_prob", "statistical_dependence",
            "content_dependent_style", "modality_n", "style_n", "content_n",
            "encoding_size", "shared_mixing", "n_mixing_layer", "shared_encoder",
            "c_param", "m_param"]
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
    np.random.seed(args.seed)
    random.seed(args.seed)
    torch.manual_seed(args.seed)

    # load training seed, which ensures consistent latent spaces for evaluation
    if args.evaluate:
        with open(os.path.join(args.save_dir, 'args.json'), 'r') as fp:
            train_seed = json.load(fp)["seed"]
        assert args.seed != train_seed
    else:
        train_seed = args.seed

    # set device
    if torch.cuda.is_available() and not args.no_cuda:
        device = "cuda"
    else:
        device = "cpu"
        warnings.warn("cuda is not available or --no-cuda was set.")

    # define loss function
    loss_func = LpSimCLRLoss()

    # shorthand notation for dimensionality
    nc, ns, nm = args.content_n, args.style_n, args.modality_n

    # define latents
    latent_spaces_list = []
    Sigma_c, Sigma_s, Sigma_a, Sigma_m1, Sigma_m2, a, B = [None] * 7
    rgen = torch.Generator(device=device)
    rgen.manual_seed(train_seed)  # ensures same latents for train and eval
    if args.statistical_dependence:
        Sigma_c = wishart.rvs(nc, np.eye(nc), size=1, random_state=train_seed)
        Sigma_s = wishart.rvs(ns, np.eye(ns), size=1, random_state=train_seed)
        Sigma_a = wishart.rvs(ns, np.eye(ns), size=1, random_state=train_seed)
        Sigma_m1 = wishart.rvs(nm, np.eye(nm), size=1, random_state=train_seed)
        Sigma_m2 = wishart.rvs(nm, np.eye(nm), size=1, random_state=train_seed)
    if args.content_dependent_style:
        B = torch.randn(ns, nc, device=device, generator=rgen)
        a = torch.randn(ns, device=device, generator=rgen)
    # content c
    space_c = NRealSpace(nc)
    sample_marginal_c = lambda space, size, device=device: \
        space.normal(None, args.m_param, size, device, Sigma=Sigma_c)
    sample_conditional_c = lambda space, z, size, device=device: z
    latent_spaces_list.append(LatentSpace(
        space=space_c,
        sample_marginal=sample_marginal_c,
        sample_conditional=sample_conditional_c))
    # style s
    space_s = NRealSpace(ns)
    sample_marginal_s = lambda space, size, device=device: \
        space.normal(None, args.m_param, size, device, Sigma=Sigma_s)
    sample_conditional_s = lambda space, z, size, device=device: \
        space.normal(z, args.c_param, size, device,
                     change_prob=args.style_change_prob, Sigma=Sigma_a)
    latent_spaces_list.append(LatentSpace(
        space=space_s,
        sample_marginal=sample_marginal_s,
        sample_conditional=sample_conditional_s))
    # modality-specific m1 and m2
    if nm > 0:
        space_m1 = NRealSpace(nm)
        sample_marginal_m1 = lambda space, size, device=device: \
            space.normal(None, args.m_param, size, device, Sigma=Sigma_m1)
        sample_conditional_m1 = lambda space, z, size, device=device: z
        latent_spaces_list.append(LatentSpace(
            space=space_m1,
            sample_marginal=sample_marginal_m1,
            sample_conditional=sample_conditional_m1))
        space_m2 = NRealSpace(nm)
        sample_marginal_m2 = lambda space, size, device=device: \
            space.normal(None, args.m_param, size, device, Sigma=Sigma_m2)
        sample_conditional_m2 = lambda space, z, size, device=device: z
        latent_spaces_list.append(LatentSpace(
            space=space_m2,
            sample_marginal=sample_marginal_m2,
            sample_conditional=sample_conditional_m2))
    # combine latents
    latent_space = ProductLatentSpace(spaces=latent_spaces_list, a=a, B=B)

    # define mixing functions
    f1 = construct_invertible_mlp(
        n=nc + ns + nm,
        n_layers=args.n_mixing_layer,
        cond_thresh_ratio=0.001,
        n_iter_cond_thresh=25000)
    f1 = f1.to(device)
    f2 = construct_invertible_mlp(
        n=nc + ns + nm,
        n_layers=args.n_mixing_layer,
        cond_thresh_ratio=0.001,
        n_iter_cond_thresh=25000)
    f2 = f2.to(device)
    # for evaluation, always load saved mixing functions
    if args.evaluate:
        f1_path = os.path.join(args.save_dir, 'f1.pt')
        f1.load_state_dict(torch.load(f1_path, map_location=device))
        f2_path = os.path.join(args.save_dir, 'f2.pt')
        f2.load_state_dict(torch.load(f2_path, map_location=device))
    # freeze parameters
    for p in f1.parameters():
        p.requires_grad = False
    for p in f2.parameters():
        p.requires_grad = False
    # save mixing functions to disk
    if args.save_dir and not args.evaluate:
        if not os.path.exists(args.save_dir):
            os.makedirs(args.save_dir)
        torch.save(f1.state_dict(), os.path.join(args.save_dir, "f1.pt"))
        torch.save(f2.state_dict(), os.path.join(args.save_dir, "f2.pt"))

    # define encoders
    g1 = encoders.get_mlp(
        n_in=nc + ns + nm,
        n_out=args.encoding_size,
        layers=[(nc + ns + nm) * 10,
                (nc + ns + nm) * 50,
                (nc + ns + nm) * 50,
                (nc + ns + nm) * 50,
                (nc + ns + nm) * 50,
                (nc + ns + nm) * 10])
    g1 = g1.to(device)
    g2 = encoders.get_mlp(
        n_in=nc + ns + nm,
        n_out=args.encoding_size,
        layers=[(nc + ns + nm) * 10,
                (nc + ns + nm) * 50,
                (nc + ns + nm) * 50,
                (nc + ns + nm) * 50,
                (nc + ns + nm) * 50,
                (nc + ns + nm) * 10])
    g2 = g2.to(device)
    # for evaluation, always load saved encoders
    if args.evaluate:
        g1_path = os.path.join(args.save_dir, 'g1.pt')
        g1.load_state_dict(torch.load(g1_path, map_location=device))
        g2_path = os.path.join(args.save_dir, 'g2.pt')
        g2.load_state_dict(torch.load(g2_path, map_location=device))

    # for convenience, define h as a composition of mixing function and encoder
    if args.shared_mixing:
        f2 = f1  # overwrites the second mixing function
    if args.shared_encoder:
        g2 = g1  # overwrites the second encoder
    h1 = lambda z: g1(f1(z))
    h2 = lambda z: g2(f2(z))

    # define optimizer
    if not args.evaluate:
        if args.shared_encoder:
            params = list(g1.parameters())
        else:
            params = list(g1.parameters()) + list(g2.parameters())
        optimizer = torch.optim.Adam(params, lr=args.lr)

    # training
    # --------
    step = 1
    while step <= args.train_steps and not args.evaluate:

        # training step
        z1, z2 = latent_space.sample_z1_and_z2(args.batch_size, device)
        z1_, z2_ = latent_space.sample_z1_and_z2(args.batch_size, device)
        data = [z1, z2, z1_, z2_]
        train_step(data, h1, h2, loss_func, optimizer, params)

        # every log_steps, we have a checkpoint and small evaluation
        if step % args.log_steps == 1 or step == args.train_steps:

            # save encoders to disk
            if args.save_dir and not args.evaluate:
                torch.save(g1.state_dict(), os.path.join(args.save_dir, "g1.pt"))
                torch.save(g2.state_dict(), os.path.join(args.save_dir, "g2.pt"))

            # lightweight evaluation with linear classifiers
            print(f"\nStep: {step} \t")
            data_dict = generate_data(latent_space, h1, h2, device, loss_func=loss_func)
            print(f"<Loss>: {np.mean(data_dict['loss_values']):.4f} \t")
            data_dict["hz1"] = StandardScaler().fit_transform(data_dict["hz1"])
            for k in ["c", "s", "s~", "m1", "m2"]:
                inputs, labels = data_dict["hz1"], data_dict[k]
                train_inputs, test_inputs, train_labels, test_labels = \
                    train_test_split(inputs, labels)
                data = [train_inputs, train_labels, test_inputs, test_labels]
                r2_linear = evaluate_prediction(
                    linear_model.LinearRegression(n_jobs=-1), r2_score, *data)
                print(f"{k} r2_linear: {r2_linear}")
        step += 1

    # evaluation
    # ----------
    if args.evaluate:

        # generate encodings and labels for the validation and test data
        val_dict = generate_data(
            latent_space, h1, h2, device,
            num_batches=args.num_eval_batches,
            loss_func=loss_func,
            permuted_content=args.permuted_content)
        test_dict = generate_data(
            latent_space, h1, h2, device,
            num_batches=args.num_eval_batches,
            loss_func=loss_func,
            permuted_content=args.permuted_content)

        # print average loss value
        print(f"<Val Loss>: {np.mean(val_dict['loss_values']):.4f} \t")
        print(f"<Test Loss>: {np.mean(test_dict['loss_values']):.4f} \t")

        # standardize the encodings
        for m in [1, 2]:
            scaler = StandardScaler()
            val_dict[f"hz{m}"] = scaler.fit_transform(val_dict[f"hz{m}"])
            test_dict[f"hz{m}"] = scaler.transform(test_dict[f"hz{m}"])

        # train predictors on data from val_dict and evaluate on test_dict
        results = []
        for m in [1, 2]:
            for k in ["c", "s", "s~", "m1", "m2", "c'"]:

                # select data
                train_inputs, test_inputs = val_dict[f"hz{m}"], test_dict[f"hz{m}"]
                train_labels, test_labels = val_dict[k], test_dict[k]
                data = [train_inputs, train_labels, test_inputs, test_labels]

                # linear regression
                r2_linear = evaluate_prediction(
                    linear_model.LinearRegression(n_jobs=-1), r2_score, *data)

                # nonlinear regression
                if args.mlp_eval:
                    model = MLPRegressor(max_iter=1000)  # lightweight option
                else:
                    # grid search is time- and memory-intensive
                    model = GridSearchCV(
                        kernel_ridge.KernelRidge(kernel='rbf', gamma=0.1),
                        param_grid={"alpha": [1e0, 0.1, 1e-2, 1e-3],
                                    "gamma": np.logspace(-2, 2, 4)},
                        cv=3, n_jobs=-1)
                r2_nonlinear = evaluate_prediction(model, r2_score, *data)

                # append results
                results.append((f"hz{m}", k, r2_linear, r2_nonlinear))

        # convert evaluation results into tabular form
        cols = ["encoding", "predicted_factors", "r2_linear", "r2_nonlinear"]
        df_results = pd.DataFrame(results, columns=cols)
        df_results.to_csv(os.path.join(args.save_dir, "results.csv"))
        print("Regression results:")
        print(df_results.to_string())


if __name__ == "__main__":
    main()
