import logging
import numpy as np
import pickle
import pandas as pd
from pathlib import Path
from custom_hmm import HMM
from hmmlearn_hmm import HMMLearnModel
from typing import List, Literal, Dict, Union
from mfcc_extract import load_mfccs, load_mfccs_by_word
from matplotlib import pyplot as plt
from decoder import Decoder

logging.basicConfig(level=logging.INFO)
logging.getLogger("matplotlib").setLevel(logging.WARNING)


def plot_training_progress(all_likelihoods: Dict[str, List[float]], implementation: str) -> None:
    num_models = len(all_likelihoods)
    n_cols = 4
    n_rows = (num_models + n_cols - 1) // n_cols

    plt.figure(figsize=(15, 3 * n_rows))

    for idx, (word, log_likelihoods) in enumerate(all_likelihoods.items(), 1):
        plt.subplot(n_rows, n_cols, idx)
        iterations = range(len(log_likelihoods))

        plt.plot(iterations, log_likelihoods, "b-", linewidth=2)
        plt.plot(iterations, log_likelihoods, "bo", markersize=4)

        total_improvement = log_likelihoods[-1] - log_likelihoods[0]

        plt.xlabel("Iteration", fontsize=10)
        plt.ylabel("Log Likelihood", fontsize=10)
        plt.title(f"{word}\nImprovement: {total_improvement:.2f}", fontsize=12)
        plt.grid(True, linestyle="--", alpha=0.7)

        print(f"\nTraining Summary for {word}:")
        print(f"Initial log likelihood: {log_likelihoods[0]:.2f}")
        print(f"Final log likelihood: {log_likelihoods[-1]:.2f}")
        print(f"Total improvement: {total_improvement:.2f}")

    plt.tight_layout()
    
    # Save the training progress plot
    figures_dir = Path("figures")
    figures_dir.mkdir(exist_ok=True)
    plt.savefig(figures_dir / f"training_progress_{implementation}.png")
    plt.close()


def pretty_print_matrix(matrix: np.ndarray, precision: int = 3) -> None:
    n = matrix.shape[0]
    df = pd.DataFrame(
        matrix,
        columns=[
            f"S{i}" if i != 0 and i != n - 1 else ("Entry" if i == 0 else "Exit")
            for i in range(n)
        ],
        index=[
            f"S{i}" if i != 0 and i != n - 1 else ("Entry" if i == 0 else "Exit")
            for i in range(n)
        ],
    )

    row_sums = df.sum(axis=1).round(precision)
    df = df.replace(0, ".")

    print("\nTransition Matrix:")
    print("==================")
    print(df.round(precision))
    assert np.allclose(row_sums, 1.0), "Row sums should be equal to 1.0"


def save_model(model, model_path: Path) -> None:
    model_path.parent.mkdir(parents=True, exist_ok=True)
    with open(model_path, "wb") as f:
        pickle.dump(model, f)
    logging.info(f"Saved model to {model_path}")



# def train2_hmm():
#     vocabs = [
#         "heed",
#         #"hid",
#         #"head",
#         #"had",
#         #"hard",
#         #"hud",
#         #"hod",
#         #"hoard",
#         #"hood",
#         #"whod",
#         #"heard",
#     ]

#     feature_set = load_mfccs("feature_set")
#     features = {word: load_mfccs_by_word("feature_set", word) for word in vocabs}
#     # total_features_length = sum(len(features[word]) for word in vocabs)
#     # assert total_features_length == len(feature_set)
    
#     hmms = {word: HMM(8, 13, feature_set, model_name=word) for word in vocabs}
    
#     model = hmms["heed"]
#     testing_feature = features["heed"]

#     new_model = baum_welch(testing_feature,model)
#     HMM.print_parameters(hmms["heed"])

#     #print(viterbi_algorithm(observations,model.A,model.B))
#     HMM.print_parameters(new_model)
    


def train_hmm(
    implementation: Literal["custom", "hmmlearn"] = "hmmlearn",
    num_states: int = 8,
    num_features: int = 13,
    n_iter: int = 15,
    min_covar: float = 0.01,
    var_floor_factor: float = 0.001
) -> Dict[str, Union[HMM, HMMLearnModel]]:
    vocabs = [
        "heed", "hid", "head", "had", "hard", "hud", 
        "hod", "hoard", "hood", "whod", "heard"
    ]

    models_dir = Path("trained_models")
    impl_dir = models_dir / implementation
    impl_dir.mkdir(parents=True, exist_ok=True)

    feature_set = load_mfccs("feature_set")
    features = {word: load_mfccs_by_word("feature_set", word) for word in vocabs}
    total_features_length = sum(len(features[word]) for word in vocabs)
    assert total_features_length == len(feature_set)

    hmms = {}
    training_histories = {}

    for word in vocabs:
        logging.info(f"\nTraining model for word: {word}")
        model_path = impl_dir / f"{word}_{implementation}_{n_iter}.pkl"

        if implementation == "custom":
            hmm = HMM(num_states, num_features, feature_set, model_name=word, var_floor_factor=var_floor_factor)
            log_likelihoods = hmm.baum_welch(features[word], n_iter)
            trained_model = hmm
        elif implementation == "hmmlearn":
            hmm = HMMLearnModel(num_states=num_states, model_name=word, n_iter=n_iter, min_covar=min_covar)
            trained_model, _ = hmm.fit(features[word])
            log_likelihoods = hmm.model.monitor_.history

        training_histories[word] = log_likelihoods
        save_model(trained_model, model_path)
        hmms[word] = hmm

    plot_training_progress(training_histories, implementation)
    return hmms






def train_hmm_for_each_epoch(
    implementation: Literal["custom", "hmmlearn"] = "hmmlearn",
    num_states: int = 8,
    num_features: int = 13,
    n_iter: int = 15,
    min_covar: float = 0.01,
    var_floor_factor: float = 0.001
) -> Dict[str, Union[HMM, HMMLearnModel]]:
    vocabs = [
        "heed", "hid", "head", "had", "hard", "hud", 
        "hod", "hoard", "hood", "whod", "heard"
    ]
    epochs = n_iter
    models_dir = Path("trained_models")
    impl_dir = models_dir / implementation
    impl_dir.mkdir(parents=True, exist_ok=True)

    feature_set = load_mfccs("feature_set")
    features = {word: load_mfccs_by_word("feature_set", word) for word in vocabs}
    eval_features = {word: load_mfccs_by_word("eval_feature_set", word) for word in vocabs}
    total_features_length = sum(len(features[word]) for word in vocabs)
    assert total_features_length == len(feature_set)

    hmms = {}
    training_histories = {}
    for epoch in range(epochs+1):
        print(f"Epoch {epoch + 1}")
        for word in vocabs:
            logging.info(f"\nTraining model for word: {word}")
            model_path = impl_dir / f"{word}_{implementation}_epoch_{epoch}.pkl"
            # not implemented
            if implementation == "custom":
                hmm = HMM(num_states, num_features, feature_set, model_name=word, var_floor_factor=var_floor_factor)
                log_likelihoods = hmm.baum_welch(features[word], epoch)
                trained_model = hmm
            elif implementation == "hmmlearn":
                hmm = HMMLearnModel(num_states=num_states, model_name=word, n_iter=epoch, min_covar=min_covar)
                trained_model, _ = hmm.fit(features[word])
                log_likelihoods = hmm.model.monitor_.history

            training_histories[word] = log_likelihoods
            save_model(trained_model, model_path)
            hmms[word] = hmm

        #plot_training_progress(training_histories, implementation)
    return hmms






if __name__ == "__main__":
    print("\nTraining with `hmmlearn` implementation:")
    num_states = 8
    num_features = 13
    n_iter = 15
    min_covar = 0.1
    print(f"Number of states: {num_states} | Number of features: {num_features}" f" | Number of iterations: {n_iter} | Minimum covariance: {min_covar}")
    #train_hmm("hmmlearn", num_states, num_features, n_iter, min_covar)
    train_hmm_for_each_epoch("hmmlearn", num_states, num_features, n_iter, min_covar)
    
    # print("\nTraining with `custom` implementation:")
    # num_states = 8
    # num_features = 13
    # n_iter = 15
    # var_floor_factor = 0.01
    # print(f"Number of states: {num_states} | Number of features: {num_features}" f" | Number of iterations: {n_iter} | Variance floor factor: {var_floor_factor}")
    # train_hmm("custom", num_states, num_features, n_iter, var_floor_factor)