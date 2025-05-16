import numpy as np
import pandas as pd
from mfcc_extract import load_mfccs_by_word
from mfcc_extract import load_mfccs

class HMM:
    def __init__(
        self,
        num_states: int,
        num_obs: int,
        feature_set: list[np.ndarray] = None,
        model_name: str = None,
        var_floor_factor: float = 0.001,
    ):
        assert num_states > 0, "Number of states must be greater than 0."
        assert num_obs > 0, "Number of observations must be greater than 0."
        self.model_name = model_name
        self.num_states = num_states
        self.num_obs = num_obs
        self.var_floor_factor = var_floor_factor
        self.total_states = num_states + 2

        self.pi = np.zeros(self.total_states)
        self.pi[0] = 1.0

        if feature_set is not None:
            assert all(
                feature.shape[0] == num_obs for feature in feature_set
            ), "All features must have the same dimension as the number of observations."
            self.init_parameters(feature_set)

    def init_parameters(self, feature_set: list[np.ndarray]) -> None:
        self.global_mean = self.calculate_means(feature_set)

        # Calculate full covariance then zero out off-diagonal elements
        self.global_covariance = self.calculate_covariance(
            feature_set, self.global_mean
        )
        self.global_covariance *= np.eye(self.num_obs)  # Zero out off-diagonal elements

        # Apply variance floor to diagonal
        var_floor = self.var_floor_factor * np.mean(np.diag(self.global_covariance))
        np.fill_diagonal(
            self.global_covariance,
            np.maximum(np.diag(self.global_covariance), var_floor),
        )

        self.A = self.initialize_transitions(feature_set, self.num_states)

        # Initialize B using global statistics
        means = np.tile(self.global_mean, (self.total_states, 1 )).T

        # Initialize covariance matrices for each state
        covars = np.zeros((self.total_states, self.num_obs, self.num_obs))
        for i in range(self.total_states):
            covars[i] = self.global_covariance.copy()

        self.B = {"mean": means, "covariance": covars}

        assert self.B["mean"].shape == (self.num_obs,self.total_states)
        assert self.B["covariance"].shape == (
            self.total_states,
            self.num_obs,
            self.num_obs,
        )

    def calculate_means(self, feature_set: list[np.ndarray]) -> np.ndarray:
        sum = np.zeros(self.num_obs)
        count = 0
        for feature in feature_set:
            sum += np.sum(feature, axis=1)
            count += feature.shape[1]
        mean = sum / count
        return mean

    def calculate_covariance(
        self, feature_set: list[np.ndarray], mean: np.ndarray
    ) -> np.ndarray:
        covariance = np.zeros((self.num_obs, self.num_obs))
        count = 0
        for feature in feature_set:
            centered = feature - mean[:, np.newaxis]
            covariance += centered @ centered.T
            count += feature.shape[1]
        return covariance / count

    def initialize_transitions(
        self, feature_set: list[np.ndarray], num_states: int
    ) -> np.ndarray:
        total_frames = sum(feature.shape[1] for feature in feature_set)
        avg_frames_per_state = total_frames / (len(feature_set) * num_states)

        # self-loop probability
        aii = np.exp(-1 / (avg_frames_per_state - 1))
        aij = 1 - aii

        # Create transition matrix (including entry and exit states)
        total_states = num_states + 2
        A = np.zeros((total_states, total_states))

        # Entry state (index 0)
        A[0, 1] = 1.0

        for i in range(1, num_states + 1):
            A[i, i] = aii
            A[i, i + 1] = aij

        A[-1, -1] = 1.0  # set exit state self-loop for easier computation
        return A





    
   
def multivariate_gaussian(x, mean, covariance):
    K = len(mean)  # Dimensionality
    diff = x - mean
    # Compute probability density
    prob = (1 / np.sqrt((2 * np.pi) ** K * np.linalg.det(covariance))) * \
            np.exp(-(np.linalg.solve(covariance, diff).T.dot(diff)) / 2)
    return prob



def forward_algorithm(observations, A, B):
    num_states = A.shape[0] - 2
    T = observations.shape[1]
    alpha = np.zeros((T, num_states))
    pi = A[0,1:-1]
    
    # Initialize
    means = B["mean"]
    covariances = B["covariance"]
    scale_factor = 1e24
    alpha[0, :] = pi * multivariate_gaussian(observations[:, 0], means[:, 0], covariances[0])
    # Recursion
    for t in range(1, T):
        for j in range(0, num_states):

            for k in range(0,num_states):
                alpha[t,j] += alpha[t-1,k] * A[k+1,j+1]
            alpha[t,j] *= multivariate_gaussian(observations[:, t], means[:,j], covariances[j])*scale_factor

    return alpha


def backward_algorithm(observations, A, B):
    num_states = A.shape[0] - 2
    T = observations.shape[1]
    beta = np.zeros((T, num_states))
    means = B["mean"]
    covariances = B["covariance"]
    scale_factor = 1e24
    # Initialize
    beta[T-1,:] = np.transpose(A[1:-1,-1])
    # Recursion
    for t in range(T - 2, -1, -1):
        for i in range(0, num_states):

            for j in range(0,num_states):
                beta[t,i] += A[i+1,j+1] * beta[t+1,j] * multivariate_gaussian(observations[:, t+1], means[:,j], covariances[j])*scale_factor


    return beta


def baum_welch(self,observations_list):
    A = self.A
    B = self.B
    means = B["mean"]
    covariances = B["covariance"]
    scale_factor = 1e24
    num_states = A.shape[0] - 2
    K = observations_list[0].shape[0]  # Dimensionality of feature space


    # Initialize accumulators for re-estimation
    A_accum = np.zeros_like(A)
    means_accum = np.zeros_like(means)
    covariances_accum = np.zeros_like(covariances)
    gamma_sum_accum = np.zeros(num_states)

    for observations in observations_list:
        T = observations.shape[1]  # Number of time steps

        # Forward and backward probabilities
        alpha = forward_algorithm(observations, A, B)
        beta = backward_algorithm(observations, A, B)

        # Calculate gamma (state occupancy probabilities)
        gamma = np.zeros((T, num_states))
        for t in range(T):
            gamma[t, :] = alpha[t, :] * beta[t, :]
            gamma[t, :] /= np.sum(gamma[t, :])  # Normalize

        # Accumulate gamma sums for re-estimation(capital gamma)
        gamma_sum_accum += np.sum(gamma, axis=0)

        # Calculate xi (state transition probabilities)
        xi = np.zeros((T - 1, num_states, num_states))
        for t in range(T - 1):
            for i in range(num_states):
                for j in range(num_states):
                    xi[t, i, j] = alpha[t-1, i] * A[i + 1, j + 1] * \
                                    multivariate_gaussian(observations[:, t], means[:,j], covariances[j])*scale_factor * \
                                    beta[t, j]
            xi[t, :, :] /= np.sum(xi[t, :, :])  # Normalize

        # Accumulate transition matrix updates(Xij)
        for i in range(num_states):
            for j in range(num_states):
                A_accum[i + 1, j + 1] += np.sum(xi[:, i, j])

        # Accumulate mean and covariance updates
        for j in range(num_states):
            gamma_sum = np.sum(gamma[:, j])

            weighted_sum = np.sum(gamma[:, j][:, np.newaxis] * observations.T, axis=0)

            means_accum[:,j] += weighted_sum
            diff = observations.T - means[:,j]
            
            covariances_accum[j] += np.sum(
                gamma[:, j][:, np.newaxis, np.newaxis] *
                np.einsum('ij,ik->ijk', diff, diff),
                axis=0
            )

    # Normalize transition matrix
    for i in range(num_states):
        A_accum[i + 1, 1:-1] /= np.sum(A_accum[i + 1, 1:-1])

    # Normalize means and covariances
    for j in range(num_states):
        means[j] = means_accum[j] / gamma_sum_accum[j]
        covariances[j] = covariances_accum[j] / gamma_sum_accum[j]

    A = A_accum  # Update transition matrix

    # Return the updated model
    model.A = A
    model.B["mean"] = means
    model.B["covariance"] = covariances
    return model



def viterbi_algorithm(observations, A, B):
    num_states = A.shape[0] - 2  # Exclude start and end states
    T = observations.shape[1]  # Number of time steps
    delta = np.zeros((T, num_states))  # Maximum cumulative likelihoods
    psi = np.zeros((T, num_states), dtype=int)  # Backpointer table

    pi = A[0, 1:-1]  # Initial state probabilities
    means = B["mean"]
    covariances = B["covariance"]
    scale_factor = 1e24

    # Initialize
    delta[0, :] = pi * multivariate_gaussian(observations[:, 0], means[:, 0], covariances[0]) * scale_factor
    psi[0, :] = 0

    # Recursion
    for t in range(1, T):
        for j in range(num_states):
            max_value = -np.inf
            max_state = -1
            for i in range(num_states):
                value = delta[t - 1, i] * A[i + 1, j + 1]
                if value > max_value:
                    max_value = value
                    max_state = i
            delta[t, j] = max_value * multivariate_gaussian(observations[:, t], means[:, j], covariances[j]) * scale_factor
            psi[t, j] = max_state

    # Finalize
    P_star = delta[-1, -1]*A[-2, -1]


    return  P_star


if __name__ == "__main__":
    # Inline tests
    vocabs = [
        "heed",
        #"hid",
        #"head",
        #"had",
        #"hard",
        #"hud",
        #"hod",
        #"hoard",
        #"hood",
        #"whod",
        #"heard",
    ]

    feature_set = load_mfccs("feature_set")
    features = {word: load_mfccs_by_word("feature_set", word) for word in vocabs}
    # total_features_length = sum(len(features[word]) for word in vocabs)
    # assert total_features_length == len(feature_set)
    
    hmms = {word: HMM(8, 13, feature_set, model_name=word) for word in vocabs}
    
    model = hmms["heed"]
    testing_feature = features["heed"]
    observations = testing_feature[7]

    #print(model.B["covariance"][0].shape)
    new_alpha = forward_algorithm(observations,model.A,model.B)
    new_beta = backward_algorithm(observations,model.A,model.B)
    
    # Test for forward and backward compatible
    print(multivariate_gaussian(observations[:, 0], model.B["mean"][:,0], model.B["covariance"][0]) * new_beta[0, 0])
    print(model.A[-2, -1] * new_alpha[-1, 7])
    print(hmms["heed"].A)
    new_model = baum_welch(testing_feature,model)
    print(hmms["heed"].A)
    print(new_model.A)
    

