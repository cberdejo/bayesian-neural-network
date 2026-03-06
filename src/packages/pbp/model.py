from __future__ import annotations

import math
import sys

import numpy as np
import pytensor
import pytensor.tensor as T

from .config import PBPConfig


def _to_numpy_2d(x, *, dtype: np.dtype | None = None) -> np.ndarray:
    x = np.asarray(x)
    if x.ndim == 1:
        x = x.reshape(-1, 1)
    if dtype is not None and x.dtype != dtype:
        x = x.astype(dtype, copy=False)
    return x


class Prior:
    def __init__(self, layer_sizes, var_targets):
        n_samples = 3.0
        v_observed = 1.0
        self.a_w = 2.0 * n_samples
        self.b_w = 2.0 * n_samples * v_observed

        a_sigma = 2.0 * n_samples
        b_sigma = 2.0 * n_samples * var_targets

        self.a_sigma_hat_nat = a_sigma - 1
        self.b_sigma_hat_nat = -b_sigma

        self.rnd_m_w: list[np.ndarray] = []
        self.m_w_hat_nat: list[np.ndarray] = []
        self.v_w_hat_nat: list[np.ndarray] = []
        self.a_w_hat_nat: list[np.ndarray] = []
        self.b_w_hat_nat: list[np.ndarray] = []
        for size_out, size_in in zip(layer_sizes[1:], layer_sizes[:-1]):
            self.rnd_m_w.append(
                1.0 / np.sqrt(size_in + 1) * np.random.randn(size_out, size_in + 1)
            )
            self.m_w_hat_nat.append(np.zeros((size_out, size_in + 1)))
            self.v_w_hat_nat.append(
                (self.a_w - 1) / self.b_w * np.ones((size_out, size_in + 1))
            )
            self.a_w_hat_nat.append(np.zeros((size_out, size_in + 1)))
            self.b_w_hat_nat.append(np.zeros((size_out, size_in + 1)))

    def get_initial_params(self):
        m_w = [arr.copy() for arr in self.rnd_m_w]
        v_w = [1.0 / arr for arr in self.v_w_hat_nat]
        return {
            "m_w": m_w,
            "v_w": v_w,
            "a": self.a_sigma_hat_nat + 1,
            "b": -self.b_sigma_hat_nat,
        }

    def get_params(self):
        m_w = [self.m_w_hat_nat[i] / self.v_w_hat_nat[i] for i in range(len(self.rnd_m_w))]
        v_w = [1.0 / arr for arr in self.v_w_hat_nat]
        return {
            "m_w": m_w,
            "v_w": v_w,
            "a": self.a_sigma_hat_nat + 1,
            "b": -self.b_sigma_hat_nat,
        }

    def refine_prior(self, params):
        for i in range(len(params["m_w"])):
            for j in range(params["m_w"][i].shape[0]):
                for k in range(params["m_w"][i].shape[1]):
                    v_w_nat = 1.0 / params["v_w"][i][j, k]
                    m_w_nat = params["m_w"][i][j, k] / params["v_w"][i][j, k]
                    v_w_cav_nat = v_w_nat - self.v_w_hat_nat[i][j, k]
                    m_w_cav_nat = m_w_nat - self.m_w_hat_nat[i][j, k]
                    v_w_cav = 1.0 / v_w_cav_nat
                    m_w_cav = m_w_cav_nat / v_w_cav_nat
                    a_w_nat = self.a_w - 1
                    b_w_nat = -self.b_w
                    a_w_cav_nat = a_w_nat - self.a_w_hat_nat[i][j, k]
                    b_w_cav_nat = b_w_nat - self.b_w_hat_nat[i][j, k]
                    a_w_cav = a_w_cav_nat + 1
                    b_w_cav = -b_w_cav_nat

                    if v_w_cav > 0 and b_w_cav > 0 and a_w_cav > 1 and v_w_cav < 1e6:
                        v = v_w_cav + b_w_cav / (a_w_cav - 1)
                        v1 = v_w_cav + b_w_cav / a_w_cav
                        v2 = v_w_cav + b_w_cav / (a_w_cav + 1)
                        logZ = -0.5 * np.log(v) - 0.5 * m_w_cav**2 / v
                        logZ1 = -0.5 * np.log(v1) - 0.5 * m_w_cav**2 / v1
                        logZ2 = -0.5 * np.log(v2) - 0.5 * m_w_cav**2 / v2
                        d_logZ_d_m = -m_w_cav / v
                        d_logZ_d_v = -0.5 / v + 0.5 * m_w_cav**2 / v**2
                        m_w_new = m_w_cav + v_w_cav * d_logZ_d_m
                        v_w_new = v_w_cav - v_w_cav**2 * (d_logZ_d_m**2 - 2 * d_logZ_d_v)
                        a_w_new = 1.0 / (
                            np.exp(logZ2 - 2 * logZ1 + logZ) * (a_w_cav + 1) / a_w_cav - 1.0
                        )
                        b_w_new = 1.0 / (
                            np.exp(logZ2 - logZ1) * (a_w_cav + 1) / b_w_cav
                            - np.exp(logZ1 - logZ) * a_w_cav / b_w_cav
                        )
                        v_w_new_nat = 1.0 / v_w_new
                        m_w_new_nat = m_w_new / v_w_new
                        a_w_new_nat = a_w_new - 1
                        b_w_new_nat = -b_w_new

                        self.m_w_hat_nat[i][j, k] = m_w_new_nat - m_w_cav_nat
                        self.v_w_hat_nat[i][j, k] = v_w_new_nat - v_w_cav_nat
                        self.a_w_hat_nat[i][j, k] = a_w_new_nat - a_w_cav_nat
                        self.b_w_hat_nat[i][j, k] = b_w_new_nat - b_w_cav_nat

                        params["m_w"][i][j, k] = m_w_new
                        params["v_w"][i][j, k] = v_w_new

                        self.a_w = a_w_new
                        self.b_w = b_w_new

        return params


class Network_layer:
    def __init__(self, m_w_init, v_w_init, non_linear=True):
        self.m_w = pytensor.shared(value=m_w_init.astype(pytensor.config.floatX), name="m_w", borrow=True)
        self.v_w = pytensor.shared(value=v_w_init.astype(pytensor.config.floatX), name="v_w", borrow=True)
        self.w = pytensor.shared(value=m_w_init.astype(pytensor.config.floatX), name="w", borrow=True)
        self.non_linear = non_linear
        self.n_inputs = pytensor.shared(float(m_w_init.shape[1]))

    @staticmethod
    def n_pdf(x):
        return 1.0 / T.sqrt(2 * math.pi) * T.exp(-0.5 * x**2)

    @staticmethod
    def n_cdf(x):
        return 0.5 * (1.0 + T.erf(x / T.sqrt(2.0)))

    @staticmethod
    def gamma(x):
        return Network_layer.n_pdf(x) / Network_layer.n_cdf(-x)

    @staticmethod
    def beta(x):
        return Network_layer.gamma(x) * (Network_layer.gamma(x) - x)

    def output_probabilistic(self, m_w_previous, v_w_previous):
        m_in = T.concatenate([m_w_previous, T.alloc(1, 1)], 0)
        v_in = T.concatenate([v_w_previous, T.alloc(0, 1)], 0)

        m_linear = T.dot(self.m_w, m_in) / T.sqrt(self.n_inputs)
        v_linear = (
            T.dot(self.v_w, v_in) + T.dot(self.m_w**2, v_in) + T.dot(self.v_w, m_in**2)
        ) / self.n_inputs

        if self.non_linear:
            alpha = m_linear / T.sqrt(v_linear)
            gamma_val = Network_layer.gamma(-alpha)
            gamma_robust = -alpha - 1.0 / alpha + 2.0 / alpha**3
            gamma_final = T.switch(T.lt(-alpha, T.fill(alpha, 30)), gamma_val, gamma_robust)
            v_aux = m_linear + T.sqrt(v_linear) * gamma_final
            m_a = Network_layer.n_cdf(alpha) * v_aux
            v_a = (
                m_a * v_aux * Network_layer.n_cdf(-alpha)
                + Network_layer.n_cdf(alpha) * v_linear * (1 - gamma_final * (gamma_final + alpha))
            )
            return m_a, v_a
        return m_linear, v_linear

    def output_deterministic(self, output_previous):
        out = T.concatenate([output_previous, T.alloc(1, 1)], 0) / T.sqrt(self.n_inputs)
        a = T.dot(self.w, out)
        if self.non_linear:
            a = T.switch(T.lt(a, T.fill(a, 0)), T.fill(a, 0), a)
        return a


class Network:
    def __init__(self, m_w_init, v_w_init, a_init, b_init):
        self.layers: list[Network_layer] = []
        if len(m_w_init) > 1:
            for m_w, v_w in zip(m_w_init[:-1], v_w_init[:-1]):
                self.layers.append(Network_layer(m_w, v_w, True))
        self.layers.append(Network_layer(m_w_init[-1], v_w_init[-1], False))
        self.params_m_w = [layer.m_w for layer in self.layers]
        self.params_v_w = [layer.v_w for layer in self.layers]
        self.params_w = [layer.w for layer in self.layers]
        self.a = pytensor.shared(float(a_init))
        self.b = pytensor.shared(float(b_init))

    def output_deterministic(self, x):
        for layer in self.layers:
            x = layer.output_deterministic(x)
        return x

    def output_probabilistic(self, m):
        v = T.zeros_like(m)
        for layer in self.layers:
            m, v = layer.output_probabilistic(m, v)
        return m[0], v[0]

    def logZ_Z1_Z2(self, x, y):
        m, v = self.output_probabilistic(x)
        v_final = v + self.b / (self.a - 1)
        v_final1 = v + self.b / self.a
        v_final2 = v + self.b / (self.a + 1)
        logZ = -0.5 * (T.log(v_final) + (y - m) ** 2 / v_final)
        logZ1 = -0.5 * (T.log(v_final1) + (y - m) ** 2 / v_final1)
        logZ2 = -0.5 * (T.log(v_final2) + (y - m) ** 2 / v_final2)
        return logZ, logZ1, logZ2

    def generate_updates(self, logZ, logZ1, logZ2):
        updates = []
        for i in range(len(self.params_m_w)):
            updates.append((
                self.params_m_w[i],
                self.params_m_w[i] + self.params_v_w[i] * T.grad(logZ, self.params_m_w[i]),
            ))
            updates.append((
                self.params_v_w[i],
                self.params_v_w[i]
                - self.params_v_w[i] ** 2 * (T.grad(logZ, self.params_m_w[i]) ** 2 - 2 * T.grad(logZ, self.params_v_w[i])),
            ))
        updates.append((self.a, 1.0 / (T.exp(logZ2 - 2 * logZ1 + logZ) * (self.a + 1) / self.a - 1.0)))
        updates.append((self.b, 1.0 / (T.exp(logZ2 - logZ1) * (self.a + 1) / self.b - T.exp(logZ1 - logZ) * self.a / self.b)))
        return updates

    def get_params(self):
        return {
            "m_w": [layer.m_w.get_value() for layer in self.layers],
            "v_w": [layer.v_w.get_value() for layer in self.layers],
            "a": self.a.get_value(),
            "b": self.b.get_value(),
        }

    def set_params(self, params):
        for i in range(len(self.layers)):
            self.layers[i].m_w.set_value(params["m_w"][i])
            self.layers[i].v_w.set_value(params["v_w"][i])
        self.a.set_value(params["a"])
        self.b.set_value(params["b"])

    def remove_invalid_updates(self, new_params, old_params):
        for i in range(len(self.layers)):
            idx1 = np.where(new_params["v_w"][i] <= 1e-100)
            idx2 = np.where(np.logical_or(np.isnan(new_params["m_w"][i]), np.isnan(new_params["v_w"][i])))
            index = [np.concatenate((idx1[0], idx2[0])), np.concatenate((idx1[1], idx2[1]))]
            if len(index[0]) > 0:
                new_params["m_w"][i][index] = old_params["m_w"][i][index]
                new_params["v_w"][i][index] = old_params["v_w"][i][index]

    def sample_w(self):
        w = []
        for i in range(len(self.layers)):
            w.append(
                self.params_m_w[i].get_value()
                + np.random.randn(*self.params_m_w[i].get_value().shape) * np.sqrt(self.params_v_w[i].get_value())
            )
        for i in range(len(self.layers)):
            self.params_w[i].set_value(w[i])


class PBP:
    def __init__(self, layer_sizes, mean_y_train, std_y_train):
        self.std_y_train = std_y_train
        self.mean_y_train = mean_y_train
        self.prior = Prior(layer_sizes, var_targets=1)
        params = self.prior.get_initial_params()
        self.network = Network(params["m_w"], params["v_w"], params["a"], params["b"])

        self.x = T.vector("x")
        self.y = T.scalar("y")
        self.logZ, self.logZ1, self.logZ2 = self.network.logZ_Z1_Z2(self.x, self.y)

        self.adf_update = pytensor.function(
            [self.x, self.y],
            self.logZ,
            updates=self.network.generate_updates(self.logZ, self.logZ1, self.logZ2),
        )
        self.predict_probabilistic = pytensor.function([self.x], self.network.output_probabilistic(self.x))
        self.predict_deterministic = pytensor.function([self.x], self.network.output_deterministic(self.x))

    def do_pbp(self, X_train, y_train, n_iterations):
        y_train = np.asarray(y_train).reshape(-1)
        if n_iterations > 0:
            self.do_first_pass(X_train, y_train)
            params = self.network.get_params()
            params = self.prior.refine_prior(params)
            self.network.set_params(params)
            sys.stdout.write("{}\n".format(0))
            sys.stdout.flush()
            for i in range(int(n_iterations) - 1):
                self.prior.get_params()
                self.do_first_pass(X_train, y_train)
                params = self.network.get_params()
                params = self.prior.refine_prior(params)
                self.network.set_params(params)
                sys.stdout.write("{}\n".format(i + 1))
                sys.stdout.flush()

    def get_deterministic_output(self, X_test):
        output = np.zeros(X_test.shape[0])
        for i in range(X_test.shape[0]):
            output[i] = self.predict_deterministic(X_test[i, :])
            output[i] = output[i] * self.std_y_train + self.mean_y_train
        return output

    def get_predictive_mean_and_variance(self, X_test):
        mean = np.zeros(X_test.shape[0])
        variance = np.zeros(X_test.shape[0])
        for i in range(X_test.shape[0]):
            m, v = self.predict_probabilistic(X_test[i, :])
            m = m * self.std_y_train + self.mean_y_train
            v = v * self.std_y_train**2
            mean[i] = m
            variance[i] = v
        v_noise = self.network.b.get_value() / (self.network.a.get_value() - 1) * self.std_y_train**2
        return mean, variance, v_noise

    def do_first_pass(self, X, y):
        permutation = np.random.permutation(X.shape[0])
        counter = 0
        for i in permutation:
            old_params = self.network.get_params()
            self.adf_update(X[i, :], y[i])
            new_params = self.network.get_params()
            self.network.remove_invalid_updates(new_params, old_params)
            self.network.set_params(new_params)
            if counter % 1000 == 0:
                sys.stdout.write(".")
                sys.stdout.flush()
            counter += 1
        sys.stdout.write("\n")
        sys.stdout.flush()

    def sample_w(self):
        self.network.sample_w()


class PBP_net:
    """High-level wrapper: handles normalization, training, and prediction."""

    def __init__(self, X_train, y_train, n_hidden, n_epochs=40, normalize=False):
        if normalize:
            self.std_X_train = np.std(X_train, 0)
            self.std_X_train[self.std_X_train == 0] = 1
            self.mean_X_train = np.mean(X_train, 0)
        else:
            self.std_X_train = np.ones(X_train.shape[1])
            self.mean_X_train = np.zeros(X_train.shape[1])

        X_train = (X_train - self.mean_X_train) / self.std_X_train
        y_train = np.asarray(y_train).reshape(-1)
        self.mean_y_train = np.mean(y_train)
        self.std_y_train = np.std(y_train)
        if self.std_y_train == 0:
            self.std_y_train = 1

        y_train_normalized = (y_train - self.mean_y_train) / self.std_y_train
        n_units_per_layer = np.concatenate(([X_train.shape[1]], n_hidden, [1]))
        self.pbp_instance = PBP(n_units_per_layer, self.mean_y_train, self.std_y_train)
        self.pbp_instance.do_pbp(X_train, y_train_normalized, n_epochs)

    @classmethod
    def from_config(cls, X_train: np.ndarray, y_train: np.ndarray, cfg: PBPConfig) -> "PBP_net":
        if cfg.seed is not None:
            np.random.seed(cfg.seed)
        X_train = _to_numpy_2d(X_train)
        y_train = np.asarray(y_train).reshape(-1)
        neurons = cfg.layer_sizes
        hidden = neurons[1:-1]
        return cls(X_train, y_train, hidden, n_epochs=cfg.n_epochs, normalize=cfg.normalize)

    def predict(self, X_test):
        X_test = _to_numpy_2d(X_test)
        X_test = (X_test - self.mean_X_train) / self.std_X_train
        m, v, v_noise = self.pbp_instance.get_predictive_mean_and_variance(X_test)
        return m, v, v_noise

    def predict_deterministic(self, X_test):
        X_test = np.array(X_test, ndmin=2)
        X_test = (X_test - self.mean_X_train) / self.std_X_train
        return self.pbp_instance.get_deterministic_output(X_test)

    def sample_weights(self):
        self.pbp_instance.sample_w()
