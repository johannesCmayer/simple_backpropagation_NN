import numpy as np
import random
import time


def hinge_loss(pred, y_one_hot):
    if len(pred) != len(y_one_hot):
        raise Exception('Labels not same size as predicitons.')
    if len(pred) < 2 or len(y_one_hot) < 2:
        raise Exception('This hinge loss needs at least 2 classes, or else it is always 0.')
    correct_class = np.argmax(y_one_hot)
    value_correct_class = pred[correct_class]
    loss = np.maximum(0, 1 + pred - value_correct_class)
    loss[correct_class] = 0
    return np.sum(loss)


def mse_loss(pred, y_one_hot):
    return np.sum(np.square(y_one_hot - pred))


def l2_norm(x):
    return np.sqrt(np.sum(np.square(x)))


class LinearClassifier():
    def __init__(self, input_size, output_size):
        self.weights = np.random.rand(output_size, input_size)

    def eval(self, inputs, overwrite_weights=None):
        active_weights = self.weights
        if overwrite_weights is not None:
            active_weights = overwrite_weights
        return np.matmul(active_weights, inputs)

    def outputs_to_probabilities(self, outputs):
        squashed_result = outputs / np.amax(outputs)
        unnormal_probabilities = np.exp(squashed_result)
        if unnormal_probabilities[unnormal_probabilities == np.inf].any():
            raise Exception(
                'There are infinity values in result after exponentiation. Weights might be initialized to high. Value is {}'
                .format(unnormal_probabilities))
        probabilities = unnormal_probabilities / sum(unnormal_probabilities)
        return probabilities

    def batch_eval(self, inputs, overwrite_weights=None):
        return [self.eval(i, overwrite_weights=overwrite_weights) for i in inputs]

    def loss(self, output, one_hot_target, loss_func=mse_loss, regularization=None, regularization_coef=0.1):
        loss = loss_func(output, one_hot_target)
        if regularization:
            loss += regularization(self.weights) / regularization_coef
        return loss

    def loss_vrt(self, outputs, one_hot_targets, loss_func=mse_loss):
        cost = 0
        for i, t in zip(outputs, one_hot_targets):
            cost += self.loss(i, t, loss_func=loss_func)
        return cost / len(outputs)

    def gradient_vrt_DEPRECATED(self, inputs, one_hot_targets, dx=0.0001, batch_size=None):
        data = zip(inputs, one_hot_targets)
        data_length = batch_size if batch_size else len(inputs)
        if batch_size:
            data = random.sample(list(data), batch_size)
        weight_mat_shape = self.weights.shape
        weight_mat_size = np.size(self.weights)
        total_gradient = np.zeros(weight_mat_shape)
        cnt = 0
        for sample_input, target in data:
            sample_gradient = []
            print('{}/{} - '.format(cnt, data_length), end='\r')
            cnt += 1
            for i in range(weight_mat_size):
                self.weights = self.weights.reshape(-1)
                self.weights[i] += dx
                self.weights = self.weights.reshape(weight_mat_shape)
                nudged_out = self.eval(sample_input)
                self.weights = self.weights.reshape(-1)
                self.weights[i] -= dx
                self.weights = self.weights.reshape(weight_mat_shape)

                out = self.eval(sample_input)

            total_gradient += np.reshape(sample_gradient, weight_mat_shape)
        total_gradient /= data_length
        return total_gradient

    def _nugde_weight_at_flattend_index(self, idx, dx):
        weight_mat_shape = self.weights.shape
        self.weights = self.weights.reshape(-1)
        self.weights[idx] += dx
        self.weights = self.weights.reshape(weight_mat_shape)

    def _eval_flattedend_nudged_weight(self, inputs, idx, dx):
        self._nugde_weight_at_flattend_index(idx, dx)
        nudged_out = self.batch_eval(inputs)
        self._nugde_weight_at_flattend_index(idx, -dx)
        return nudged_out

    def gradient_vrt(self, inputs, one_hot_targets, dx=0.01, batch_size=None, loss=mse_loss):
        if batch_size:
            idx = np.random.randint(0, batch_size, batch_size)
            inputs = np.take(inputs, idx, 0)
            one_hot_targets = np.take(one_hot_targets, idx, 0)
        weight_mat_shape = self.weights.shape
        gradient = []
        for i in range(np.size(self.weights)):
            nudged_out = self._eval_flattedend_nudged_weight(inputs, i, dx)
            out = self.batch_eval(inputs)
            gradient.append(-1 * (
                        self.loss_vrt(nudged_out, one_hot_targets, loss_func=loss) - self.loss_vrt(out, one_hot_targets,
                                                                                                   loss_func=loss)) / dx)
        return np.reshape(gradient, weight_mat_shape)

    def eval_performance(self, inputs, one_hot_targets):
        val_res = []
        for x, y in zip(inputs, one_hot_targets):
            val_res.append(np.argmax(self.eval(x)) == np.argmax(y))
        return int(sum(val_res) / len(val_res) * 100)

    def train(self, inputs, one_hot_targets, epochs=1, learning_rate=0.01, verbose=False, batch_size=None,
              loss=mse_loss):
        start_time = time.time()
        prev_loss = None
        print('{}% Classified correctly bevore training'.format(self.eval_performance(inputs, one_hot_targets)))
        for ep in range(epochs):
            gradient = self.gradient_vrt(inputs, one_hot_targets, batch_size=batch_size, loss=loss)
            self.weights = self.weights + (gradient * learning_rate)
            print('\rBatch {}/{}'.format(ep, epochs), end='')
            if verbose:
                print()
                print('gradient max val:', np.max(gradient * learning_rate))
                current_loss = self.loss_vrt(self.batch_eval(inputs), one_hot_targets)
                print('loss:', current_loss)
                if prev_loss:
                    print('loss improvment:', prev_loss - current_loss)
                    if prev_loss - current_loss < 0:
                        print('Waring negative loss improvement detected'
                              '!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!')
                prev_loss = current_loss
                print('weights max', np.max(self.weights))
                print('weights min', np.min(self.weights))
                print('{}% Classified correctly'.format(self.eval_performance(inputs, one_hot_targets)))
                print('----------')
        print('\n' + '{}% Classified correctly after training'.format(self.eval_performance(inputs, one_hot_targets)))
        print('Learned for {}s'.format(time.time() - start_time))