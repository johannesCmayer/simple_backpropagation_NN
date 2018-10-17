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


class FullyConnectedNL():
    def __init__(self, input_size, output_size, non_linearity_func=None, loss_func=mse_loss, regularisation_func=None, regularization_coef=0.01):
        self.weights = np.random.rand(output_size, input_size)
        self.biasses = np.random.uniform(-10,10,output_size)

        self.non_linearity_func = non_linearity_func
        self.loss_func = loss_func
        self.regularisation_func = regularisation_func
        self.regularization_coef = regularization_coef

    def eval(self, inputs):
        active_weights = self.weights
        result = np.matmul(active_weights, inputs) + self.biasses
        if self.non_linearity_func:
            self.non_linearity_func(result)
        return result

    def batch_eval(self, inputs):
        return [self.eval(i) for i in inputs]

    def outputs_to_probabilities(self, outputs):
        squashed_result = outputs / np.amax(outputs)
        unnormal_probabilities = np.exp(squashed_result)
        if unnormal_probabilities[unnormal_probabilities == np.inf].any():
            raise Exception(
                'There are infinity values in result after exponentiation. Weights might be initialized to high. Value is {}'
                .format(unnormal_probabilities))
        probabilities = unnormal_probabilities / sum(unnormal_probabilities)
        return probabilities

    def loss(self, output, one_hot_target):
        loss = self.loss_func(output, one_hot_target)
        if self.regularization:
            loss += self.regularization(self.weights) / self.regularization_coef
        return loss

    def loss_vect(self, outputs, one_hot_targets):
        cost = 0
        for i, t in zip(outputs, one_hot_targets):
            cost += self.loss_func(i, t)
        return cost / len(outputs)

    def _nugde_at_flattend_index(self, arr, idx, dx):
        weight_mat_shape = arr.shape
        arr = arr.reshape(-1)
        arr[idx] += dx
        arr = arr.reshape(weight_mat_shape)

    def _eval_flattedend_nudged(self, arr, inputs, idx, dx):
        self._nugde_at_flattend_index(arr, idx, dx)
        nudged_out = self.batch_eval(inputs)
        self._nugde_at_flattend_index(arr, idx, -dx)
        return nudged_out

    def gradient_vect(self, grad_type, inputs, one_hot_targets, dx=0.01, batch_size=None):
        gradient_types = {'weights': self.weights,
                          'biasses': self.biasses}
        arr_to_calculate_grad = gradient_types.get(grad_type)
        if batch_size:
            idx = np.random.randint(0, batch_size, batch_size)
            inputs = np.take(inputs, idx, 0)
            one_hot_targets = np.take(one_hot_targets, idx, 0)
        weight_mat_shape = arr_to_calculate_grad.shape
        gradient = []
        for i in range(np.size(arr_to_calculate_grad)):
            nudged_out = self._eval_flattedend_nudged(arr_to_calculate_grad, inputs, i, dx)
            out = self.batch_eval(inputs)
            gradient.append(-1 * (self.loss_vect(nudged_out, one_hot_targets) - self.loss_vect(out, one_hot_targets)) / dx)
        return np.reshape(gradient, weight_mat_shape)

    def eval_performance(self, inputs, one_hot_targets):
        val_res = []
        for x, y in zip(inputs, one_hot_targets):
            val_res.append(np.argmax(self.eval(x)) == np.argmax(y))
        return int(sum(val_res) / len(val_res) * 100)

    def train(self, inputs, one_hot_targets, epochs=1, learning_rate=0.01, verbose=False, batch_size=None):
        start_time = time.time()
        prev_loss = None
        print('{}% Classified correctly bevore training'.format(self.eval_performance(inputs, one_hot_targets)))
        current_loss = self.loss_vect(self.batch_eval(inputs), one_hot_targets)
        print('loss at begining:', current_loss)
        try:
            for ep in range(epochs):
                weight_gradient = self.gradient_vect('weights', inputs, one_hot_targets, batch_size=batch_size)
                biasses_gradient = self.gradient_vect('biasses', inputs, one_hot_targets, batch_size=batch_size)
                self.weights = self.weights + (weight_gradient * learning_rate)
                self.biasses = self.biasses + (biasses_gradient * learning_rate)
                print('\rBatch {}/{}'.format(ep, epochs), end='')
                if verbose:
                    print()
                    print('gradient max val:', np.max(weight_gradient * learning_rate))
                    current_loss = self.loss_vect(self.batch_eval(inputs), one_hot_targets)
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
        except KeyboardInterrupt:
            print('Terminated Early')
        print('\n' + '{}% Classified correctly after training'.format(self.eval_performance(inputs, one_hot_targets)))
        print('Learned for {}s'.format(time.time() - start_time))


class Model:
    def __init__(self, loss_func=mse_loss):
        self.layers = []
        self.loss = loss_func

    def add_layer(self, layer):
        self.layers.append(layer)

    def eval(self, inputs):
        result = inputs
        for l in self.layers:
            result = l.eval(result)
        return result

    def batch_eval(self, inputs):
        return [self.eval(i) for i in inputs]

    def loss(self, inputs, labels):
        return self.loss(inputs, labels)

    def get_full_eval_graph(self, inputs):
        result = inputs
        layer_results = []
        for l in self.layers:
            result = l.eval(result)
            layer_results.append(layer_results)
        return layer_results

    def backpropagate(self, inputs, labels):
        eval_graph = self.get_full_eval_graph()
        loss = self.loss(eval_graph[-1], labels)


        gradient = []

        return gradient

    def train(self):
        pass

    def validate_inputs(self, inputs):
        if np.any(inputs > 1):
            print('Warning: Inputs to model are not normalized. Biggest input is {}'.format(np.max(inputs)))
