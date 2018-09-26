import numpy as np
from hypothesis import given, note, strategies as st
from backprop import *
from unittest.mock import patch

patch = ('backprop.mtp', lambda: print('It was patched.'))
network_spec = st.lists(st.integers(min_value=1, max_value=1), min_size=2, max_size=10)


def re():
    print('oeuth\noneuh\nnoetu\n')
    raise Exception(patch)

@patch('backprop_test.re', new=lambda: print('YEE'))
def test_et():
    re()

# @patch(*patch)
# @given(network_spec)
# def test_create_network_layers(network_spec):
#     n = Network(network_spec)
#     assert len(network_spec) - 1 == len(n.weights)


# @given(network_spec)
# def test_create_network_weights(network_spec):
#     n = Network(network_spec)
#     assert network_spec[0] == len(n.weights[0][0])
#
# @given(network_spec, st.data())
# def test_network_single_output(network_spec, data):
#     n = Network(network_spec)
#     input_data = np.array(data.draw(st.lists(st.floats(), min_size=network_spec[0], max_size=network_spec[0])))
#     assert network_spec[-1] == len(n.eval(input_data))