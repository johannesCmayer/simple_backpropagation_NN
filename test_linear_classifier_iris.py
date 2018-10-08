from linear_classifier_iris import *
import numpy as np
from numpy import array as a
import logging as lg
from hypothesis import given, strategies as st
import pytest as pt

lg.getLogger().setLevel(lg.DEBUG)

@given(st.integers(), st.integers())
def test_hinge_loss_one_input(input, output):
    with pt.raises(Exception):
        hinge_loss(a([input]), a([output]))

def test_hinge_loss_no_loss():
    assert 0 == hinge_loss(a([1, 0]), a([1, 0]))

def test_hinge_loss_no_loss():
    assert 0 == hinge_loss(a([1, 0]), a([1, 0]))

def test_hinge_loss_with_loss():
    print('loss is {}'.format(hinge_loss(a([0, 1]), a([1, 0]))))
    assert 0 < hinge_loss(a([0, 1]), a([1, 0]))

@given(st.lists(st.floats(min_value=0, max_value=1), min_size=2, max_size=10))
def test_linear_classifier(inputs):
    n_inputs = len(inputs)
    lc = LinearClassifier(n_inputs, n_inputs)
    print(sum(lc.alt_gradient_vrt([inputs], [inputs])))
    print(sum(lc.gradient_vrt_DEPRECATED([inputs], [inputs])))
    print('------')
    assert np.allclose(lc.alt_gradient_vrt([inputs], [inputs]),
                       lc.gradient_vrt_DEPRECATED([inputs], [inputs]), rtol=0.01, atol=0.01)

