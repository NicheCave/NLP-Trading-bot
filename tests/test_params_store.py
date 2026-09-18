"""
Tests for tuned-parameter persistence.

The consensus rule matters: taking the single best-scoring window's
parameters is just picking the luckiest draw. Agreement ACROSS independent
windows is the only weak evidence of generalisation available here.
"""
from dataclasses import replace

import pytest

import params_store
from strategy_v7 import DEFAULT_PARAMS, V7Params


def test_consensus_prefers_the_most_frequently_chosen_value():
    selected = [
        replace(DEFAULT_PARAMS, min_net=0.20),
        replace(DEFAULT_PARAMS, min_net=0.50),
        replace(DEFAULT_PARAMS, min_net=0.50),
    ]
    tuned, stability = params_store.consensus(selected, DEFAULT_PARAMS)
    assert tuned.min_net == pytest.approx(0.50)
    assert stability['min_net'] == pytest.approx(2 / 3)


def test_consensus_never_invents_an_untested_value():
    """
    Regression guard.

    A plain median over an even number of grid choices interpolates between
    them: {1.8, 1.8, 3.5, 3.5} → 2.65, a stop multiplier the walk-forward
    never scored. Saving that would push an untested configuration into live
    trading — the precise failure the tuning exists to prevent.
    """
    observed = [1.8, 1.8, 3.5, 3.5]
    selected = [replace(DEFAULT_PARAMS, atr_stop_mult=v) for v in observed]
    tuned, stability = params_store.consensus(selected, DEFAULT_PARAMS)

    assert tuned.atr_stop_mult in observed
    assert stability['atr_stop_mult'] > 0     # 0 would mean nothing chose it


def test_ties_break_toward_the_middle_of_the_range():
    selected = [replace(DEFAULT_PARAMS, atr_stop_mult=v)
                for v in (1.8, 1.8, 2.5, 2.5, 3.5, 3.5)]
    tuned, _ = params_store.consensus(selected, DEFAULT_PARAMS)
    assert tuned.atr_stop_mult == pytest.approx(2.5)


def test_every_consensus_field_was_actually_observed():
    """No field may hold a value that no window selected."""
    import random
    from dataclasses import fields
    rng = random.Random(0)
    grid = {'min_net': [0.2, 0.35, 0.5],
            'atr_stop_mult': [1.8, 2.5, 3.5],
            'atr_target_mult': [3.6, 5.0, 7.0],
            'min_active_blocks': [3, 4, 5]}

    selected = [replace(DEFAULT_PARAMS,
                        **{k: rng.choice(v) for k, v in grid.items()})
                for _ in range(8)]
    tuned, _ = params_store.consensus(selected, DEFAULT_PARAMS)

    for name in grid:
        observed = {getattr(p, name) for p in selected}
        assert getattr(tuned, name) in observed, f"{name} was never tested"


def test_consensus_takes_the_majority_of_flags():
    selected = [
        replace(DEFAULT_PARAMS, use_funding_gate=True),
        replace(DEFAULT_PARAMS, use_funding_gate=True),
        replace(DEFAULT_PARAMS, use_funding_gate=False),
    ]
    tuned, stability = params_store.consensus(selected, DEFAULT_PARAMS)
    assert tuned.use_funding_gate is True
    assert stability['use_funding_gate'] == pytest.approx(2 / 3)


def test_stability_is_one_when_every_window_agrees():
    selected = [replace(DEFAULT_PARAMS, min_net=0.35) for _ in range(4)]
    _, stability = params_store.consensus(selected, DEFAULT_PARAMS)
    assert stability['min_net'] == pytest.approx(1.0)


def test_integer_fields_stay_integers():
    selected = [
        replace(DEFAULT_PARAMS, min_active_blocks=3),
        replace(DEFAULT_PARAMS, min_active_blocks=4),
    ]
    tuned, _ = params_store.consensus(selected, DEFAULT_PARAMS)
    assert isinstance(tuned.min_active_blocks, int)


def test_empty_selection_falls_back_to_base():
    tuned, stability = params_store.consensus([], DEFAULT_PARAMS)
    assert tuned == DEFAULT_PARAMS
    assert stability == {}


def test_save_and_load_round_trip(tmp_path):
    path = str(tmp_path / "tuned.json")
    original = replace(DEFAULT_PARAMS, min_net=0.42, min_active_blocks=5,
                       use_funding_gate=False)
    params_store.save(original, {'oos_expectancy_r': 0.05}, path=path)

    loaded, metadata = params_store.load(DEFAULT_PARAMS, path=path)
    assert loaded.min_net == pytest.approx(0.42)
    assert loaded.min_active_blocks == 5
    assert loaded.use_funding_gate is False
    assert metadata['oos_expectancy_r'] == pytest.approx(0.05)


def test_missing_file_returns_the_defaults(tmp_path):
    loaded, metadata = params_store.load(DEFAULT_PARAMS,
                                         path=str(tmp_path / "absent.json"))
    assert loaded == DEFAULT_PARAMS
    assert metadata == {}


def test_unknown_fields_are_ignored(tmp_path):
    """An older saved file must not break a newer strategy."""
    path = tmp_path / "tuned.json"
    path.write_text('{"params": {"min_net": 0.33, "a_knob_that_no_longer_exists": 1}}')

    loaded, _ = params_store.load(DEFAULT_PARAMS, path=str(path))
    assert loaded.min_net == pytest.approx(0.33)


def test_corrupt_file_falls_back_safely(tmp_path):
    path = tmp_path / "tuned.json"
    path.write_text("{ not json")
    loaded, _ = params_store.load(DEFAULT_PARAMS, path=str(path))
    assert loaded == DEFAULT_PARAMS
