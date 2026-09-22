import numpy as np

from rf_crowdsense.generator.synthetic import (
    ACTIVITY_LABELS,
    GeneratorConfig,
    activity_class,
    generate_iq,
)


def test_activity_bins():
    assert activity_class(0) == 0
    assert activity_class(5) == 0
    assert activity_class(6) == 1
    assert activity_class(20) == 1
    assert activity_class(21) == 2
    assert activity_class(101) == 4
    assert len(ACTIVITY_LABELS) == 5


def test_generate_iq_shape_and_metadata():
    rng = np.random.default_rng(7)
    cfg = GeneratorConfig(num_samples=1024, max_devices=50)
    iq, meta = generate_iq(rng, cfg, 20)
    assert iq.shape == (1024,)
    assert np.iscomplexobj(iq)
    assert 0 <= meta["active_devices"] <= 20
    assert 0.0 <= meta["activity_score"] <= 1.0
    assert 0 <= meta["activity_class"] <= 4
