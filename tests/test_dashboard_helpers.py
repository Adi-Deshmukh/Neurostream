from neurostream.dashboard.helpers import trial_slider_key


def test_trial_slider_key_is_scoped():
    assert trial_slider_key(1, 'PCA') == 'trial_slider_1_pca'
    assert trial_slider_key(2, 'UMAP') == 'trial_slider_2_umap'
