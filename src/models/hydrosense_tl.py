"""HydroSense-TL: transfer learning on a frozen YAMNet backbone (README §7.3).

YAMNet (Hershey et al., 2017 — a MobileNetV1 pretrained on AudioSet's 521
classes, served via TensorFlow Hub) computes its own internal log-mel
features directly from a 16 kHz waveform, so this variant is a
`tf.keras.Model`, not a `torch.nn.Module` like `HydroSenseBase`/`HydroSenseSE`.
`src.training.train` dispatches on `isinstance` to pick the matching train
loop. See `scripts/download_yamnet.py` for caching the Hub module locally.
"""

from __future__ import annotations

YAMNET_HUB_URL = "https://tfhub.dev/google/yamnet/1"
YAMNET_EMBEDDING_DIM = 1024


def build_hydrosense_tl(config: dict):
    """Build the Keras model: frozen YAMNet -> mean-pool embeddings -> dense head.

    Import of `tensorflow`/`tensorflow_hub` is local to this function so the
    rest of `src.models` (and anything importing it transitively) stays
    usable in environments without TensorFlow installed.
    """
    import tensorflow as tf
    import tensorflow_hub as hub

    params = config.get("model_params", {})
    num_classes = config.get("num_classes", 5)
    hidden_dim = params.get("hidden_dim", 128)
    dropout = params.get("dropout", 0.3)
    freeze_backbone = params.get("freeze_backbone", True)
    hub_url = params.get("hub_url", YAMNET_HUB_URL)
    num_samples = int(config.get("sample_rate", 16000) * config.get("segment_length", 10.0))

    yamnet_layer = hub.KerasLayer(hub_url, trainable=not freeze_backbone, name="yamnet_backbone")

    waveform_input = tf.keras.Input(shape=(num_samples,), dtype=tf.float32, name="waveform")

    def _embed_one(wav: "tf.Tensor") -> "tf.Tensor":
        # YAMNet's KerasLayer signature: scores, embeddings, spectrogram = yamnet(waveform)
        _, embeddings, _ = yamnet_layer(wav)
        return tf.reduce_mean(embeddings, axis=0)

    pooled_embeddings = tf.keras.layers.Lambda(
        lambda batch: tf.map_fn(_embed_one, batch, fn_output_signature=tf.float32),
        name="yamnet_embed_pool",
    )(waveform_input)

    x = tf.keras.layers.Dense(hidden_dim, activation="relu", name="head_dense")(pooled_embeddings)
    x = tf.keras.layers.Dropout(dropout, name="head_dropout")(x)
    logits = tf.keras.layers.Dense(num_classes, name="head_logits")(x)

    model = tf.keras.Model(inputs=waveform_input, outputs=logits, name="HydroSense_TL")
    model.embedding_dim = YAMNET_EMBEDDING_DIM
    return model
