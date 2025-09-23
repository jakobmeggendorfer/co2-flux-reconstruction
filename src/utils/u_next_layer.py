import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.keras.utils import register_keras_serializable

@register_keras_serializable()
class DepthwiseConv(tf.keras.layers.Layer):
    def __init__(self, dim, **kwargs):
        super().__init__(**kwargs)  # handle trainable, name, dtype...
        self.dim = dim
        self.dwconv = layers.DepthwiseConv2D(kernel_size=3, padding="same")

    def call(self, x, *, H, W):
        B = tf.shape(x)[0]
        N = tf.shape(x)[1]
        C = tf.shape(x)[2]

        x = tf.transpose(x, [0, 2, 1])
        x = tf.reshape(x, [B, C, H, W])
        x = tf.transpose(x, [0, 2, 3, 1])
        x = self.dwconv(x)
        x = tf.transpose(x, [0, 3, 1, 2])
        x = tf.reshape(x, [B, C, H * W])
        x = tf.transpose(x, [0, 2, 1])
        return x

    def get_config(self):
        config = super().get_config()
        config.update({
            "dim": self.dim,
        })
        return config

@register_keras_serializable()
class ShiftMLP(tf.keras.layers.Layer):
    def __init__(self, dim, shift_size=4, dropout=0.0, **kwargs):
        super().__init__(**kwargs)  # handles dtype, name, trainable, etc.
        self.dim = dim
        self.shift_size = shift_size
        self.dropout_rate = dropout
        self.pad = shift_size // 2

        self.fc1 = layers.Dense(dim)
        self.dwconv = DepthwiseConv(dim)   # already serializable
        self.act = layers.Activation("relu")
        self.fc2 = layers.Dense(dim)
        self.dropout = layers.Dropout(dropout)

    def shift(self, x, *, H, W, axis):
        x = tf.transpose(x, [0, 2, 1])
        x = tf.reshape(x, [-1, self.dim, H, W])
        x = tf.pad(x, [[0, 0], [0, 0], [self.pad, self.pad], [self.pad, self.pad]])
        x = tf.transpose(x, [0, 2, 3, 1])

        splits = tf.split(x, self.shift_size, axis=-1)
        shifts = [
            tf.roll(s, shift, axis=axis) 
            for s, shift in zip(splits, range(-self.pad, self.pad + 1))
        ]

        x = tf.concat(shifts, axis=-1)
        x = x[:, self.pad:self.pad + H, self.pad:self.pad + W, :]
        x = tf.reshape(x, [-1, H * W, self.dim])
        return tf.transpose(x, [0, 2, 1])

    def call(self, x, *, H, W):
        x = self.shift(x, H=H, W=W, axis=1)
        x = tf.transpose(x, [0, 2, 1])
        x = self.fc1(x)
        x = self.dwconv(x, H=H, W=W)
        x = self.act(x)
        x = self.dropout(x)
        x = tf.transpose(x, [0, 2, 1])

        x = self.shift(x, H=H, W=W, axis=2)
        x = tf.transpose(x, [0, 2, 1])
        x = self.fc2(x)
        x = self.dropout(x)
        return x

    def get_config(self):
        config = super().get_config()
        config.update({
            "dim": self.dim,
            "shift_size": self.shift_size,
            "dropout": self.dropout_rate,
        })
        return config


# One shifted block
@register_keras_serializable()
class ShiftedBlock(tf.keras.layers.Layer):
    def __init__(self, dim, drop_path=0.0, **kwargs):
        super().__init__(**kwargs)
        self.dim = dim
        self.drop_path_rate = drop_path

        self.norm = layers.LayerNormalization(epsilon=1e-6)
        self.mlp = ShiftMLP(dim)
        self.drop_path = layers.Dropout(drop_path)

    def call(self, x, *, H, W):
        res = x
        x = self.norm(x)
        x = self.mlp(x, H=H, W=W)
        return res + self.drop_path(x)

    def get_config(self):
        config = super().get_config()
        config.update({
            "dim": self.dim,
            "drop_path": self.drop_path_rate,
        })
        return config


# Patch embedding layer
@register_keras_serializable()
class OverlapPatchEmbed(tf.keras.layers.Layer):
    def __init__(self, patch_size=7, stride=4, embed_dim=768, **kwargs):
        super().__init__(**kwargs)
        self.patch_size = patch_size
        self.stride = stride
        self.embed_dim = embed_dim

        self.proj = layers.Conv2D(embed_dim, kernel_size=patch_size, strides=stride, padding="same")
        self.norm = layers.LayerNormalization(epsilon=1e-6)

    def call(self, x):
        x = self.proj(x)
        B, H, W, C = tf.shape(x)[0], tf.shape(x)[1], tf.shape(x)[2], tf.shape(x)[3]
        x = tf.reshape(x, [B, H * W, C])
        x = self.norm(x)
        return x, H, W

    def get_config(self):
        config = super().get_config()
        config.update({
            "patch_size": self.patch_size,
            "stride": self.stride,
            "embed_dim": self.embed_dim,
        })
        return config
