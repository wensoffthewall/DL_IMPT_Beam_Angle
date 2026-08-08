from dataclasses import dataclass, field
import tensorflow as tf
from tensorflow.keras import layers

@dataclass
class DualStreamU_eu:
    input_shape: tuple = ((256, 256, 128, 13), (8,))
    num_angles: int = 4
    UNet_depth: int = 3
    bottle_neck_dim: int = 256
    shape_layers: int = 1
    shape_neurons: int = 16
    vol_outputs: int = 8
    pool_size: int = 2
    starting_channels: int = 32
    fusion_neurons: int = 32
    name: str | None = None
    seed: int = 777
    dilation: int = 2
    model: tf.keras.Model | None = field(default=None, init=False, repr=False)
    vol_shape: tuple = field(init=False)
    shape_feat_shape: tuple = field(init=False)
    pool_size_3d: tuple = field(init=False, repr=False)

    def __post_init__(self):
        self.name = self.name if self.name is not None else "DualStreamU_eu"
        self.vol_shape, self.shape_feat_shape = self.input_shape
        self.pool_size_3d = (self.pool_size, self.pool_size, self.pool_size)


    def normalize_sin_cos(self, x):
        x = tf.reshape(x, (-1, self.num_angles, 2))
        norm = tf.norm(x, axis=-1, keepdims=True)
        x = x / (norm + tf.keras.backend.epsilon())
        return tf.reshape(x, (-1, self.num_angles * 2))


    def conv_block(self, x, filters, dilation_rate=1):
        initializer = tf.keras.initializers.GlorotUniform(seed=self.seed)
        x = layers.Conv3D(filters, (3, 3, 3), padding="same", activation="relu",
                          kernel_initializer=initializer, dilation_rate=dilation_rate)(x)
        x = layers.BatchNormalization()(x)
        x = layers.Conv3D(filters, (3, 3, 3), padding="same", activation="relu",
                          kernel_initializer=initializer, dilation_rate=dilation_rate)(x)
        x = layers.BatchNormalization()(x)
        return x

    def upsample_block(self, x, filters, skip):
        initializer = tf.keras.initializers.GlorotUniform(seed=self.seed)
        x = layers.Conv3DTranspose(filters, (2, 2, 2), strides=(2, 2, 2), padding="same", kernel_initializer=initializer)(x)
        x = layers.Concatenate()([x, skip])
        x = self.conv_block(x, filters)
        return x

    def build_model(self):
        vol_input = tf.keras.Input(shape=self.vol_shape, name="volume_input")
        shape_input = tf.keras.Input(shape=self.shape_feat_shape, name="shape_input")
        dense_init = tf.keras.initializers.GlorotUniform(seed=self.seed)

        # Encoder
        x = vol_input
        skips = []
        for level in range(self.UNet_depth):
            filters = self.starting_channels * (self.pool_size ** level)
            x = self.conv_block(x, filters)
            skips.append(x)
            x = layers.MaxPooling3D(self.pool_size_3d)(x)

        # Bottleneck
        b = self.conv_block(x, self.bottle_neck_dim, dilation_rate=self.dilation)

        # Decoder
        x = b
        for level, skip in reversed(list(enumerate(skips))):
            filters = self.starting_channels * (self.pool_size ** level)
            x = self.upsample_block(x, filters, skip)

        # Overlap feature fusion
        x = layers.Conv3D(self.vol_outputs, 1, kernel_initializer=dense_init, name='volume_outputs')(x)
        x = layers.GlobalAveragePooling3D()(x)

        shape_feat = layers.Dense(self.shape_neurons, activation='relu', name='shape_inputs',kernel_initializer=dense_init)(shape_input)
        for _ in range(self.shape_layers):
            shape_feat = layers.BatchNormalization()(shape_feat)
            shape_feat = layers.Dense(self.shape_neurons, activation='relu',kernel_initializer=dense_init)(shape_feat)

        fusion = layers.Concatenate(name='fusion_concat')([x, shape_feat])
        fusion = layers.BatchNormalization(name='fusion_bn1')(fusion)
        fusion = layers.Dense(self.fusion_neurons, use_bias=False, name='fusion_dense1', kernel_initializer=dense_init)(fusion)
        fusion = layers.BatchNormalization(name='fusion_bn2')(fusion)
        fusion = layers.Activation('relu', name='fusion_relu1')(fusion)

        x = layers.Dense(self.num_angles * 2 , kernel_initializer=dense_init)(fusion)
        output = layers.Lambda(self.normalize_sin_cos)(x)

        self.model = tf.keras.Model(inputs=[vol_input, shape_input], outputs=output)
        return self.model

