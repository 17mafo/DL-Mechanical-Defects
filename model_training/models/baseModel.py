import tensorflow as tf
from tensorflow.keras import layers, Model


class BaseModel:
    def __init__(self, model, **params):
        if params.get('include_preprocessing', None) is not None:
            self.model = model(
                include_top=False,
                weights=params.get('weights', 'imagenet'),
                input_tensor=None,
                input_shape=params.get('input_shape', (300, 300, 3)),
                pooling=None,
                classifier_activation=params.get('classifier_activation', 'softmax'),
                include_preprocessing=params.get('include_preprocessing', False)
        ) # input_shape=(224, 224, 3), weights='imagenet', classifier_activation='softmax', dense_units=256, output_activation='sigmoid', include_preprocessing=None
        else:
            self.model = model(
                include_top=False,
                weights=params.get('weights', 'imagenet'),
                input_tensor=None,
                input_shape=params.get('input_shape', (300, 300, 3)),
                pooling=None,
                classifier_activation=params.get('classifier_activation', 'softmax')
        )
        self.model.trainable = False
        outputlayer = self.model.output
        outputlayer = layers.GlobalAveragePooling2D()(outputlayer)
        outputlayer = layers.Dense(params.get('dense_units', 256), activation=params.get('classifier_activation', 'relu'))(outputlayer)
        outputs = layers.Dense(1, activation=params.get('output_activation', 'sigmoid'))(outputlayer)
        self.model = Model(inputs=self.model.input, outputs=outputs)
    
    def preprocess(self, train_ds, val_ds, preprocess_input):
        self.train_ds = train_ds.map(
            lambda x, y: (preprocess_input(x), y),
            num_parallel_calls=tf.data.AUTOTUNE
        ).prefetch(tf.data.AUTOTUNE)

        self.val_ds = val_ds.map(
            lambda x, y: (preprocess_input(x), y),
            num_parallel_calls=tf.data.AUTOTUNE
        ).prefetch(tf.data.AUTOTUNE)



    def train(self, **params):
        self.model.compile(optimizer=params.get('optimizer', 'adam'), loss=params.get('loss', 'sparse_categorical_crossentropy'), metrics=params.get('metrics', ['accuracy', 'f1_score']))
        self.model.fit(self.train_ds, self.val_ds, epochs=params.get('epochs', 10))
        # epochs=10, batch_size=32, optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy']


    def summary(self):
        self.model.summary()