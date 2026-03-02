import tensorflow as tf
from tensorflow.keras import layers, Model, Input


class BaseModel:
    def __init__(self, model, make2_dense=False, two_inputs=False, **params):

        backbone_kwargs = dict(
            include_top=False,
            weights=params.get('weights', 'imagenet'),
            input_shape=params.get('input_shape', (300, 300, 3)),
            pooling=None,
            classifier_activation=params.get('classifier_activation', 'softmax'))

        if params.get('include_preprocessing', None) is not None:
            backbone_kwargs["include_preprocessing"] = params.get('include_preprocessing', False)

        backbone = model(**backbone_kwargs)
        backbone.trainable = False

        if two_inputs:
            input_a = Input(shape=backbone.input_shape[1:], name="image_a")
            input_b = Input(shape=backbone.input_shape[1:], name="image_b")

            feat_a = backbone(input_a)
            feat_b = backbone(input_b)

            feat_a = layers.GlobalAveragePooling2D()(feat_a)
            feat_b = layers.GlobalAveragePooling2D()(feat_b)

            merged = layers.Concatenate()([feat_a, feat_b])
            x = merged
            inputs = [input_a, input_b]

        else:
            x = backbone.output
            x = layers.GlobalAveragePooling2D()(x)
            inputs = backbone.input

        
        x = layers.Dense(params.get('dense_units', 256),
                         activation=params.get('classifier_activation', 'relu'))(x)

        if make2_dense:
            x = layers.Dense(params.get('dense_units2', 128),
                             activation=params.get('classifier_activation', 'relu'))(x)

        outputs = layers.Dense(1, activation=params.get('output_activation', 'sigmoid'))(x)

        self.model = Model(inputs=inputs, outputs=outputs)
    
    def preprocess(self, train_ds, val_ds, preprocess_input):

        def preprocess_fn(x, y):
            if isinstance(x, (tuple, list)):
                x1, x2 = x
                return (preprocess_input(x1), preprocess_input(x2)), y
            else:
                return preprocess_input(x), y

        self.train_ds = train_ds.map(
            preprocess_fn,
            num_parallel_calls=tf.data.AUTOTUNE
        ).prefetch(tf.data.AUTOTUNE)

        self.val_ds = val_ds.map(
            preprocess_fn,
            num_parallel_calls=tf.data.AUTOTUNE
        ).prefetch(tf.data.AUTOTUNE)



    def train(self, **params):
        # Return history, display graphs etc
        callback = tf.keras.callbacks.EarlyStopping(monitor='loss', patience=params.get('patience', 5), restore_best_weights=True)
        self.model.compile(optimizer=params.get('optimizer', 'adam'), 
                           loss=params.get('loss', 'binary_crossentropy'),
                           metrics=params.get('metrics', ['accuracy']))
        # Lägg till early stopping (Spara den bästa)
        self.history = self.model.fit(self.train_ds, 
                       validation_data=self.val_ds,
                       epochs=params.get('epochs', 50),
                       callbacks=[callback])
        
        return self.history

    def summary(self):
        self.model.summary()