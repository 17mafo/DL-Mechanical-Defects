import tensorflow as tf
from tensorflow.keras import layers, Model, Input


class BinaryF1Score(tf.keras.metrics.Metric):
    def __init__(self, name='f1', threshold=0.5, **kwargs):
        super().__init__(name=name, **kwargs)
        self.threshold = threshold
        self.precision_metric = tf.keras.metrics.Precision(thresholds=threshold)
        self.recall_metric = tf.keras.metrics.Recall(thresholds=threshold)

    def update_state(self, y_true, y_pred, sample_weight=None):
        self.precision_metric.update_state(y_true, y_pred, sample_weight)
        self.recall_metric.update_state(y_true, y_pred, sample_weight)

    def result(self):
        precision = self.precision_metric.result()
        recall = self.recall_metric.result()
        return 2.0 * ((precision * recall) / (precision + recall + tf.keras.backend.epsilon()))

    def reset_state(self):
        self.precision_metric.reset_state()
        self.recall_metric.reset_state()


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

    def _build_metrics(self, metrics, threshold=0.5):
        metrics = list(metrics) if metrics is not None else ['accuracy']

        metric_names = set()
        for metric in metrics:
            if isinstance(metric, str):
                metric_names.add(metric.lower())
            else:
                metric_names.add(metric.name.lower())

        if 'precision' not in metric_names:
            metrics.append(tf.keras.metrics.Precision(name='precision', thresholds=threshold))
        if 'recall' not in metric_names:
            metrics.append(tf.keras.metrics.Recall(name='recall', thresholds=threshold))
        if 'f1' not in metric_names:
            metrics.append(BinaryF1Score(name='f1', threshold=threshold))

        return metrics



    def train(self, **params):
        # Return history, display graphs etc
        callback = tf.keras.callbacks.EarlyStopping(monitor="val_loss",
                                                    mode="min",
                                                    patience=params.get('patience', 15),
                                                    min_delta=0.001,
                                                    restore_best_weights=True,
                                                    verbose=1
)
        callbacks = [callback]

        if params.get('saveModelCheckpoint', False):
            checkpoint = tf.keras.callbacks.ModelCheckpoint(
                filepath=params.get('checkpoint_path', 'best_model.h5'),
                save_best_only=True,
                monitor="val_loss",
                mode="min",
                verbose=1
            )
            callbacks.append(checkpoint)

        metrics = self._build_metrics(
            params.get('metrics', ['accuracy']),
            threshold=params.get('metric_threshold', 0.5)
        )
        self.model.compile(optimizer=params.get('optimizer', 'adam'), 
                           loss=params.get('loss', 'binary_crossentropy'),
                           metrics=metrics)
        # Lägg till early stopping (Spara den bästa)
        self.history = self.model.fit(self.train_ds, 
                       validation_data=self.val_ds,
                       epochs=params.get('epochs', 50),
                       callbacks=callbacks)
        
        return self.history

    def summary(self):
        self.model.summary()