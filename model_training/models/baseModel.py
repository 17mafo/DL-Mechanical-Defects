import tensorflow as tf
from tensorflow.keras import layers, Model


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
    def __init__(self, model, make2_dense = False, **params):
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

        if make2_dense:
            outputlayer = layers.Dense(params.get('dense_units2', 128), activation=params.get('classifier_activation', 'relu'))(outputlayer)

        outputs = layers.Dense(1, activation=params.get('output_activation', 'sigmoid'))(outputlayer)
        self.model = Model(inputs=self.model.input, outputs=outputs)
    
    def preprocess(self, train_ds, val_ds, preprocess_input):

        for x, _ in train_ds.take(1):
            print(tf.reduce_min(x).numpy(), tf.reduce_max(x).numpy())

        self.train_ds = train_ds.map(
            lambda x, y: (preprocess_input(x), y),
            num_parallel_calls=tf.data.AUTOTUNE
        ).prefetch(tf.data.AUTOTUNE)

        self.val_ds = val_ds.map(
            lambda x, y: (preprocess_input(x), y),
            num_parallel_calls=tf.data.AUTOTUNE
        ).prefetch(tf.data.AUTOTUNE)
        
        for x, _ in self.train_ds.take(1):
            print(tf.reduce_min(x).numpy(), tf.reduce_max(x).numpy())

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
                       callbacks=[callback])
        
        return self.history

    def summary(self):
        self.model.summary()