import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Input, LSTM, Dense, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping


class LSTMClassifier:
    def __init__(
        self,
        input_shape,
        num_classes,
        lstm_units=64,
        dropout_rate=0.3,
        learning_rate=0.001
    ):
        self.input_shape = input_shape
        self.num_classes = num_classes
        self.lstm_units = lstm_units
        self.dropout_rate = dropout_rate
        self.learning_rate = learning_rate
        self.model = self._build_model()

    def _build_model(self):
        model = Sequential([
            Input(shape=self.input_shape),
            LSTM(self.lstm_units),
            Dropout(self.dropout_rate),
            Dense(self.num_classes, activation="softmax")
        ])

        model.compile(
            optimizer=Adam(learning_rate=self.learning_rate),
            loss="sparse_categorical_crossentropy",
            metrics=["accuracy"]
        )

        return model

    def train(
        self,
        X_train,
        y_train,
        X_val,
        y_val,
        epochs=50,
        batch_size=32
    ):
        early_stop = EarlyStopping(
            monitor="val_loss",
            patience=5,
            restore_best_weights=True
        )

        history = self.model.fit(
            X_train,
            y_train,
            validation_data=(X_val, y_val),
            epochs=epochs,
            batch_size=batch_size,
            callbacks=[early_stop],
            verbose=1
        )

        return history

    def evaluate(self, X_test, y_test):
        loss, acc = self.model.evaluate(X_test, y_test, verbose=0)
        return loss, acc

    def predict(self, X):
        probs = self.model.predict(X)
        return np.argmax(probs, axis=1)
