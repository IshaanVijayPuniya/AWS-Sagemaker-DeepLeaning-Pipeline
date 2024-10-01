import tensorflow as tf
import argparse
import os
import numpy as np

def build_model(input_shape, num_layers, dropout_rate):
    """Build a simple deep learning model based on input parameters."""
    model = tf.keras.Sequential()
    model.add(tf.keras.layers.InputLayer(input_shape=input_shape))
    
    for i in range(num_layers):
        model.add(tf.keras.layers.LSTM(128, return_sequences=True))
        model.add(tf.keras.layers.Dropout(dropout_rate))
    
    model.add(tf.keras.layers.Dense(1, activation='sigmoid'))
    
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    return model

def main(args):
    # Load raw sensor data from the SageMaker directory
    train_data = np.load(os.path.join(args.train, 'train_data.npy'))
    val_data = np.load(os.path.join(args.validation, 'val_data.npy'))

    X_train, y_train = train_data[:, :-1], train_data[:, -1]
    X_val, y_val = val_data[:, :-1], val_data[:, -1]

    # Define the model based on input hyperparameters
    model = build_model(input_shape=X_train.shape[1:], num_layers=args.num_layers, dropout_rate=args.dropout_rate)

    # Train the model
    model.fit(X_train, y_train, epochs=args.epochs, batch_size=args.batch_size, validation_data=(X_val, y_val))

    # Save the model to the SageMaker output directory
    model.save(os.path.join(args.model_dir, 'model.h5'))

if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    # SageMaker specific arguments
    parser.add_argument('--train', type=str, default=os.environ['SM_CHANNEL_TRAIN'])
    parser.add_argument('--validation', type=str, default=os.environ['SM_CHANNEL_VALIDATION'])
    parser.add_argument('--model-dir', type=str, default=os.environ['SM_MODEL_DIR'])

    # Hyperparameters passed by SageMaker HyperparameterTuner
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--learning_rate', type=float, default=0.001)
    parser.add_argument('--dropout_rate', type=float, default=0.3)
    parser.add_argument('--num_layers', type=int, default=2)

    args = parser.parse_args()
    main(args)
