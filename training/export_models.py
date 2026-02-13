import json
import numpy as np
import joblib
import tensorflow as tf
def export_sklearn_model(model_path: str, output_path: str):
    """Export sklearn linear model coefficients to JSON"""
    data = joblib.load(model_path)
    if isinstance(data, dict):
        model = data['model']
        scaler = data.get('scaler')
        y_scaler = data.get('y_scaler')
    else:
        model = data
        scaler = None
        y_scaler = None
    
    export_data = {
        'coefficients': model.coef_.tolist(),
        'intercept': float(model.intercept_),
    }
    if scaler is not None:
        export_data['scaler'] = {
            'mean': scaler.mean_.tolist(),
            'scale': scaler.scale_.tolist()
        }
    if y_scaler is not None:
        export_data['y_scaler'] = {
            'mean': float(y_scaler.mean_[0]),
            'scale': float(y_scaler.scale_[0])
        }
    with open(output_path, 'w') as f:
        json.dump(export_data, f, indent=2)
    print(f"Exported sklearn model to {output_path}")
    print(f"  Coefficients: {len(export_data['coefficients'])}")
    print(f"  Has scaler: {scaler is not None}")
    print(f"  Has y_scaler: {y_scaler is not None}")

def export_keras_lstm(model_path: str, output_path: str):
    model = tf.keras.models.load_model(model_path)
    weights = {}
    for i, layer in enumerate(model.layers):
        layer_weights = layer.get_weights()
        if layer_weights:
            layer_info = {
                'type': layer.__class__.__name__,
                'weights': [w.tolist() for w in layer_weights]
            }
            # Add LSTM-specific config
            if hasattr(layer, 'return_sequences'):
                layer_info['return_sequences'] = layer.return_sequences
            # Add Dense activation config
            if hasattr(layer, 'activation'):
                layer_info['activation'] = layer.activation.__name__
            weights[f"layer_{i}_{layer.name}"] = layer_info
    config = {
        'input_shape': list(model.input_shape[1:]),
        'output_shape': list(model.output_shape[1:]) if model.output_shape[1:] else [1],
        'layers': weights
    }
    with open(output_path, 'w') as f:
        json.dump(config, f, indent=2)
    print(f"Exported Keras model to {output_path}")
    print(f"  Input shape: {config['input_shape']}")
    print(f"  Layers: {len(weights)}")
if __name__ == "__main__":
    import os
    models_dir = "models"
    # Export sklearn models
    for name in ['regularizedLinear', 'elasticNet']:
        joblib_path = os.path.join(models_dir, f"{name}.joblib")
        if os.path.exists(joblib_path):
            export_sklearn_model(joblib_path, os.path.join(models_dir, f"{name}.json"))
    # Export Keras LSTM model
    lstm_path = os.path.join(models_dir, "lstm.keras")
    if os.path.exists(lstm_path):
        export_keras_lstm(lstm_path, os.path.join(models_dir, "lstm.json"))