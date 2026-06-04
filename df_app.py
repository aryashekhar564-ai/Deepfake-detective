import os, io, base64, warnings
import numpy as np
from flask import Flask, request, jsonify, render_template
from PIL import Image
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.applications.vgg16 import preprocess_input as vgg_preprocess
from tensorflow.keras.applications.resnet50 import preprocess_input as resnet_preprocess
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import cv2

warnings.filterwarnings('ignore')

app = Flask(__name__)

MODELS_DIR = os.path.join(os.path.dirname(__file__), 'models')
MODEL_CONFIGS = [
    ('LeNet',    'LeNet_best.keras',    64,  None),
    ('AlexNet',  'AlexNet_best.keras',  64,  None),
    ('VGG16',    'VGG16_best.keras',    224, vgg_preprocess),
    ('ResNet50', 'ResNet50_best.keras', 224, resnet_preprocess),
]

print('Loading models...')
loaded_models = {}
for name, fname, _, _ in MODEL_CONFIGS:
    path = os.path.join(MODELS_DIR, fname)
    if os.path.exists(path):
        loaded_models[name] = load_model(path)
        dummy_size = 64 if name in ('LeNet', 'AlexNet') else 224
        loaded_models[name].predict(np.zeros((1, dummy_size, dummy_size, 3)), verbose=0)
        print(f'  Loaded: {name}')
    else:
        print(f'  WARNING: {fname} not found at {path}')
print('Models ready.')


def find_last_conv_layer(model):
    def _all_layers(m):
        result = []
        for layer in m.layers:
            if hasattr(layer, 'layers'):
                result.extend(_all_layers(layer))
            else:
                result.append(layer)
        return result
    for layer in reversed(_all_layers(model)):
        if isinstance(layer, tf.keras.layers.Conv2D):
            return layer.name
    raise ValueError(f'No Conv2D layer found in {model.name}')


def make_gradcam_heatmap(img_array, model, last_conv_name, target_class=None):
    if isinstance(model, tf.keras.Sequential):
        inp = tf.keras.Input(shape=img_array.shape[1:])
        x = inp
        conv_out_tensor = None
        for layer in model.layers:
            x = layer(x)
            if layer.name == last_conv_name:
                conv_out_tensor = x
        grad_model = tf.keras.models.Model(inputs=inp, outputs=[conv_out_tensor, x])
    else:
        grad_model = tf.keras.models.Model(
            inputs=model.inputs,
            outputs=[model.get_layer(last_conv_name).output, model.output]
        )

    with tf.GradientTape() as tape:
        inputs = tf.cast(img_array, tf.float32)
        conv_outputs, preds = grad_model(inputs)
        pred_score = float(preds[0][0])
        if target_class is None:
            target_class = int(pred_score >= 0.5)
        loss = preds[:, 0] if target_class == 1 else 1.0 - preds[:, 0]

    grads    = tape.gradient(loss, conv_outputs)
    pooled   = tf.reduce_mean(grads, axis=(0, 1, 2))
    conv_out = conv_outputs[0]
    heatmap  = conv_out @ pooled[..., tf.newaxis]
    heatmap  = tf.squeeze(heatmap)
    heatmap  = tf.maximum(heatmap, 0)
    heatmap  = heatmap / (tf.math.reduce_max(heatmap) + 1e-8)
    return heatmap.numpy(), pred_score


def overlay_heatmap(img_array, heatmap, alpha=0.45):
    H, W       = img_array.shape[:2]
    hm_uint8   = np.uint8(255 * heatmap)
    hm_resized = cv2.resize(hm_uint8, (W, H))
    hm_colored = cv2.applyColorMap(hm_resized, cv2.COLORMAP_VIRIDIS)
    hm_rgb     = cv2.cvtColor(hm_colored, cv2.COLOR_BGR2RGB)
    img_uint8  = np.uint8(img_array * 255)
    blended    = (img_uint8 * (1 - alpha) + hm_rgb * alpha).astype(np.uint8)
    return blended


def compute_fft(img_array):
    gray = (0.299 * img_array[:,:,0] + 0.587 * img_array[:,:,1] + 0.114 * img_array[:,:,2])
    f_shifted = np.fft.fftshift(np.fft.fft2(gray))
    log_mag   = np.log1p(np.abs(f_shifted))
    return log_mag


def arr_to_b64(arr, cmap=None):
    fig, ax = plt.subplots(figsize=(3, 3))
    if cmap:
        ax.imshow(arr, cmap=cmap)
    else:
        ax.imshow(arr)
    ax.axis('off')
    plt.tight_layout(pad=0)
    buf = io.BytesIO()
    plt.savefig(buf, format='png', bbox_inches='tight', pad_inches=0, dpi=100)
    plt.close(fig)
    buf.seek(0)
    return base64.b64encode(buf.read()).decode('utf-8')


def compute_verdict(results):
    """
    Weighted ensemble verdict.
    VGG16 and ResNet50 carry 99% of the weight (49.5% each).
    LeNet and AlexNet share the remaining 1% (0.5% each).
    If VGG16 and ResNet50 disagree, verdict is UNCERTAIN.
    """
    weights = {'LeNet': 0.005, 'AlexNet': 0.005, 'VGG16': 0.495, 'ResNet50': 0.495}
    scores  = {r['model']: r['score'] for r in results}

    if 'VGG16' in scores and 'ResNet50' in scores:
        vgg_pred    = 'Real' if scores['VGG16']    >= 0.5 else 'Fake'
        resnet_pred = 'Real' if scores['ResNet50'] >= 0.5 else 'Fake'
        if vgg_pred != resnet_pred:
            return {
                'label':      'UNCERTAIN',
                'confidence': 0,
                'message':    'VGG16 and ResNet50 disagree — image could not be classified with confidence.',
                'color':      'uncertain',
            }

    weighted_score = sum(scores[r['model']] * weights[r['model']] for r in results if r['model'] in weights)
    total_weight   = sum(weights[r['model']] for r in results if r['model'] in weights)
    final_score    = weighted_score / total_weight

    label      = 'REAL' if final_score >= 0.5 else 'FAKE'
    confidence = final_score if final_score >= 0.5 else 1 - final_score

    return {
        'label':      label,
        'confidence': round(float(confidence) * 100, 1),
        'message':    'Weighted ensemble verdict — VGG16 and ResNet50 carry 99% of the decision.',
        'color':      'real' if label == 'REAL' else 'fake',
    }


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/analyze', methods=['POST'])
def analyze():
    if 'image' not in request.files:
        return jsonify({'error': 'No image uploaded'}), 400

    file    = request.files['image']
    pil_img = Image.open(file.stream).convert('RGB')

    # FFT
    fft_arr  = np.array(pil_img.resize((224, 224)), dtype=np.float32) / 255.0
    fft_spec = compute_fft(fft_arr)
    fft_b64  = arr_to_b64(fft_spec, cmap='inferno')

    # Original image
    orig_arr = np.array(pil_img.resize((224, 224)), dtype=np.float32) / 255.0
    orig_b64 = arr_to_b64(orig_arr)

    results = []
    for name, fname, img_size, preprocess_fn in MODEL_CONFIGS:
        if name not in loaded_models:
            continue

        model   = loaded_models[name]
        resized = pil_img.resize((img_size, img_size))
        arr     = np.array(resized, dtype=np.float32) / 255.0

        if preprocess_fn:
            inp = preprocess_fn(np.array(resized, dtype=np.float32)[np.newaxis, ...])
        else:
            inp = arr[np.newaxis, ...]

        try:
            conv_name           = find_last_conv_layer(model)
            heatmap, pred_score = make_gradcam_heatmap(inp, model, conv_name)
            overlay             = overlay_heatmap(arr, heatmap)
            gradcam_b64         = arr_to_b64(overlay)
        except Exception:
            pred_score  = float(model.predict(inp, verbose=0)[0][0])
            gradcam_b64 = orig_b64

        pred_label = 'Real' if pred_score >= 0.5 else 'Fake'
        confidence = pred_score if pred_score >= 0.5 else 1 - pred_score

        results.append({
            'model':      name,
            'prediction': pred_label,
            'score':      round(float(pred_score), 4),
            'confidence': round(float(confidence) * 100, 1),
            'gradcam':    gradcam_b64,
            'coarse':     img_size == 64,
        })

    verdict = compute_verdict(results)

    return jsonify({
        'original': orig_b64,
        'fft':      fft_b64,
        'results':  results,
        'verdict':  verdict,
    })


if __name__ == '__main__':
    app.run(debug=True, port=5000)