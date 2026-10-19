TOXICITY AI — STATIC BROWSER WEBSITE

This is a server-free demo of your actual trained Bidirectional LSTM weights.

Files:
- index.html: user interface
- app.js: browser inference (TensorFlow.js loaded from CDN)
- weights.bin + weights.json: your original trained model weights
- vocabulary.json: original word-to-index vocabulary

To publish for free:
1. Create a public GitHub repository.
2. Upload all five files to its root (not a nested folder).
3. Under Settings > Pages, choose Deploy from a branch, main, / (root).
4. Wait for GitHub Pages to publish and visit the generated URL.

Important: This website uses a manual implementation of Keras LSTM inference. Before relying on predictions, compare several results with your Colab Python model to confirm numerical agreement. It uses the original saved weights; no retraining was performed.

To test locally: In this folder run `python -m http.server 8000` and visit http://localhost:8000.
Opening index.html directly via file:// will not work because browsers block fetch requests.

TensorFlow.js is loaded from a public CDN, so visitors need internet access.
