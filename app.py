from flask import Flask, render_template, jsonify, request
import pickle
import numpy as np
import os

app = Flask(__name__)

# ---------- Global Variables ----------
kmeans = None
scaler = None
model_info = {
    'error': None,
    'loaded': False
}

def load_models():
    global kmeans, scaler, model_info
    try:
        # KMeans model load කරමු
        if not os.path.exists('kmeans_model.pkl'):
            model_info['error'] = 'kmeans_model.pkl not found!'
            print("❌ kmeans_model.pkl හමු නොවුණා!")
            return
            
        with open('kmeans_model.pkl', 'rb') as f:
            kmeans = pickle.load(f)
        
        # Scaler load කරමු
        if not os.path.exists('scaler.pkl'):
            model_info['error'] = 'scaler.pkl not found!'
            print("❌ scaler.pkl හමු නොවුණා!")
            return
            
        with open('scaler.pkl', 'rb') as f:
            scaler = pickle.load(f)
        
        # Model info extract
        model_info['model_type'] = 'KMeans'
        model_info['n_clusters'] = kmeans.n_clusters
        model_info['n_features'] = kmeans.n_features_in_
        model_info['inertia'] = float(kmeans.inertia_)
        model_info['n_iter'] = int(kmeans.n_iter_)
        
        # Cluster centers (2D list)
        centers = kmeans.cluster_centers_.tolist()
        model_info['cluster_centers'] = centers
        
        # Scaler info (optional)
        if hasattr(scaler, 'mean_'):
            model_info['scaler_mean'] = scaler.mean_.tolist()
        if hasattr(scaler, 'scale_'):
            model_info['scaler_scale'] = scaler.scale_.tolist()
            
        model_info['loaded'] = True
        print("✅ Models loaded successfully!")
        print(f"   Clusters: {model_info['n_clusters']}, Features: {model_info['n_features']}")
        
    except Exception as e:
        model_info['error'] = str(e)
        print(f"❌ Error loading models: {e}")

load_models()

# ---------- Routes ----------
@app.route('/')
def index():
    return render_template('index.html')

@app.route('/api/model_info')
def get_model_info():
    # Unserializable දේවල් ඉවත් කරමු
    safe_data = {k: v for k, v in model_info.items() if k not in ['_model_obj']}
    return jsonify(safe_data)

@app.route('/api/predict', methods=['POST'])
def predict_cluster():
    if not model_info['loaded']:
        return jsonify({'error': 'Models not loaded. Check server logs.'}), 500
        
    try:
        req = request.get_json()
        features = req.get('features')
        
        if not features:
            return jsonify({'error': 'No features provided'}), 400
        
        # Convert to numpy array
        if isinstance(features[0], list):
            X = np.array(features)
        else:
            X = np.array([features])
        
        # Scale
        X_scaled = scaler.transform(X)
        
        # Predict
        cluster_ids = kmeans.predict(X_scaled).tolist()
        
        # Get corresponding cluster centers
        centers = [kmeans.cluster_centers_[c].tolist() for c in cluster_ids]
        
        return jsonify({
            'success': True,
            'prediction': cluster_ids,
            'centers': centers,
            'n_clusters': kmeans.n_clusters
        })
    
    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)