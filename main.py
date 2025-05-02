from flask import Flask, request, jsonify, render_template, send_file, abort
import os
from algo import predict_word, combine_lexicon, predic_dataset
import pandas as pd

app = Flask(__name__)

image_dir = './images'

@app.get("/api/data")
def get_api_data():
    dataset = pd.read_csv("./data.csv", index_col="tweet")
    data_dict = dataset.reset_index().to_dict(orient="records")
    return jsonify(data_dict)

@app.post("/api/process")
def post_api_proccess():
    predic_dataset()
    return "success"

@app.get("/api/lexicon")
def get_api_lexicon():
    dataset = combine_lexicon()
    data_dict = dataset.reset_index().to_dict(orient="records")
    return jsonify(data_dict)

@app.get("/api/lexicon-seed")
def get_api_lexicon_seed():
    dataset = pd.read_csv("./default_lexicon.csv", index_col="word")
    data_dict = dataset.reset_index().to_dict(orient="records")
    return jsonify(data_dict)

@app.post("/api/data")
def post_api_data():
    try:
        # Ambil data JSON dari request
        data = request.json
        
        if not data or not isinstance(data, list):
            return jsonify({"error": "Data harus dalam format array/list"}), 400
        
        # Validasi format data
        for item in data:
            if 'tweet' not in item or 'emotion' not in item:
                return jsonify({"error": "Setiap item harus memiliki field 'tweet' dan 'emotion'"}), 400
            
            # Pastikan emotion adalah integer
            try:
                item['emotion'] = int(item['emotion'])
            except ValueError:
                return jsonify({"error": f"Nilai emotion harus berupa integer, ditemukan: {item['emotion']}"}), 400
        
        # Buat DataFrame dari data
        df = pd.DataFrame(data)
        df = df.drop_duplicates(subset=['tweet'])
        # Simpan ke CSV, index=False agar tidak menyimpan nomor baris
        df.to_csv('data.csv', index=False)
        
        # Kembalikan response sukses
        return jsonify({
            "message": "Data berhasil disimpan ke data.csv", 
            "count": len(data)
        }), 200
    
    except Exception as e:
        # Tangani error lainnya
        return jsonify({"error": f"Terjadi kesalahan: {str(e)}"}), 500


@app.get("/")
def get_home():
    return render_template('index.html')

@app.get("/upload")
def get_upload_page():
    return render_template('upload.html')

@app.get("/word-processing")
def get_word_level_page():
    return render_template('word-level.html')

@app.get("/sentence-processing")
def get_sentence_level_processing():
    return render_template('sentence-level.html')

@app.get("/training-results")
def get_training_result_page():
    return render_template('training-result.html')

@app.get("/validation")
def get_validation_page():
    return render_template('validation.html')

@app.route('/image/<path:filename>')
def get_images(filename):
    filepath = os.path.join(image_dir, filename)

    if '..' in filename or filename.startswith('/'):
        return abort(400, 'Nama file tidak valid.')

    if not os.path.isfile(filepath):
        return abort(404, 'File tidak ditemukan.')

    ext = filename.split('.')[-1].lower()
    
    mimetypes = {
        'jpg': 'image/jpeg',
        'jpeg': 'image/jpeg',
        'png': 'image/png'
    }

    mimetype = mimetypes.get(ext, 'application/octet-stream')

    return send_file(filepath, mimetype=mimetype)


@app.route("/predict", methods=['POST'])
def predict():
    try:
        data = request.get_json()

        # Ambil nilai dari key "message"
        message = data.get("message")
        
        if message is None:
            return jsonify({
                "status": "error",
                "message": "Field 'message' is required."
            }), 400
        max_emotion, total = predict_word(message)
        return jsonify({
                "status": "success",
                "result": {
                    "emotion": max_emotion,
                    "desc": total
                }
            })
    except Exception as e:
        return jsonify({
            "status": "error",
            "message": str(e)
        }), 500

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)