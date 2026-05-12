from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
from pipline import pipeline

import os

app = Flask(__name__)
CORS(app)

# 上传文件保存目录
UPLOAD_DIR = "./uploads"

os.makedirs(UPLOAD_DIR, exist_ok=True)


@app.route('/predict', methods=['POST'])
def predict():

    # =========================
    # 1. 获取文件
    # =========================
    file = request.files.get("file")

    if file is None:
        return jsonify({
            "status": "error",
            "message": "未接收到文件"
        }), 400

    # 保存文件
    save_path = os.path.join(UPLOAD_DIR, file.filename)

    file.save(save_path)

    print(f"文件已保存: {save_path}")

    # =========================
    # 2. 获取其它参数
    # =========================
    model = request.form.get('model')

    lr = float(request.form.get('lr'))

    batchsize = int(request.form.get('batchsize'))

    epoch = int(request.form.get('epoch'))

    task = request.form.get('task')

    encode = request.form.get('encode')

    estimator = int(request.form.get('estimators'))

    max_depth = int(request.form.get('max_depth'))

    wt_seq = request.form.get('wild_type')

    # =========================
    # 3. 打印调试
    # =========================
    print("========== PARAMS ==========")

    print("model:", model)
    print("lr:", lr)
    print("batchsize:", batchsize)
    print("epoch:", epoch)
    print("task:", task)
    print("encode:", encode)
    print("estimators:", estimator)
    print("max_depth:", max_depth)
    print("wild_type:", wt_seq)

    # =========================
    # 4. 调用 pipeline
    # =========================
    pipeline(
        save_path,
        model,
        lr,
        encode,
        epoch,
        batchsize,
        estimator,
        max_depth,
        task,
        wt_seq
    )

    # =========================
    # 5. 返回结果
    # =========================
    result = {
        "status": "success",
        "message": "分析完成！请查看并保存热图"
    }

    return jsonify(result)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

HEATMAP_DIR = os.path.join(BASE_DIR, "../img/heatmap")

@app.route("/heatmap")
def get_heatmap():

    files = os.listdir(HEATMAP_DIR)

    images = [
        f"/image/{f}"
        for f in files
        if f.endswith((".png", ".jpg", ".jpeg"))
    ]

    return jsonify(images)

@app.route("/image/<filename>")
def serve_image(filename):

    return send_from_directory(
        HEATMAP_DIR,
        filename
    )

if __name__ == '__main__':

    app.run(
        host="0.0.0.0",
        port=5000,
        debug=True
    )