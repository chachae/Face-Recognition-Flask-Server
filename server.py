import os
import time

import tensorflow as tf
from flask import Flask, request, render_template, jsonify
from scipy.misc import imread
from waitress import serve

from lib.mtcnn import detect_face
from utils import (
    load_model,
    get_face,
    save_image,
    forward_pass,
    identify_face_1v1,
    allowed_file
)

app = Flask(__name__)
app.secret_key = os.urandom(24)
APP_ROOT = os.path.dirname(os.path.abspath(__file__))
uploads_path = os.path.join(APP_ROOT, 'uploads')
embeddings_path = os.path.join(APP_ROOT, 'embeddings')
allowed_set = {'png', 'jpg', 'jpeg'}


@app.route("/", methods=['GET'])
def index_page():
    return render_template(template_name_or_list="compare.html")


@app.route("/compare", methods=['POST'])
def compare():
    embedding1, face_count1 = cal(request.files['file1'])
    embedding2, face_count2 = cal(request.files['file2'])
    if face_count1 == face_count2 == 1:
        identify = identify_face_1v1(embedding1, embedding2)
        return jsonify({'score': identify})
    else:
        return jsonify({'result': '未检测到人脸'})


def cal(file):
    face_count = 0
    if file and allowed_file(filename=file.filename, allowed_set=allowed_set):
        img = imread(name=file, mode='RGB')
        img = get_face(
            img=img,
            pnet=pnet,
            rnet=rnet,
            onet=onet,
            image_size=image_size
        )
        # 存在人脸
        if img is not None:
            # 保存人脸
            filename = str(time.time()) + '.' + file.filename.rsplit('.', 1)[1].lower()
            save_image(img=img, filename=filename, uploads_path=uploads_path)
            embedding = forward_pass(
                img=img,
                session=facenet_persistent_session,
                images_placeholder=images_placeholder,
                embeddings=embeddings,
                phase_train_placeholder=phase_train_placeholder,
                image_size=image_size
            )
            return embedding, face_count + 1
        else:
            return 0, face_count
    else:
        return 0, face_count


if __name__ == '__main__':
    """Server and FaceNet Tensorflow configuration."""

    # 加载预训练模型
    model_path = 'model/20180408-102900.pb'
    facenet_model = load_model(model_path)
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    image_size = 160
    images_placeholder = tf.get_default_graph().get_tensor_by_name("input:0")
    embeddings = tf.get_default_graph().get_tensor_by_name("embeddings:0")
    phase_train_placeholder = tf.get_default_graph().get_tensor_by_name("phase_train:0")
    print('facenet embedding 模型建立完毕')
    facenet_persistent_session = tf.Session(graph=facenet_model, config=config)
    pnet, rnet, onet = detect_face.create_mtcnn(sess=facenet_persistent_session, model_path='embeddings')
    serve(app=app, host='0.0.0.0', port=5000)
