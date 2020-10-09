import os

import tensorflow as tf
from flask import jsonify, request
from scipy.misc import imread

from app.base import BaseResource
from lib.mtcnn import detect_face
from utils import get_face, identify_face_1v1, forward_pass, load_model


class FirstApi(BaseResource):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.parser.add_argument('name', required=True, type=str,
                                 location=['form', 'json', 'args', 'files', 'values', 'headers'])

    def get(self):
        return 'POST request only'

    def post(self):
        embedding1, face_count1 = cal(request.files['file1'])
        embedding2, face_count2 = cal(request.files['file2'])
        if face_count1 == face_count2 == 1:
            identify = identify_face_1v1(embedding1, embedding2)
            return jsonify({'score': identify})
        else:
            return jsonify({'result': '未检测到人脸'})
        # return 1


def cal(file):
    face_count = 0
    img = imread(name=file, mode='RGB')
    image_size = 160
    img = get_face(
        img=img,
        pnet=pnet,
        rnet=rnet,
        onet=onet,
        image_size=image_size
    )
    # 存在人脸
    if img is not None:
        # 获取当前人脸的 embedding
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


APP_ROOT = os.path.dirname(os.path.abspath(__file__))
embeddings_path = os.path.join(APP_ROOT, 'embeddings')

# 加载预训练模型
model_path = 'model/20190218-164145.pb'
facenet_model = load_model(model_path)
config = tf.ConfigProto()
config.gpu_options.allow_growth = True
images_placeholder = tf.get_default_graph().get_tensor_by_name("input:0")
embeddings = tf.get_default_graph().get_tensor_by_name("embeddings:0")
phase_train_placeholder = tf.get_default_graph().get_tensor_by_name("phase_train:0")
facenet_persistent_session = tf.Session(graph=facenet_model, config=config)
pnet, rnet, onet = detect_face.create_mtcnn(sess=facenet_persistent_session, model_path=embeddings_path)
print('facenet embedding 模型建立完毕')
