# Face-Recognition-Flask-Server

<p align="center">
        <img src="https://img.shields.io/badge/License-Apache%202.0-blue.svg?label=license"/>
             <img src="https://tokei.rs/b1/github/chachae/Face-Recognition-Flask-Server?category=line"/>
</p>
<br/>

本服务基于 <a href="https://github.com/pallets/flask">Flask</a> 框架构建，使用 MTCNN 实现人脸检测和人脸对齐，利用 FacNet 计算人脸128距离向量，使用欧拉公式计算两个目标向量距离，本服务只做一件事情——人脸相似度对比，对外只暴露一个接口，上传两张图片即可完成相似度计算，将持续更新。

--------------------------------------------------------------------------------

# 配置要求

* Python 3+
* Tensorflow < 2.0
* Unix（Gunicorn 目前不能运行于 Windows）

--------------------------------------------------------------------------------

# 配置和启动

使用 pip 安装相关依赖模块并启动项目

```shell
pip3 install -r requirements_cpu.txt 
gunicorn -c gun.py manage:app
```

--------------------------------------------------------------------------------

# 相关链接

FaceNet 官方预训练模型下载地址：<a href="https://drive.google.com/file/d/1R77HmFADxe87GmoLwzfgMu_HY0IhcyBz/view">20180408-102900.zip</a>，下载后放置于<code>model</code>文件夹下，使用亚洲人脸图库训练的模型对亚洲人脸识别效果会更友好，具体网上有相关资源可供下载，有能力可自行训练。

--------------------------------------------------------------------------------

# TODO List

- [x] 提高 Flask 的并发性能 
- [ ] 提供模型训练

--------------------------------------------------------------------------------

# License

```reStructuredText
Copyright [2020] [chachae]

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
```
