# Face-Recognition-Flask-Server

<p align="center">
        <img src="https://img.shields.io/badge/License-Apache%202.0-blue.svg?label=license"/>
             <img src="https://tokei.rs/b1/github/chachae/Face-Recognition-Flask-Server?category=line"/>
</p>
<br/>

本服务基于 <a href="https://github.com/pallets/flask">Flask</a> 框架构建，使用 MTCNN 实现人脸检测，利用 FacNet 计算人脸距离向量，只做一件事情——人脸相似度对比，对外只暴露一个接口，上传两张图片即可完成相似度计算，将持续更新。

--------------------------------------------------------------------------------

# 配置和启动

```shell
pip3 install -r requirements_cpu.txt 
```
```shell
python3 server.py
```

--------------------------------------------------------------------------------

# 配置要求

本服务<strong>不支持</strong> Tensorflow 2.0，并且 python 版本必须高于3，可以根据 <code>requirements_cpu.txt</code> 进行安装配置。

* Python 3+
* Tensorflow < 2.0

--------------------------------------------------------------------------------

# 相关链接

FaceNet 预训练模型下载地址：<a href="https://drive.google.com/file/d/1R77HmFADxe87GmoLwzfgMu_HY0IhcyBz/view">20180408-102900.zip</a>，下载后放置于<code>model</code>文件夹下。

--------------------------------------------------------------------------------

# TODO List

* 提高 Flask 的并发性能
* 提供模型训练

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
