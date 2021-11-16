# 两个系统

1. Python + Tensorflow + OpenCV 服务器系统

	（1）flower_photos.py

	作用：收集大量的花的图按，用这些原始数据训练模型

	输入数据：flower_photos 文件夹下 daisy、dandelion、roses、sunflowers、tulips文件夹下3670张图片

	输出文件：训练好的模型 Model / model.h5  - 129MB

	(2)  open_model.py

	作用：把原来训练好的数据，用Tensoflow.js转换为不依赖环境的通用文件

	输入数据：训练好的模型 Model / model.h5  - 129MB
	     # 仅适用于安装了TensorFlow和Python环境的服务器使用

	输出文件：转换成通用文件 docs/ModelJS 43.2MB
	     # 可以用于网页端、微信小程序的客户端使用数据，不依赖python环境


2. HTML + JS + CSS 客户端系统

	（1）docs/index.html

	作用：给用户去识别花花

	a.核心识别数据 docs/ModelJS 43.2MB
	b.加载模型数据的js库 tf.2.3.0.min.js
	c.绘制图表的库 tfjs-vis.umd.min.js