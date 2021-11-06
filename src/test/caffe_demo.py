import caffe

caffe.set_mode_cpu()

model_def = 'caffe_model/deploy.prototxt'
model_weights = 'caffe_model/deploy.caffemodel'

net = caffe.Net(model_def,
                model_weights,
                caffe.TEST)

print(net)