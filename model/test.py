import caffe

# Load the network using the deploy prototxt and weights (.caffemodel file)
net = caffe.Net('colorization_deploy_v2.prototxt', 'colorization_release_v2.caffemodel', caffe.TEST)
