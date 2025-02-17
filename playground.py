import os
# Reduce TensorFlow log level for minimal logs
os.environ["TF_CPP_MIN_LOG_LEVEL"] = '1'


from ResNet182DD import ResNet182DD


model = ResNet182DD()
model.build((None, 256, 256, 3))
model.summary()