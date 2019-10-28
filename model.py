from MobileNetV2 import mobilenet_v2
    
class Model:
    def train(self, X, y):
        pass
    
    def test(self, X, y):
        pass
    
    def __init__(self):
        self.net = mobilenet_v2(pretrained=True)
        pass