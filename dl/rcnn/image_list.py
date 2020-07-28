
class ImageList:

    def __init__(self, tensors, image_size):
        self.tensors = tensors
        self.image_size = image_size

    def to(self, device):
        return ImageList(self.tensors.to(device), self.image_size)