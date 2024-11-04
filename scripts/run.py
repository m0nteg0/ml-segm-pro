from PIL import Image
import depth_pro
import numpy as np
import matplotlib.pyplot as plt

# Load model and preprocessing transform
model, transform = depth_pro.create_model_and_transforms()
model.train()
model.cuda()

test_image = np.random.uniform(
    0, 255, [768, 768, 3]
).astype(np.uint8)
input_tensor = transform(test_image).unsqueeze(0)
input_tensor = input_tensor.to('cuda')
predict = model(input_tensor)
a = 0