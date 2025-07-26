# Importing dependencies
import streamlit as st
from PIL import Image
import torch
import torchvision.models as models
from torchvision.transforms import transforms
import torch.nn as nn
from torchvision.utils import save_image
from tqdm import tqdm

print("Torch version : ", torch.__version__)
print()
print("Cuda availability : ", torch.cuda.is_available())

model = models.vgg19(pretrained = True).features
feature_index = [0, 5 , 10, 19, 28]

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(device)

img_size = 256
loader = transforms.Compose([transforms.Resize((img_size, img_size)),
                             transforms.ToTensor()])

class vgg(nn.Module):
    def __init__(self):
        super().__init__()
        self.chosen_features = feature_index
        self.model = models.vgg19(pretrained=True).features[:29]   # Taking the features from the pretrained vgg 19 model upto 29 indexing as we need features only upto 28 indexing for our model.

    def forward(self, x):
        features = []
        for layer_num, layer in enumerate(self.model):   # layer_num will be containing the index of the layer currently being iterated and layer will contain the convolution layer at that indexing.
            x = layer(x)
            if layer_num in self.chosen_features:
                features.append(x)   # Collecting the value of only after it is passed through the choosen feature/convolution layer.
        return features


model = vgg().to(device)
model.eval()

def stylize_img(img_path, style_path):
    def load_image(image_path):
        """
        Function to load an image from the specified file path.
        :param image_path: str
        :return: torch.Tensor
        """
        image = Image.open(image_path).convert('RGB')
        image = loader(image).unsqueeze(0)
        return image.to(device)

    img = load_image(img_path)
    style = load_image(style_path)

    origin_img = img
    generated_image = origin_img.clone().requires_grad_(True)
    # Requires grad specifies that the gradient descent or the optimisation will be done on the generated_image.

    # Setting The hyperparameters
    epochs = 5000
    lr = 3e-3
    alpha = 1     # To be multiplied with the content loss.
    beta = 0.0001      # To be multiplied with the style loss.
    # The alpha and beta determines how much of the structure from the original image or how much style do we need in the generated image.
    optimizer = torch.optim.Adam([generated_image], lr = lr)    # Generally we take model.params() as argument but here we need to freeze the model weights so, the loss optimisation will be done on the generated image rather than the model parameters.

    torch.manual_seed(21)
    # Loop for training our model
    for epoch in tqdm(range(epochs)):
        generated_features = model(generated_image)
        original_img_features = model(origin_img)
        style_features = model(style)
        style_loss, original_loss = 0, 0
        for gen_feature, orig_feature, style_feature in zip(generated_features, original_img_features, style_features):
            batch_size, channel, height, width = gen_feature.shape
            original_loss = original_loss + torch.mean((gen_feature - orig_feature)**2)    # Calculating the mean squared error.

            # Computing the gram matrix
            G = gen_feature.view(channel, height*width).mm(gen_feature.view(channel, height*width).t())   # Here mm means matrix multiplication with self and the matrix passed inside it and t means transpose of self.
            # Here we are not multiplying the batch because batch is 1.
            A = style_feature.view(channel, height*width).mm(style_feature.view(channel, height*width).t())
            style_loss = style_loss + torch.mean((G - A)**2)
        total_loss = alpha * original_loss + beta * style_loss
        optimizer.zero_grad()
        total_loss.backward()
        optimizer.step()
        if epoch % 200 == 0:
            print("Total loss = ", total_loss.item())
    generated_image_path = "Output_Image/Style_3_Scenery.png"
    save_image(generated_image, generated_image_path)
    print("Final_Loss : ", total_loss.item())
    return generated_image_path

stylize_img(img_path = "Images/Scenery.jpg",
            style_path = "Styles/Style 3.png")