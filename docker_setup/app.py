import numpy as np
import gradio as gr
import torch
import torchvision.transforms as transforms
from PIL import Image
import intel_extension_for_pytorch as ipex

def get_prediction(input_img):
    print(input_img.shape)  # Assuming input_img is a NumPy array or PIL image

    # Define the necessary transformations
    transform = transforms.Compose([
        transforms.ToPILImage(),                       # Convert NumPy array to PIL image
        transforms.Resize((224, 224)),                # Resize to (224, 224)
        transforms.ToTensor(),                        # Convert image to PyTorch tensor
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # Normalize
    ])

    # Apply the transformations
    input_img = transform(input_img)
    print(input_img.shape)  # Torch tensor shape will be [3, 224, 224]

    # Reshape to match batch size [1, 3, 224, 224]
    input_img = input_img.unsqueeze(0)
    print(input_img.shape)

    # Move input to the appropriate device (GPU/CPU)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    input_img = input_img.to(device)

    # Ensure the model is on the same device
    model.to(device)
    model.eval()  # Set model to evaluation mode

    with torch.no_grad():  # No gradients needed for inference
        pred = model(input_img)
    
    print(pred)  # Raw logits from the model

    # Assuming the model outputs logits, apply softmax to get probabilities
    pred = torch.softmax(pred, dim=1).cpu().numpy()  # Move to CPU and convert to NumPy array
    print(pred)

    # Map predictions to class labels
    results = {
        'Benign': pred[0][0],
        'Malignant': pred[0][1]
    }
    # results = {
    #     'Benign': 0.5,
    #     'Malignant': 0.5
    # }

    return results


demo = gr.Interface(get_prediction, inputs="image", outputs= "label", title= "Skin Cancer Detection")
if __name__ == "__main__":
    model = torch.jit.load('./vit_tiny_patch16_224_scripted.pt')
    # model.compile(optimizer='Adam',loss='categorical_crossentropy',metrics=['accuracy'])
    demo.launch(share= False)
