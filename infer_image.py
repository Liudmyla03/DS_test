import argparse, torch
from PIL import Image
from torchvision import transforms, models

def load_model(path):
    ckpt = torch.load(path, map_location="cpu")
    model = models.resnet18(pretrained=True)
    model.fc = torch.nn.Linear(model.fc.in_features, len(ckpt['classes']))
    model.load_state_dict(ckpt['model_state_dict'])
    model.eval()
    return model, ckpt['classes']

def predict(image_path, model, classes, topk=3):
    tf = transforms.Compose([transforms.Resize((224,224)), transforms.ToTensor()])
    img = Image.open(image_path).convert("RGB")
    x = tf(img).unsqueeze(0)
    with torch.no_grad():
        out = model(x)
        probs = torch.nn.functional.softmax(out, dim=1)[0]
        topk_vals, topk_idx = probs.topk(topk)
    return [{"label": classes[i], "score": float(topk_vals[j])} for j,i in enumerate(topk_idx)]

if __name__=="__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", required=True)
    parser.add_argument("--image", required=True)
    args = parser.parse_args()
    model, classes = load_model(args.model_path)
    print(predict(args.image, model, classes, topk=5))
