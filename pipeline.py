import subprocess, json, sys
from difflib import SequenceMatcher
from pathlib import Path

def run_ner(model_dir, text):
    out = subprocess.check_output(["python","ner/infer_ner.py","--model_dir",model_dir,"--text",text])
    return json.loads(out.decode())

def run_image(model_path, image):
    out = subprocess.check_output(["python","image_model/infer_image.py","--model_path",model_path,"--image",image])
    return json.loads(out.decode())  # adjust if infer_image prints list

def similar(a,b):
    return SequenceMatcher(None,a,b).ratio()

def canonicalize(word):
    m = word.lower().strip().rstrip('s')
    # map synonyms
    map_syn = {"hen":"chicken", "cow":"cow", "sheep":"sheep", "spider":"spider"}
    return map_syn.get(m, m)

def decision(ner_animals, image_preds, threshold=0.6):
    ner_set = set(canonicalize(x) for x in ner_animals)
    img_top = [canonicalize(p['label']) for p in image_preds]
    for n in ner_set:
        for i,p in zip(img_top, image_preds):
            if n == i or similar(n, i) >= threshold:
                return True, {"ner":ner_animals, "image_top":image_preds}
    return False, {"ner":ner_animals, "image_top":image_preds}

if __name__=="__main__":
    model_dir = "ner_model"
    image_model = "image_model.pth"
    text = sys.argv[1]
    image = sys.argv[2]
    ner = run_ner(model_dir, text)['animals']
    img_preds = run_image(image_model, image)
    ok, meta = decision(ner, img_preds)
    print({"result": ok, "meta": meta})
