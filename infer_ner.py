import argparse
from transformers import AutoTokenizer, AutoModelForTokenClassification, pipeline

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_dir", required=True)
    parser.add_argument("--text", required=True)
    args = parser.parse_args()

    tokenizer = AutoTokenizer.from_pretrained(args.model_dir)
    model = AutoModelForTokenClassification.from_pretrained(args.model_dir)
    nlp = pipeline("ner", model=model, tokenizer=tokenizer, aggregation_strategy="simple")
    entities = nlp(args.text)
    animals = [e['word'].lower() for e in entities if e['entity_group'].lower().startswith('animal')]
    print({"animals": animals, "entities": entities})

if __name__ == "__main__":
    main()
