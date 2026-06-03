from finsentiment.evaluation.metrics import load_finetuned_model, predict

def predict_interactive(config_path: str):
    model, tokenizer = load_finetuned_model(config_path)
    print("\nFinSentinel — Financial Sentiment Classifier")
    print("Type a financial headline to classify. Type 'quit' or 'q' to exit.\n")

    while True:
        headline = input("Headline: ").strip()
        if headline.lower() in ["quit", "exit", 'q']:
            print("Exiting...")
            break
        if not headline:
            continue
        prediction = predict(model, tokenizer, headline)
        print(f"Predicted sentiment: {prediction}")