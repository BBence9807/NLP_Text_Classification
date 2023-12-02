from transformers import pipeline


def modelTest():
    text = "Mi történt a nagyvilágban?"

    classifier = pipeline("text-classification", model="model_exported")
    result = classifier(text)

    print(result)

if __name__ == '__main__':
    modelTest()