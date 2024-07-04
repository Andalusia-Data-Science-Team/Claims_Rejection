import re
from AHBS_AIServices import similarity
from transformers import pipeline
import os
import joblib

class RejectionCategory:
    def __init__(self, model_path='data/bart_model/bart_zero_classifier'):
        if model_path and os.path.exists(model_path):
            self.classifier = joblib.load(model_path)
        else:
            self.classifier = pipeline("zero-shot-classification", model="facebook/bart-large-mnli")

        self.categories = [
            "Medical Justification Denials",
            "Contractual or Financial Benefit Denials",
            "Information Errors or Omissions Denials"]
        print('Model loading is done')

    def save_classifier(self, model_path):
        joblib.dump(self.classifier, model_path)

    def classify_message(self, message,predictions):
        result = self.classifier(message, self.categories)
        labels = []
        for i in range(len(result)):
            if predictions[i] == 0:
                most_probable_label = result[i]['labels'][0]
                labels.append(most_probable_label)
            else:
                labels.append('APPROVED_PREDICTION')
            if i%10 == 0:
                print(f"A batch of {i} is done")
        return labels

    def preprocess_messages(self, messages,predictions):
        cleaned_messages = []
        nphies_code_pattern = r'[A-Z]{2}-\d+-\d+'

        for looper in range(len(messages)):
            if predictions[looper] == 0:
                message = messages[looper]
                message = str(message).strip().strip("'''")

                message = re.sub(f"({nphies_code_pattern})", r' \1 ', message)
                parts = message.split()

                unique_parts = []
                seen = set()
                for part in parts:
                    if part not in seen:
                        seen.add(part)
                        unique_parts.append(part)

                cleaned_message = ' '.join(unique_parts)
                cleaned_message = re.sub(r'([a-z])([A-Z])', r'\1 \2', cleaned_message)
                cleaned_messages.append(cleaned_message.strip())
            else:
                cleaned_messages.append('APPROVED_PREDICTION')

        return cleaned_messages

categories = [ "Medical Justification Denials",  "Contractual or Financial Benefit Denials", "Information Errors or Omissions Denials"]
def find_category_similarity(ls1:list,ls2 =categories):
    sim_out = similarity.sentences_cosine_similarity([(ls1)], [(ls2)])
    similarity_match = []; similarity_element = []
    for elem in sim_out:
        similarity_element.append(elem[0])
        similarity_match.append(elem[1])

    return similarity_element, similarity_match