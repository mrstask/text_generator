import torch
import torch.nn.functional as F
from transformers import GPT2Tokenizer, GPT2LMHeadModel
from random import choice


class TextGenerator:
    def __init__(self):
        self.tokenizer = GPT2Tokenizer.from_pretrained("gpt2")

    def calculate_next_prediction(self, text: str):
        input_ids = torch.tensor(self.tokenizer.encode(text)).unsqueeze(0)
        logits = GPT2LMHeadModel.from_pretrained("gpt2")(input_ids)[0][:, -1]
        probs = F.softmax(logits, dim=-1).squeeze()
        idxs = torch.argsort(probs, descending=True)
        res, cumsum = [], 0.
        for idx in idxs:
            res.append(idx)
            cumsum += probs[idx]
            if cumsum > 0.7:
                pred_idx = idxs.new_tensor([choice(res)])
                break
        return int(pred_idx)

    def convert_prediction(self, prediction: int):
        pred = self.tokenizer.convert_ids_to_tokens(int(prediction))
        return self.tokenizer.convert_tokens_to_string(pred)


def generate_article(topic: str, words_count: int):
    generator = TextGenerator()
    for _ in range(words_count):
        prediction = generator.calculate_next_prediction(topic)
        topic += generator.convert_prediction(prediction)
    return topic


def generate_articles_on_topics(articles_number: int, words_count: int):
    topics = read_topics('essay_topics')
    for _ in range(articles_number):
        topic = topics.pop()
        article = generate_article(topic, words_count)
        with open(f'articles/{topic}', 'w') as f:
            f.write(article)


def read_topics(path: str):
    with open(path, 'r') as f:
        topics = f.read()
    return topics.splitlines()


if __name__ == '__main__':
    generate_articles_on_topics(10, 100)