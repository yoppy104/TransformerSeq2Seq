from pytorch_transformers import BertForMaskedLM, BertTokenizer
import torch

text = "[CLS] I like to play football with my friends [SEP] ."
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
tokenized_text = tokenizer.tokenize(text)

masked_index = 5
tokenized_text[masked_index] = '[MASK]'

indexed_tokens = tokenizer.convert_tokens_to_ids(tokenized_text)
tokens_tensor = torch.tensor([indexed_tokens])

model = BertForMaskedLM.from_pretrained('bert-base-uncased')
model.eval()

with torch.no_grad():
    outputs = model(tokens_tensor)
    predictions = outputs[0]

_, predicted_indexes = torch.topk(predictions[0, masked_index], k=5)
predicted_tokens = tokenizer.convert_ids_to_tokens(predicted_indexes.tolist())
print(predicted_tokens)