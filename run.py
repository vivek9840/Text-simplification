from transformers import T5ForConditionalGeneration, T5Tokenizer

# Load the trained model and tokenizer
model_path = ""  # Update this path to the directory where the model is saved
model = T5ForConditionalGeneration.from_pretrained(model_path)
tokenizer = T5Tokenizer.from_pretrained("t5-base")

# Input Text
input_sentence = "simplify this sentence: Adjacent counties are Marin (to the south), Mendocino (to the north), Lake (northeast), Napa (to the east), and Solano and Contra Costa (to the southeast)."

# Tokenize the input sentence
input_ids = tokenizer(input_sentence, return_tensors="pt").input_ids

# Generate the simplified output
output_ids = model.generate(input_ids,max_new_tokens = 512)

# Decode the output into text
output_text = tokenizer.decode(output_ids[0], skip_special_tokens=True)

# Print the results
print("Input Sentence:", input_sentence)
print("Simplified Output:", output_text)
