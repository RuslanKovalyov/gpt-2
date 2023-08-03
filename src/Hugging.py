from transformers import GPT2LMHeadModel, GPT2Tokenizer
import torch

#'gpt2', 'gpt2-medium', 'gpt2-large', and 'gpt2-xl'
def interact_model(model_name='gpt2-large',
                   temperature=0.8,
                   top_k=40,
                   top_p=1.0):

    # Initialize tokenizer and model
    tokenizer = GPT2Tokenizer.from_pretrained(model_name)
    model = GPT2LMHeadModel.from_pretrained(model_name)

    # We will use this variable to store past states
    past = None

    prompt = input("Model prompt >>> ")

    while True:
        with torch.no_grad():
            # Tokenize text input
            input_ids = tokenizer.encode(prompt, return_tensors='pt')
            # Generate text output
            outputs = model(input_ids, past_key_values=past)
        logits, past = outputs.logits, outputs.past_key_values
        # Apply temperature and select token
        logits = logits[:, -1, :] / temperature
        filtered_logits = top_k_top_p_filtering(logits, top_k=top_k, top_p=top_p)
        probabilities = torch.nn.functional.softmax(filtered_logits, dim=-1)
        next_token = torch.multinomial(probabilities, 1)
        # Add the new token to the output and to the prompt
        generated = torch.cat([input_ids, next_token], dim=-1)
        prompt = tokenizer.decode(next_token[0])

        # print(prompt)
        #make it in line
        print(prompt, end='', flush=True)

def top_k_top_p_filtering(logits, top_k=0, top_p=1.0, filter_value=-float('Inf')):
    """ Filter a distribution of logits using top-k and/or nucleus (top-p) filtering
        Args:
            logits: logits distribution shape (batch size, vocabulary size)
            top_k > 0: keep only top k tokens with highest probability (top-k filtering).
            top_p > 0.0: keep the top tokens with cumulative probability >= top_p (nucleus filtering).
                Nucleus filtering is described in Holtzman et al. (http://arxiv.org/abs/1904.09751)
    """
    top_k = min(top_k, logits.size(-1))  # Safety check
    if top_k > 0:
        # Remove all tokens with a probability less than the last token of the top-k
        indices_to_remove = logits < torch.topk(logits, top_k)[0][..., -1, None]
        logits[indices_to_remove] = filter_value

    if top_p > 0.0:
        sorted_logits, sorted_indices = torch.sort(logits, descending=True)
        cumulative_probs = torch.cumsum(torch.nn.functional.softmax(sorted_logits, dim=-1), dim=-1)

        # Remove tokens with cumulative probability above the threshold (nucleus filtering)
        sorted_indices_to_remove = cumulative_probs > top_p
        # Shift the indices to the right to keep also the first token above the threshold
        sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
        sorted_indices_to_remove[..., 0] = 0

        # Create mask
        mask = torch.zeros_like(logits, dtype=torch.bool)
        mask.scatter_(dim=-1, index=sorted_indices, src=sorted_indices_to_remove)
        logits[mask] = filter_value

    return logits



if __name__ == '__main__':
    interact_model()
