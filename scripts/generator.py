"""
Transformer language model text generator.
@source: https://gist.github.com/thomwolf/1a5a29f6962089e871b94cbd09daf317
"""
import torch
import torch.nn.functional as F

def top_k_top_p_filtering(logits, top_k=15, top_p=0.0, filter_value=-float('Inf')) :
    """Filter a distribution of logits using top-k and/or nucleus (top-p) filtering
        Args:
            logits: logits distribution shape (vocabulary size)
            top_k >0: keep only top k tokens with highest probability (top-k filtering).
            top_p >0.0: keep the top tokens with cumulative probability >= top_p (nucleus filtering).
                Nucleus filtering is described in Holtzman et al. (http://arxiv.org/abs/1904.09751)
    """
    assert logits.dim() == 1  # batch size 1 for now - could be updated for more but the code would be less clear
    top_k = min(top_k, logits.size(-1))  # Safety check
    if top_k > 0:
        # Remove all tokens with a probability less than the last token of the top-k
        indices_to_remove = logits < torch.topk(logits, top_k)[0][..., -1, None]
        logits[indices_to_remove] = filter_value

    if top_p > 0.0:
        sorted_logits, sorted_indices = torch.sort(logits, descending=True)
        cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)

        # Remove tokens with cumulative probability above the threshold
        sorted_indices_to_remove = cumulative_probs > top_p
        # Shift the indices to the right to keep also the first token above the threshold
        sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
        sorted_indices_to_remove[..., 0] = 0

        indices_to_remove = sorted_indices[sorted_indices_to_remove]
        logits[indices_to_remove] = filter_value
    return logits

def nucleus_sampling(model,
                     tokenizer,
                     prompted,
                     device,
                     temperature=0.01,
                     top_k=1,
                     top_p=0.01,
                     max_predicted_units=64) :
    with torch.inference_mode() :
        gen_idxs = [] + prompted
        gen_again = True
        for _ in range(max_predicted_units) :
            if not gen_again :
                continue
            logits = model(torch.tensor(gen_idxs).view(1, -1).long().to(device))
            filtered_logits = top_k_top_p_filtering(logits[0, -1, :] / temperature, top_k=top_k, top_p=top_p)
            probabilities = F.softmax(filtered_logits, dim=-1)
            next_token = torch.multinomial(probabilities, 1)
            gen_idxs.append(next_token.item())
            if next_token.item() == tokenizer.eos_id() :
                gen_again = False
        try :
            out = tokenizer.decode(gen_idxs)
        except:
            print(gen_idxs)
            out = "..."
        return out