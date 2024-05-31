
import torch
import wandb
from transformers import Trainer


class ORPOTrainer(Trainer):
    def __init__(self, alpha, pad, disable_prompt_loss, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.pad = pad
        self.alpha = alpha
        self.loss_fct = torch.nn.CrossEntropyLoss(reduction='none')
        self.disable_prompt_loss = disable_prompt_loss
        print("Pad Token ID: ", self.pad)
        
    def compute_custom_loss(self, logits, labels):
        
        logits = logits.contiguous()
        
        if labels is not None:
            # move labels to correct device to enable model parallelism
            labels = labels.to(logits.device)
            # Shift so that tokens < n predict n
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            
            # Flatten the tokens
            loss = self.loss_fct(shift_logits.transpose(2, 1), shift_labels).mean(dim=-1)
            
        return loss
    
    def compute_logps(self, prompt_attention_mask, chosen_inputs, chosen_attention_mask, logits):
        mask = chosen_attention_mask[:, :-1] - prompt_attention_mask[:, 1:]
        per_token_logps = torch.gather(logits[:, :-1, :].log_softmax(-1), dim=2, 
                                       index=(mask * chosen_inputs[:, 1:]).unsqueeze(2)).squeeze(2)
        return torch.mul(per_token_logps, mask.to(dtype=torch.bfloat16)).sum(dim=1).to(dtype=torch.float64) / mask.sum(dim=1).to(dtype=torch.float64)
        
    def compute_loss(self, model, inputs, return_outputs=False):
        if self.label_smoother is not None and "labels" in inputs:
            labels = inputs.pop("labels")
        else:
            labels = None
        
        # Generate the hidden states for 'chosen' and 'reject'
        neg_labels = inputs['negative_input_ids'].clone()
        pos_labels = inputs['positive_input_ids'].clone()

        ### Discard the prompt tokens in NLL loss if true
        if self.disable_prompt_loss:
            mask = inputs['attention_mask'] * inputs['positive_attention_mask']
            pos_labels = pos_labels * mask.logical_not()
            pos_labels[pos_labels == 0] = self.pad
        ##################################################

        neg_labels[neg_labels == self.pad] = -100
        pos_labels[pos_labels == self.pad] = -100

        

        outputs_neg = model(**{'input_ids': inputs['negative_input_ids'],
                               'attention_mask': inputs['negative_attention_mask'],
                               'labels': neg_labels,}, output_hidden_states=True)      
        outputs_pos = model(**{'input_ids': inputs['positive_input_ids'],
                               'attention_mask': inputs['positive_attention_mask'],
                               'labels': pos_labels,}, output_hidden_states=True)
            
        # Calculate NLL loss
        pos_loss = outputs_pos.loss
        
        # Calculate Log Probability
        pos_prob = self.compute_logps(prompt_attention_mask=inputs['attention_mask'], 
                                      chosen_inputs=inputs['positive_input_ids'], 
                                      chosen_attention_mask=inputs['positive_attention_mask'], 
                                      logits=outputs_pos.logits)
        neg_prob = self.compute_logps(prompt_attention_mask=inputs['attention_mask'], 
                                      chosen_inputs=inputs['negative_input_ids'], 
                                      chosen_attention_mask=inputs['negative_attention_mask'], 
                                      logits=outputs_neg.logits)

        # Calculate log odds
        log_odds = (pos_prob - neg_prob) - (torch.log1p(-torch.exp(pos_prob)) - torch.log1p(-torch.exp(neg_prob)))
        sig_ratio = torch.nn.functional.sigmoid(log_odds)
        ratio = torch.log(sig_ratio)
        
        # Calculate the Final Loss
        loss = torch.mean(pos_loss - self.alpha * ratio).to(dtype=torch.bfloat16)
        
        wandb.log({'Positive Geometric Mean': torch.mean(pos_prob).item(),
                   'Negative Geometric Mean': torch.mean(neg_prob).item(),
                   'Log Odds Ratio': torch.mean(ratio).item(),
                   'Log Odds': torch.mean(log_odds).item()})
        
        return (loss, outputs_pos) if return_outputs else loss