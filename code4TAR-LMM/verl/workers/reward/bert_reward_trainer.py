import os
import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader
from transformers import BertTokenizer, BertModel
from torch.optim import AdamW
from sklearn.metrics import accuracy_score, f1_score
import logging
import torch.nn as nn
import torch.nn.functional as F

logger = logging.getLogger(__name__)


LOCAL_MODEL_PATH = "sentence_bert"


class BertMeanMaxPoolingClassifier(nn.Module):
    def __init__(self, bert_model_path, hidden_size=768, dropout_rate=0.3):
        super(BertMeanMaxPoolingClassifier, self).__init__()

        self.bert = BertModel.from_pretrained(bert_model_path)
        

        self.dropout = nn.Dropout(dropout_rate)
        self.hidden_size = hidden_size
        

        classifier_input_size = hidden_size * 2
        

        self.classifier = nn.Sequential(
            nn.Linear(classifier_input_size, hidden_size // 2),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(hidden_size // 2, 2)
        )

        
    def forward(self, input_ids, attention_mask, labels=None):

        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        

        hidden_states = outputs.last_hidden_state  # [batch_size, seq_len, hidden_size]
        

        attention_mask_expanded = attention_mask.unsqueeze(-1).expand(hidden_states.size())
        sum_hidden = torch.sum(hidden_states * attention_mask_expanded, 1)
        sum_mask = torch.sum(attention_mask_expanded, 1)
        sum_mask = torch.clamp(sum_mask, min=1e-9)
        mean_pool = sum_hidden / sum_mask
        

        hidden_states_masked = hidden_states + (1.0 - attention_mask_expanded) * -10000.0
        max_pool, _ = torch.max(hidden_states_masked, dim=1)
        

        pooled_output = torch.cat([mean_pool, max_pool], dim=-1)
        

        pooled_output = self.dropout(pooled_output)
        

        logits = self.classifier(pooled_output)
        

        loss = None
        if labels is not None:
            loss_fn = nn.CrossEntropyLoss()
            loss = loss_fn(logits, labels)
        
        return {
            'loss': loss,
            'logits': logits
        }

class ThinkingQualityDataset(Dataset):

    
    def __init__(self, questions, thinking_processes, labels, tokenizer, max_length=128):

        self.questions = questions
        self.thinking_processes = thinking_processes
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length
        
    def __len__(self):
        return len(self.questions)
    
    def __getitem__(self, idx):
        question = self.questions[idx]
        thinking = self.thinking_processes[idx]
        label = self.labels[idx]
        

        text = f"{question} {thinking}"
        

        encoding = self.tokenizer(
            text,
            truncation=True,
            max_length=self.max_length,
            padding="max_length",
            return_tensors="pt"
        )
        

        input_ids = encoding["input_ids"].squeeze()
        attention_mask = encoding["attention_mask"].squeeze()
        
        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": torch.tensor(label, dtype=torch.long)
        }

class BertRewardTrainer:

    
    def __init__(self, model_name=LOCAL_MODEL_PATH, device=None, checkpoint_dir="bert_reward_model"):

        self.model_name = model_name
        

        if device is None:

            print(f"CUDA is available: {torch.cuda.is_available()}")
            if torch.cuda.is_available():
                print(f"Available CUDA devices: {torch.cuda.device_count()}")
                for i in range(torch.cuda.device_count()):
                    print(f"CUDA device {i}: {torch.cuda.get_device_name(i)}")
                print(f"Current CUDA device: {torch.cuda.current_device()}")
                

                self.device = "cuda"
                print(f"Forced BERT model to use CUDA device")
            else:

                try:

                    if "CUDA_VISIBLE_DEVICES" in os.environ and os.environ["CUDA_VISIBLE_DEVICES"] == "-1":
                        os.environ["CUDA_VISIBLE_DEVICES"] = "0"
                        print("Try to reset CUDA_VISIBLE_DEVICES environment variable")
 
                        if torch.cuda.is_available():
                            self.device = "cuda"
                            print("CUDA is available after environment variable adjustment")
                        else:
                            self.device = "cpu"
                            print("CUDA is still unavailable after environment variable adjustment")
                    else:
                        self.device = "cpu"
                except Exception as e:
                    print(f"Error adjusting CUDA environment variable: {e}")
                    self.device = "cpu"
        else:
 
            self.device = device
            
        self.checkpoint_dir = checkpoint_dir
        
        

        checkpoint_path = os.path.abspath(checkpoint_dir)
        os.makedirs(checkpoint_path, exist_ok=True)
        

        self.batch_size = 16
        self.learning_rate = 5e-6
        self.dropout_rate = 0.3
        self.weight_decay = 0.01
        self.max_length = 128
        self.epochs = 10
        
        try:

            print(f"Load tokenizer from local path: {model_name}")
            self.tokenizer = BertTokenizer.from_pretrained(model_name)
            

            self.model = BertMeanMaxPoolingClassifier(
                bert_model_path=model_name,
                dropout_rate=self.dropout_rate
            )
            self.model.to(self.device)
            
            model_device = next(self.model.parameters()).device

            if str(model_device) != str(self.device):

                if torch.cuda.is_available() and str(model_device) == 'cpu':

                    self.model = self.model.to('cuda')
                    self.device = 'cuda'

        except Exception as e:

            print(f"Error initializing BERT model: {str(e)}")
            raise RuntimeError(f"Failed to load BERT model: {str(e)}")
    
    def train(self, questions, thinking_processes, labels, batch_size=None, epochs=None, learning_rate=None):


        if torch.cuda.is_available() and str(next(self.model.parameters()).device) == 'cpu':
            print("Before training, found the model still on CPU, try to move to GPU...")
            try:
                self.model = self.model.to('cuda')
                self.device = 'cuda'
                print(f"Re-moved the model to device: {next(self.model.parameters()).device}")
            except Exception as e:
                print(f"Error moving model to GPU: {e}")
        
        
        print(f"【Before training, confirm the device】BERT model is currently on device: {next(self.model.parameters()).device}")
        print(f"CUDA is available (training): {torch.cuda.is_available()}")
        if torch.cuda.is_available():
            print(f"GPU memory usage:")
            for i in range(torch.cuda.device_count()):
                print(f"   Device {i}: Used {torch.cuda.memory_allocated(i)/1024**3:.2f}GB / Total {torch.cuda.get_device_properties(i).total_memory/1024**3:.2f}GB")
        
        
        batch_size = batch_size if batch_size is not None else self.batch_size
        learning_rate = learning_rate if learning_rate is not None else self.learning_rate
        epochs = epochs if epochs is not None else self.epochs
        
        print(f"Start training BERT reward model, epochs: {epochs}, batch size: {batch_size}, learning rate: {learning_rate}")
        
        dataset = ThinkingQualityDataset(
            questions, 
            thinking_processes, 
            labels, 
            self.tokenizer, 
            max_length=self.max_length
        )
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
        
        
        no_decay = ['bias', 'LayerNorm.weight']
        optimizer_grouped_parameters = [
            {
                'params': [p for n, p in self.model.named_parameters() if not any(nd in n for nd in no_decay)],
                'weight_decay': self.weight_decay
            },
            {
                'params': [p for n, p in self.model.named_parameters() if any(nd in n for nd in no_decay)],
                'weight_decay': 0.0
            }
        ]
        optimizer = AdamW(optimizer_grouped_parameters, lr=learning_rate)
        
        
        self.model.train()
        for epoch in range(epochs):
            total_loss = 0
            all_preds = []
            all_labels = []
            
            for batch in dataloader:
                
                input_ids = batch["input_ids"].to(self.device)
                attention_mask = batch["attention_mask"].to(self.device)
                labels = batch["labels"].to(self.device)
                
                
                if epoch == 0 and batch is next(iter(dataloader)):
                    print(f"【Data device check】")
                    print(f"   Model device: {next(self.model.parameters()).device}")
                    print(f"   Input data device: {input_ids.device}")
                    print(f"   Attention mask device: {attention_mask.device}")
                    print(f"   Label device: {labels.device}")
                
                
                optimizer.zero_grad()
                
                
                outputs = self.model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
                loss = outputs['loss']
                logits = outputs['logits']
                
                
                loss.backward()
                
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                optimizer.step()
                
                
                total_loss += loss.item()
                
                
                preds = torch.argmax(logits, dim=1).cpu().numpy()
                all_preds.extend(preds)
                all_labels.extend(labels.cpu().numpy())
            
            
            accuracy = accuracy_score(all_labels, all_preds)
            f1 = f1_score(all_labels, all_preds, average='binary')
            
           
            avg_loss = total_loss / len(dataloader)

            print(f"Epoch {epoch+1}/{epochs}, Loss: {avg_loss:.4f}, Accuracy: {accuracy:.4f}, F1: {f1:.4f}")
        

        checkpoint_path = os.path.abspath(self.checkpoint_dir)
        os.makedirs(checkpoint_path, exist_ok=True)

        model_save_path = os.path.join(checkpoint_path, "pytorch_model.bin")
        torch.save(self.model.state_dict(), model_save_path)
        self.tokenizer.save_pretrained(checkpoint_path)

        print(f"BERT reward model has been saved to: {model_save_path}")
        
        return self.model
    
    def predict(self, questions, thinking_processes, batch_size=None):
        
        batch_size = batch_size if batch_size is not None else self.batch_size
        
        
        dummy_labels = [0] * len(questions)  
        dataset = ThinkingQualityDataset(
            questions, 
            thinking_processes, 
            dummy_labels, 
            self.tokenizer, 
            max_length=self.max_length
        )
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
        
        
        self.model.eval()
        all_preds = []
        
        with torch.no_grad():
            for batch in dataloader:
                input_ids = batch["input_ids"].to(self.device)
                attention_mask = batch["attention_mask"].to(self.device)
                
                outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)
                logits = outputs['logits']
                
                preds = torch.argmax(logits, dim=1).cpu().numpy()
                all_preds.extend(preds)
        
        return all_preds
    
    def predict_proba(self, questions, thinking_processes, batch_size=None):
        
        if batch_size is None:
            batch_size = self.batch_size
            
        
        if not hasattr(self, 'model') or self.model is None:
            self.load_model()  
            if not hasattr(self, 'model') or self.model is None:
                print("BERT model failed to load, cannot make predictions")
                return [0.5] * len(questions)  
        

        if torch.cuda.is_available() and str(next(self.model.parameters()).device) == 'cpu':
 
            try:
                self.model = self.model.to('cuda')
                self.device = 'cuda'
                print(f"Moved the model to device: {next(self.model.parameters()).device}")
            except Exception as e:
                print(f"Error moving model to GPU: {e}")
        
        device = next(self.model.parameters()).device
        print(f"Model device during prediction: {device}")
        
        
        dataset = ThinkingQualityDataset(
            questions=questions,
            thinking_processes=thinking_processes,
            labels=[0] * len(questions), 
            tokenizer=self.tokenizer,
            max_length=self.max_length
        )
        
        
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
        
        
        self.model.eval()
        all_probs = []
        
        
        with torch.no_grad():
            for batch in dataloader:
                
                input_ids = batch['input_ids'].to(device)
                attention_mask = batch['attention_mask'].to(device)
                
                
                if batch is next(iter(dataloader)):
                    print(f"【Prediction data device check】Input:{input_ids.device}, Mask:{attention_mask.device}")
                
                
                outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)
                logits = outputs['logits']
                
                
                probs = torch.softmax(logits, dim=1)
                
                
                positive_probs = probs[:, 1].cpu().numpy().tolist()
                all_probs.extend(positive_probs)
        
        return all_probs
    
    def load_model(self, model_path=None):

        if model_path is None:
            model_path = self.checkpoint_dir
        
        try:
            
            model_file = os.path.join(model_path, "pytorch_model.bin")
            if os.path.exists(model_file):
               
                self.model = BertMeanMaxPoolingClassifier(
                    bert_model_path=self.model_name,
                    dropout_rate=self.dropout_rate
                )
                
                
                self.model.load_state_dict(torch.load(model_file, map_location=self.device))
                self.model.to(self.device)
                
                
                if os.path.exists(os.path.join(model_path, "vocab.txt")):
                    self.tokenizer = BertTokenizer.from_pretrained(model_path)
                
                return True
            else:
                
                print(f"Model file does not exist: {model_file}")
                return False
        except Exception as e:
            print(f"Error loading model: {str(e)}")
            return False 
