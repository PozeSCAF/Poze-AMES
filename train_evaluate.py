import logging 
import torch
from box_wrap import create_boxed_text
from sklearn.metrics import precision_score, recall_score, f1_score, confusion_matrix, classification_report, roc_auc_score, roc_curve,accuracy_score
log_filename = "test.log"
logging.basicConfig(
    level=logging.DEBUG,
    format="%(asctime)s - %(filename)s:%(lineno)d - %(funcName)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    handlers=[
        logging.FileHandler(log_filename, mode='a'),
        logging.StreamHandler()
    ]
)
class ModelTrainer:
    def __init__(self, model, criterion, optimizer, device):
        self.model = model.to(device)
        self.criterion = criterion
        self.optimizer = optimizer
        self.device = device
        model_str = str(self.model)
        boxed_model_str = create_boxed_text(model_str)
        logging.info('\nModel Architecture:\n%s', boxed_model_str)

    def train(self, dataloader, num_epochs=120, patience=10, min_delta=0.001):
        best_loss = float('inf')
        patience_counter = 0
        early_stop = False
        final_true_labels = []
        final_predicted_labels = []
        train_losses = []
        train_accuracies = []
        best_model = None  

        for epoch in range(num_epochs):
            if early_stop:
                print(f"Early stopping triggered at epoch {epoch}")
                break

            self.model.train()
            total_loss = 0
            correct = 0
            total = 0
            true_labels = []
            predicted_labels = []

            for batch_graph_data, fingerprint_data in dataloader:
                # print(f"Batch size: {batch_graph_data.x.shape[0]}")
                batch_graph_data = batch_graph_data.to(self.device)
                fingerprint_data = fingerprint_data.to(self.device)
                self.optimizer.zero_grad()
                out = self.model(batch_graph_data, fingerprint_data)
                loss = self.criterion(out.squeeze(), batch_graph_data.y.float())
                loss.backward()
                self.optimizer.step()
                total_loss += loss.item()
                pred = (torch.sigmoid(out).squeeze() > 0.5).int()
                true_labels.extend(batch_graph_data.y.cpu().numpy())
                predicted_labels.extend(pred.cpu().numpy())
                correct += (pred == batch_graph_data.y.float()).sum().item()
                total += batch_graph_data.y.size(0)

            avg_loss = total_loss / len(dataloader)
            train_accuracy = correct / total
            train_losses.append(avg_loss)
            train_accuracies.append(train_accuracy)

            print(f'Epoch {epoch + 1}, Loss: {avg_loss:.4f}, Accuracy: {train_accuracy:.4f}')
            if avg_loss < best_loss - min_delta:  
                best_loss = avg_loss
                patience_counter = 0
                best_model = self.model  # Store the best model
                final_true_labels = true_labels
                final_predicted_labels = predicted_labels
            else: 
                patience_counter += 1
                if patience_counter >= patience:
                    early_stop = True
                    print(f"\nEarly stopping triggered. Best loss: {best_loss:.4f}")
                    break
        
        # If early stopping is not triggered, best_model should be the last trained model
        if not early_stop:
            best_model = self.model 
        

        accuracy = accuracy_score(final_true_labels, final_predicted_labels)
        precision = precision_score(final_true_labels, final_predicted_labels)
        recall = recall_score(final_true_labels, final_predicted_labels)
        f1 = f1_score(final_true_labels, final_predicted_labels)
        conf_matrix = confusion_matrix(final_true_labels, final_predicted_labels)
        tn, fp, fn, tp = conf_matrix.ravel()
        specificity_class_0 = tp / (tp + fn)  
        specificity_class_1 = tn / (tn + fp) 
        class_report = classification_report(final_true_labels, final_predicted_labels, target_names=["class_0", "class_1"])

        print("\nFinal Results:")
        print(f'Accuracy: {accuracy:.4f}')
        print(f'Precision: {precision:.4f}')
        print(f'Recall: {recall:.4f}')
        print(f'Specificity for class_0: {specificity_class_0:.4f}')
        print(f'Specificity for class_1: {specificity_class_1:.4f}')
        print(f'F1 Score: {f1:.4f}')
        print(f'Confusion Matrix:\n{conf_matrix}')
        print(f'\nClassification Report:\n{class_report}')
        logging.info("Training Completed.\n" + create_boxed_text(f"Loss: {best_loss:.4f} and Accuracy: {accuracy:.4f}"))
       
        
        return best_model, train_losses, train_accuracies, final_true_labels, final_predicted_labels

    def evaluate(self, best_model, dataloader):
        best_model.eval()
        total_loss = 0
        correct = 0
        total = 0
        all_preds = []
        all_targets = []
        all_probs = []

        with torch.no_grad():
            for batch_graph_data, fingerprint_data in dataloader:
                batch_graph_data = batch_graph_data.to(self.device)
                fingerprint_data = fingerprint_data.to(self.device)
                out = best_model(batch_graph_data, fingerprint_data)
                loss = self.criterion(out.squeeze(), batch_graph_data.y.float())
                total_loss += loss.item()
                
                pred = (torch.sigmoid(out).squeeze() > 0.5).int()
                correct += (pred == batch_graph_data.y.float()).sum().item()
                total += batch_graph_data.y.size(0)
                
                all_probs.extend(out.squeeze().cpu().numpy())
                all_preds.extend(pred.cpu().numpy())
                all_targets.extend(batch_graph_data.y.cpu().numpy())

        avg_loss = total_loss / len(dataloader)
        print(f'Loss: {avg_loss:.4f}')
        accuracy = correct / total
        print(f'Accuracy: {accuracy:.4f}')
        conf_matrix = confusion_matrix(all_targets, all_preds)
        print(conf_matrix)
        tn, fp, fn, tp = conf_matrix.ravel()
        specificity_class_0 = tp / (tp + fn)  
        specificity_class_1 = tn / (tn + fp)  
        print(f'Specificity class_0 : {specificity_class_0:.4f}')
        print(f'Specificity class_1 : {specificity_class_1:.4f}')
        
        class_report = classification_report(all_targets, all_preds)
        print(class_report)
        roc_auc = roc_auc_score(all_targets, all_probs)
        fpr, tpr, thresholds = roc_curve(all_targets, all_probs)
        logging.info("Evaluation Completed.\n"+create_boxed_text(f"Loss:{avg_loss:.4f} and Accuracy:{accuracy:.4f}"))
        logging.info("\nClassification Report:\n"+create_boxed_text(class_report))
                    
        
        # return all_probs,all_preds, all_targets,avg_loss, accuracy, conf_matrix, class_report, roc_auc, fpr, tpr
        return avg_loss, accuracy, conf_matrix, class_report, roc_auc, fpr, tpr
