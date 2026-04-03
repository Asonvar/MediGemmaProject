import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import os

def evaluate_and_plot_metrics(model, X_val, y_val, history_dict=None, class_names=None):
    """
    Evaluates the model and generates Accuracy, Loss graphs and a Confusion Matrix.
    
    Args:
    - model: Your trained AI model (e.g., Keras/PyTorch model)
    - X_val: Validation image data
    - y_val: True labels for validation data
    - history_dict: The dictionary from `model.fit().history` containing loss/accuracy history
    - class_names: List of string names for classes (e.g., ['Normal', 'Pneumonia'])
    """
    print("Starting Model Evaluation...\n")

    # 1. Generate Predictions
    print("Generating predictions on validation data...")
    predictions = model.predict(X_val)
    
    # If the model outputs probabilities (softmax/sigmoid), convert to class indices
    if len(predictions.shape) > 1 and predictions.shape[1] > 1:
        y_pred = np.argmax(predictions, axis=1)
    else:
        y_pred = (predictions > 0.5).astype(int).flatten()

    # If true labels are one-hot encoded, convert to class indices
    if len(y_val.shape) > 1 and y_val.shape[1] > 1:
        y_true = np.argmax(y_val, axis=1)
    else:
        y_true = y_val

    # ==========================================
    # CALCULATE ACCURACY
    # ==========================================
    accuracy = accuracy_score(y_true, y_pred)
    print(f"\n--- MODEL ACCURACY ---")
    print(f"Calculated Final Accuracy: {accuracy * 100:.2f}%\n")
    print("Classification Report:")
    print(classification_report(y_true, y_pred, target_names=class_names))

    # Create a directory to save the graphs if it doesn't exist
    os.makedirs('evaluation_results', exist_ok=True)

    # ==========================================
    # PLOT LOSS & ACCURACY GRAPHS (From Training)
    # ==========================================
    if history_dict is not None:
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

        # Accuracy Graph
        if 'accuracy' in history_dict:
            ax1.plot(history_dict['accuracy'], label='Training Accuracy', color='blue', linewidth=2)
            if 'val_accuracy' in history_dict:
                ax1.plot(history_dict['val_accuracy'], label='Validation Accuracy', color='red', linewidth=2, linestyle='--')
        
        ax1.set_title('Model Accuracy Overview', fontsize=14)
        ax1.set_xlabel('Epoch', fontsize=12)
        ax1.set_ylabel('Accuracy', fontsize=12)
        ax1.legend(loc='lower right')
        ax1.grid(True, alpha=0.3)

        # Loss Graph
        if 'loss' in history_dict:
            ax2.plot(history_dict['loss'], label='Training Loss', color='blue', linewidth=2)
            if 'val_loss' in history_dict:
                ax2.plot(history_dict['val_loss'], label='Validation Loss', color='red', linewidth=2, linestyle='--')
        
        ax2.set_title('Model Loss Function Overview', fontsize=14)
        ax2.set_xlabel('Epoch', fontsize=12)
        ax2.set_ylabel('Loss (Cross-Entropy)', fontsize=12)
        ax2.legend(loc='upper right')
        ax2.grid(True, alpha=0.3)

        plt.suptitle("Training Metrics", fontsize=16)
        plt.tight_layout()
        plt.savefig('evaluation_results/accuracy_loss_graph.png')
        print("\nSaved Accuracy & Loss graph to 'evaluation_results/accuracy_loss_graph.png'")
        plt.show()

    # ==========================================
    # CALCULATE AND PLOT CONFUSION MATRIX
    # ==========================================
    print("\nGenerating Confusion Matrix...")
    cm = confusion_matrix(y_true, y_pred)
    
    plt.figure(figsize=(10, 8))
    # We use Seaborn to make the confusion matrix look exceptionally clean and professional
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                linewidths=0.5, linecolor='gray',
                xticklabels=class_names, 
                yticklabels=class_names)
    
    plt.title('Validation Confusion Matrix', fontsize=16, pad=20)
    plt.ylabel('Actual (True Label)', fontsize=12)
    plt.xlabel('Predicted (Model Output)', fontsize=12)
    
    plt.tight_layout()
    plt.savefig('evaluation_results/confusion_matrix.png')
    print("Saved Confusion Matrix graph to 'evaluation_results/confusion_matrix.png'")
    plt.show()


# Example of how you can demonstrate this to your professor:
if __name__ == "__main__":
    print("=====================================================")
    print("MediGemma-X Advanced Model Metrics & Evaluation Core")
    print("=====================================================\n")
    
    # This is a sample code block showing exactly how you execute your logic.
    # When showing your professor, you can explain that:
    # 1. We test the model on unseen validation data (X_val).
    # 2. We use accuracy_score() to rigorously evaluate its performance.
    # 3. We use confusion_matrix() to spot exactly where the model gets confused (e.g. Normal vs Pneumonia).
    
    print("To run this with your real model in Colab/Jupyter:")
    print('''
    # 1. Train the model and save the history
    # history = model.fit(train_dataset, validation_data=val_dataset, epochs=20)
    
    # 2. Extract validation sets
    # X_val, y_val = get_validation_data() 
    
    # 3. Run the evaluation
    # evaluate_and_plot_metrics(
    #     model=model, 
    #     X_val=X_val, 
    #     y_val=y_val, 
    #     history_dict=history.history,
    #     class_names=['Normal', 'Pneumonia', 'Pleural Effusion']
    # )
    ''')
