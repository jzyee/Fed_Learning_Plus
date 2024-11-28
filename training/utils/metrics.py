import matplotlib.pyplot as plt
import os
import csv

class TrainingMetrics:
    def __init__(self):
        self.global_accuracies = []
        self.class_boundaries = []  # Mark where new classes are added
        self.classes_seen = []      # Track number of classes seen at each point
        self.losses = []
        
    def update(self, accuracy, classes_learned, loss=None, new_classes=False):
        """
        Update training metrics
        Args:
            accuracy: Global model accuracy
            classes_learned: Total number of classes learned so far
            loss: Training loss (optional)
            new_classes: Boolean indicating if new classes were added
        """
        self.global_accuracies.append(accuracy)
        self.classes_seen.append(classes_learned)
        if loss is not None:
            self.losses.append(loss)
        if new_classes:
            self.class_boundaries.append(len(self.global_accuracies) - 1)
            
    def plot_metrics(self, output_dir):
        """
        Plot training metrics as separate images for accuracy and loss
        
        Args:
            output_dir: Directory path where plot images will be saved
        """
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
            print(f"Directory created: {output_dir}")
        else:
            print(f"Directory already exists: {output_dir}")
        
        
        
        # Create metrics CSV file path
        metrics_csv_path = os.path.join(output_dir, "training_metrics.csv")
        
        # Write classes seen and accuracies to CSV
        with open(metrics_csv_path, "w", newline="") as csvfile:
            writer = csv.writer(csvfile)
            # Write header row
            writer.writerow(["Classes Learned", "Global Accuracy (%)"])
            # Write data rows
            for classes, acc in zip(self.classes_seen, self.global_accuracies):
                writer.writerow([classes, acc])
                
        print(f"Training metrics saved to: {metrics_csv_path}")

        # Plot accuracy
        plt.figure(figsize=(12, 5))
        plt.plot(self.classes_seen, self.global_accuracies, label="Global Accuracy", marker='^', markersize=10, linewidth=2)
        plt.title("Global Model Accuracy vs Classes Learned")
        plt.xlabel("Number of Classes Learned")
        plt.ylabel("Accuracy (%)")
        
        # Add vertical lines for class boundaries
        for boundary in self.class_boundaries:
            plt.axvline(x=self.classes_seen[boundary], color="r", 
                       linestyle="--", alpha=0.3,
                       label="New Classes Added" if boundary == self.class_boundaries[0] else "")
        plt.grid(True)
        plt.legend()
        plt.tight_layout()
        plt.savefig(f"{output_dir}/accuracy_metrics.png")
        plt.savefig(f"accuracy_metrics.png")
        plt.close()
        
        # Plot loss if available
        if self.losses:
            plt.figure(figsize=(12, 5))
            plt.plot(self.classes_seen, self.losses, label="Training Loss")
            plt.title("Training Loss vs Classes Learned")
            plt.xlabel("Number of Classes Learned")
            plt.ylabel("Loss")
            
            for boundary in self.class_boundaries:
                plt.axvline(x=self.classes_seen[boundary], color="r", 
                           linestyle="--", alpha=0.3,
                           label="New Classes Added" if boundary == self.class_boundaries[0] else "")
            plt.grid(True)
            plt.legend()
            plt.tight_layout()
            plt.savefig(f"{output_dir}/loss_metrics.png")
            plt.close()