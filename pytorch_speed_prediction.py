import pandas as pd
import numpy as np
import ast
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import matplotlib.pyplot as plt

class SimpleSpeedPredictor(nn.Module):
    def __init__(self, input_size=800):
        super(SimpleSpeedPredictor, self).__init__()
        self.network = nn.Sequential(
            nn.Linear(input_size, 512),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 1)
        )
    
    def forward(self, x):
        return self.network(x)

def load_and_preprocess_data():
    """Load and preprocess the tennis data."""
    print("Loading data...")
    df = pd.read_csv('/Users/jasonwang/Library/CloudStorage/OneDrive-UniversityofVirginia/Coding/tennis/data/full/final.csv')
    
    print(f"Original dataset shape: {df.shape}")
    
    # Remove rows with NaN or empty values in 'joint_angles_100'
    df_clean = df[df['joint_angles_100'].notna() & (df['joint_angles_100'] != '') & df['Speed_MPH'].notna()].copy()
    print(f"After cleaning: {df_clean.shape}")
    
    # Extract target variable
    y = df_clean['Speed_MPH'].values
    
    # Convert joint_angles_100 to numpy array
    print("Processing joint angles...")
    x = np.stack(df_clean['joint_angles_100'].apply(
        lambda s: np.array(ast.literal_eval(s)).reshape(800, 1)
    ).values)
    
    print(f"X shape: {x.shape}")  # Should be (n, 800, 1)
    print(f"Y shape: {y.shape}")  # Should be (n,)
    print(f"Speed range: {y.min():.2f} - {y.max():.2f} mph")
    
    # Flatten for fully connected network
    x_flattened = x.reshape(x.shape[0], 800)
    
    return x_flattened, y

def train_model(model, train_loader, val_loader, num_epochs=100, learning_rate=0.001):
    """Train the PyTorch model."""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=10, factor=0.5)
    
    train_losses = []
    val_losses = []
    best_val_loss = float('inf')
    patience = 20
    patience_counter = 0
    
    print(f"Training on device: {device}")
    
    for epoch in range(num_epochs):
        # Training phase
        model.train()
        train_loss = 0.0
        
        for batch_x, batch_y in train_loader:
            batch_x, batch_y = batch_x.to(device), batch_y.to(device)
            
            optimizer.zero_grad()
            outputs = model(batch_x).squeeze()
            loss = criterion(outputs, batch_y)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
        
        # Validation phase
        model.eval()
        val_loss = 0.0
        
        with torch.no_grad():
            for batch_x, batch_y in val_loader:
                batch_x, batch_y = batch_x.to(device), batch_y.to(device)
                outputs = model(batch_x).squeeze()
                loss = criterion(outputs, batch_y)
                val_loss += loss.item()
        
        train_loss /= len(train_loader)
        val_loss /= len(val_loader)
        
        train_losses.append(train_loss)
        val_losses.append(val_loss)
        
        scheduler.step(val_loss)
        
        # Early stopping
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            # Save best model
            torch.save(model.state_dict(), 'best_model.pth')
        else:
            patience_counter += 1
        
        if epoch % 10 == 0:
            print(f'Epoch [{epoch}/{num_epochs}], Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}')
        
        if patience_counter >= patience:
            print(f"Early stopping at epoch {epoch}")
            break
    
    # Load best model
    model.load_state_dict(torch.load('best_model.pth'))
    
    return train_losses, val_losses

def evaluate_model(model, test_loader):
    """Evaluate the model and return predictions."""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.eval()
    
    all_predictions = []
    all_targets = []
    
    with torch.no_grad():
        for batch_x, batch_y in test_loader:
            batch_x, batch_y = batch_x.to(device), batch_y.to(device)
            outputs = model(batch_x).squeeze()
            
            all_predictions.extend(outputs.cpu().numpy())
            all_targets.extend(batch_y.cpu().numpy())
    
    return np.array(all_predictions), np.array(all_targets)

def plot_results(train_losses, val_losses, y_true, y_pred):
    """Plot training history and predictions."""
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 10))
    
    # Plot training history
    ax1.plot(train_losses, label='Training Loss')
    ax1.plot(val_losses, label='Validation Loss')
    ax1.set_title('Training History')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.legend()
    ax1.grid(True)
    
    # Plot predictions vs actual
    ax2.scatter(y_true, y_pred, alpha=0.6, s=20)
    min_val = min(y_true.min(), y_pred.min())
    max_val = max(y_true.max(), y_pred.max())
    ax2.plot([min_val, max_val], [min_val, max_val], 'r--', linewidth=2)
    ax2.set_xlabel('Actual Speed (mph)')
    ax2.set_ylabel('Predicted Speed (mph)')
    ax2.set_title('Predictions vs Actual')
    ax2.grid(True)
    
    # Plot residuals
    residuals = y_true - y_pred
    ax3.scatter(y_pred, residuals, alpha=0.6, s=20)
    ax3.axhline(y=0, color='r', linestyle='--')
    ax3.set_xlabel('Predicted Speed (mph)')
    ax3.set_ylabel('Residuals')
    ax3.set_title('Residual Plot')
    ax3.grid(True)
    
    # Plot histogram of residuals
    ax4.hist(residuals, bins=30, alpha=0.7)
    ax4.set_xlabel('Residuals')
    ax4.set_ylabel('Frequency')
    ax4.set_title('Residual Distribution')
    ax4.grid(True)
    
    plt.tight_layout()
    plt.savefig('pytorch_results.png', dpi=300, bbox_inches='tight')
    plt.show()

def main():
    # Load and preprocess data
    X, y = load_and_preprocess_data()
    
    # Split the data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    
    # Scale the features
    print("Scaling features...")
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Convert to PyTorch tensors
    X_train_tensor = torch.FloatTensor(X_train_scaled)
    y_train_tensor = torch.FloatTensor(y_train)
    X_test_tensor = torch.FloatTensor(X_test_scaled)
    y_test_tensor = torch.FloatTensor(y_test)
    
    # Create data loaders
    train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
    test_dataset = TensorDataset(X_test_tensor, y_test_tensor)
    
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)
    
    # Split training data for validation
    train_size = int(0.8 * len(train_dataset))
    val_size = len(train_dataset) - train_size
    train_subset, val_subset = torch.utils.data.random_split(train_dataset, [train_size, val_size])
    
    train_loader = DataLoader(train_subset, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_subset, batch_size=32, shuffle=False)
    
    # Create model
    model = SimpleSpeedPredictor(input_size=800)
    print("Model architecture:")
    print(model)
    
    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"\nTotal parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")
    
    # Train model
    print("\nTraining model...")
    train_losses, val_losses = train_model(model, train_loader, val_loader)
    
    # Evaluate model
    print("\nEvaluating model...")
    y_pred, y_true = evaluate_model(model, test_loader)
    
    # Calculate metrics
    mse = mean_squared_error(y_true, y_pred)
    mae = mean_absolute_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)
    rmse = np.sqrt(mse)
    
    print("\n" + "="*50)
    print("MODEL PERFORMANCE METRICS")
    print("="*50)
    print(f"Mean Squared Error (MSE): {mse:.4f}")
    print(f"Root Mean Squared Error (RMSE): {rmse:.4f}")
    print(f"Mean Absolute Error (MAE): {mae:.4f}")
    print(f"R² Score: {r2:.4f}")
    print(f"Explained Variance: {r2*100:.2f}%")
    
    print("\n" + "="*50)
    print("ADDITIONAL INSIGHTS")
    print("="*50)
    print(f"Average actual speed: {y_true.mean():.2f} mph")
    print(f"Average predicted speed: {y_pred.mean():.2f} mph")
    print(f"Speed standard deviation (actual): {y_true.std():.2f} mph")
    print(f"Speed standard deviation (predicted): {y_pred.std():.2f} mph")
    
    # Sample predictions
    print("\nSample predictions:")
    for i in range(min(10, len(y_true))):
        print(f"Actual: {y_true[i]:.2f} mph, Predicted: {y_pred[i]:.2f} mph, Error: {abs(y_true[i] - y_pred[i]):.2f} mph")
    
    # Plot results
    plot_results(train_losses, val_losses, y_true, y_pred)
    
    # Save final model
    torch.save({
        'model_state_dict': model.state_dict(),
        'scaler': scaler,
        'model_architecture': 'SimpleSpeedPredictor',
        'input_size': 800
    }, 'tennis_speed_pytorch_model.pth')
    
    print("\nModel saved as 'tennis_speed_pytorch_model.pth'")
    
    return model, scaler

if __name__ == "__main__":
    model, scaler = main() 