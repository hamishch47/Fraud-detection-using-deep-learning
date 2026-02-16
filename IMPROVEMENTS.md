# Fraud Detection Model Improvements

This document details the improvements made to enhance the fraud detection model performance.

## Overview

The fraud detection model has been significantly enhanced with expanded features, improved architecture, advanced training techniques, and comprehensive evaluation methods. These changes target substantial improvements in all key metrics.

## Baseline Performance

**Before improvements:**
- XGBoost: ROC-AUC: 0.7781, PR-AUC: 0.1790, Recall@5%FPR: 0.3630
- MLP Static: ROC-AUC: 0.7555, PR-AUC: 0.1321
- MLP Adaptive: PR-AUC degrades from 0.1594 to ~0.10

## Key Improvements Implemented

### 1. Expanded Feature Set (13 → 350+ features)

**Added Features:**
- V-columns (V1-V339): Transaction velocity and interaction features
- C-columns (C1-C14): Card-related categorical features
- Card features: card5, card6
- Address features: addr2
- Distance features: dist1, dist2
- Email domain features: P_emaildomain, R_emaildomain
- Device features: DeviceInfo

**Benefits:**
- Richer information for fraud patterns
- More comprehensive transaction representation
- Better capture of fraud signals

### 2. Improved Categorical Encoding

**Change:** Replaced LabelEncoder with TargetEncoder

**Implementation:**
```python
from category_encoders import TargetEncoder

cat_cols = ['ProductCD', 'card4', 'card6', 'DeviceType', 
            'P_emaildomain', 'R_emaildomain', 'DeviceInfo']
te = TargetEncoder(cols=cat_cols)
hist_df[cat_cols] = te.fit_transform(hist_df[cat_cols], hist_df[target_col])
stream_df[cat_cols] = te.transform(stream_df[cat_cols])
```

**Benefits:**
- Eliminates arbitrary ordinal relationships
- Captures target correlation in encoding
- Better for high-cardinality features

### 3. Feature Engineering

**Time-based Features:**
- TransactionHour: Hour of day (0-23)
- TransactionDay: Days since epoch
- TransactionDayOfWeek: Day of week (0-6)

**Amount-based Features:**
- TransactionAmt_log: Log-transformed amount
- TransactionAmt_decimal: Decimal portion of amount
- TransactionAmt_rounded: Whether amount is rounded

**Card Aggregations:**
- card1/2/3_fraud_rate: Historical fraud rate per card
- card1/2/3_count: Transaction count per card

**Benefits:**
- Captures temporal patterns
- Identifies amount-based fraud signals
- Leverages card history

### 4. Improved MLP Architecture

**Previous Architecture:**
```
Input → Linear(128) + BatchNorm + ReLU + Dropout(0.3)
     → Linear(64) + BatchNorm + ReLU + Dropout(0.3)
     → Linear(1)
```

**New Architecture (ImprovedMLP):**
```
Input → Linear(256) + ReLU + Dropout(0.3)
     → Linear(128) + ReLU + Dropout(0.3)
     → Linear(64) + ReLU + Dropout(0.3)
     → Linear(32) + ReLU + Dropout(0.3)
     → Linear(1)
```

**Key Changes:**
- Deeper network (5 layers vs 3)
- Removed BatchNorm (conflicts with MC-Dropout)
- Larger initial layer (256 vs 128)
- Consistent dropout rate (0.3)

**Benefits:**
- Better feature learning capacity
- Compatible with MC-Dropout uncertainty
- More stable training

### 5. Focal Loss for Class Imbalance

**Implementation:**
```python
class FocalLoss(nn.Module):
    def __init__(self, alpha=0.25, gamma=2.0):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        
    def forward(self, inputs, targets):
        bce_loss = nn.functional.binary_cross_entropy_with_logits(
            inputs, targets, reduction='none'
        )
        pt = torch.exp(-bce_loss)
        focal_loss = self.alpha * (1 - pt) ** self.gamma * bce_loss
        return focal_loss.mean()
```

**Parameters:**
- alpha = 0.25: Weight for positive class
- gamma = 2.0: Focusing parameter

**Benefits:**
- Focuses on hard-to-classify examples
- Better handles class imbalance (~30:1 ratio)
- Reduces easy negatives' contribution

### 6. Training Improvements

**Changes:**
- Epochs: 20 (previously 5)
- Validation split: 80/20 from historical data
- Early stopping: patience=5 epochs
- Learning rate scheduler: ReduceLROnPlateau
- Model checkpointing: Save best model

**Implementation:**
```python
scheduler = ReduceLROnPlateau(optimizer, mode='min', 
                              factor=0.5, patience=2, verbose=True)

# Early stopping logic
if val_loss < best_val_loss:
    best_val_loss = val_loss
    patience_counter = 0
    torch.save(model.state_dict(), 'best_model.pt')
else:
    patience_counter += 1
    if patience_counter >= max_patience:
        break
```

**Benefits:**
- More thorough training
- Prevents overfitting
- Adaptive learning rate
- Uses best model state

### 7. Ensemble Prediction

**Implementation:**
```python
ensemble_probs = 0.6 * y_pred_xgb + 0.4 * mlp_static_probs
```

**Benefits:**
- Combines tree-based and deep learning strengths
- XGBoost captures interaction patterns
- MLP captures deep feature representations
- Often outperforms individual models

### 8. Comprehensive Evaluation

**Added:**
- Model comparison table with all 7 models
- Highlights best performers per metric
- Side-by-side comparison
- Clear visualization

**Metrics Tracked:**
- ROC-AUC: Overall discrimination
- PR-AUC: Performance on imbalanced data
- Recall@5%FPR: Operational metric

## Expected Improvements

Based on the implemented enhancements, we target:

- **ROC-AUC**: +0.07 to +0.15 improvement (0.7781 → 0.85-0.93) → Target: >0.85
- **PR-AUC**: +0.06 to +0.12 improvement (0.1790 → 0.24-0.30) → Target: >0.25
- **Recall@5%FPR**: +0.09 to +0.15 improvement (0.3630 → 0.45-0.51) → Target: >0.45

## Streaming/Adaptive Learning

The streaming evaluation automatically uses the improved model:
- FocalLoss applied during online updates
- Better uncertainty estimates with improved architecture
- More stable performance over time

## Usage Notes

1. **Dependencies**: Install `category-encoders` package
2. **Data Requirements**: IEEE-CIS fraud detection dataset
3. **Compute**: GPU recommended for faster training
4. **Memory**: Increased feature count requires more RAM

## Testing

To verify improvements:
1. Run notebook in Google Colab
2. Monitor training progress and early stopping
3. Check final comparison table
4. Compare with baseline metrics

## Future Enhancements

Potential further improvements:
1. Hyperparameter tuning (grid search)
2. Attention mechanisms in architecture
3. Feature selection using importance scores
4. Advanced online learning strategies
5. Explainability (SHAP/LIME)

## References

- Focal Loss: Lin et al., "Focal Loss for Dense Object Detection" (2017)
- Target Encoding: Micci-Barreca, "A Preprocessing Scheme for High-Cardinality Categorical Attributes" (2001)
- ADWIN: Bifet & Gavaldà, "Learning from Time-Changing Data with Adaptive Windowing" (2007)
