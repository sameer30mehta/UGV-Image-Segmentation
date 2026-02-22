# Model Persistence System - Complete Explanation

## Overview

The fine-tuned model persistence system ensures that when a user trains a custom model, it **automatically becomes active** and **persists across page reloads** until explicitly reset.

## How It Works

### 1. **Fine-Tuning Flow**

```
User uploads images → Trains model → Model saved → Model loaded → Model set as active
                                         ↓
                                   user_id saved to localStorage
```

**Backend** ([backend/app.py](backend/app.py#L540-L590)):
```python
POST /api/finetune
├── Saves images/masks to temp directory
├── Calls run_finetune_pipeline(dataset_path, user_id)
├── Returns: { user_id, metrics, model_path }
└── Model file: models/user_<id>.pth
```

**Frontend** ([frontend/app.js](frontend/app.js#L1150-L1180)):
```javascript
1. Fine-tuning completes
2. Save user_id to localStorage: 'finetuned_model_id'
3. Call POST /api/load-user-model with user_id
4. Server loads custom model and sets it as active
5. Update UI to show "Custom Model" in status
```

### 2. **Automatic Model Loading on Page Load**

**On Every Page Load** ([frontend/app.js](frontend/app.js#L41)):
```javascript
DOMContentLoaded:
  ├── checkHealth()        // Check server status
  ├── loadUserModelIfExists()  // ← Key function
  └── Update UI indicators
```

**loadUserModelIfExists()** ([frontend/app.js](frontend/app.js#L288-L309)):
```javascript
1. Check localStorage for 'finetuned_model_id'
2. If found:
   - Call POST /api/load-user-model
   - Server loads the .pth file
   - Sets it as active model
3. If model file missing:
   - Clear localStorage
   - Fall back to base model
```

### 3. **Backend Model Registry**

**New Methods Added** ([backend/models.py](backend/models.py#L137-L168)):

```python
class ModelRegistry:
    def load_user_model(self, user_id, weights_path):
        """Load custom fine-tuned model"""
        model_key = f"user_{user_id}"
        
        # Create FPN model with same architecture
        model = smp.FPN(encoder='mit_b3', ...)
        
        # Load user weights
        state_dict = torch.load(weights_path)
        model.load_state_dict(state_dict)
        
        # Register and activate
        self.models[model_key] = model
        self.active_model_key = model_key  # ← Sets as active
        
        return model_key
```

### 4. **API Endpoints**

**Load User Model** ([backend/app.py](backend/app.py#L220-L237)):
```python
POST /api/load-user-model
Input:  user_id (form data)
Output: { status, model_key, active_model }
Action: Loads models/user_<id>.pth and sets as active
```

**Reset to Base Model** ([backend/app.py](backend/app.py#L240-L251)):
```python
POST /api/reset-to-base-model
Output: { status, active_model: "mit_b3" }
Action: Switches back to base model
```

### 5. **UI Indicators**

**Status Bar** ([frontend/app.js](frontend/app.js#L256-L275)):
```javascript
checkHealth():
  ├── If active_model starts with "user_":
  │   ├── Display: "Custom Model" (instead of "MiT-B3")
  │   ├── Color: Green (#4ade80)
  │   └── Tooltip: "Using your fine-tuned model"
  └── Else: Show base model name
```

**Visual Feedback**:
- Navbar status: `CPU · Custom Model` (in green)
- Hero section: "Active Model: Custom Model"
- All inference automatically uses custom model

### 6. **Persistence Lifecycle**

```
┌─────────────────────────────────────────────────────┐
│ 1. User Fine-Tunes Model                           │
│    └─> models/user_abc123.pth created              │
│    └─> localStorage: 'finetuned_model_id' = abc123 │
│    └─> Server: active_model = user_abc123          │
└─────────────────────────────────────────────────────┘
                        ↓
┌─────────────────────────────────────────────────────┐
│ 2. User Goes to "Analyze" Tab                      │
│    └─> Custom model already active                 │
│    └─> All inference uses user_abc123.pth          │
└─────────────────────────────────────────────────────┘
                        ↓
┌─────────────────────────────────────────────────────┐
│ 3. User Refreshes Page / Reopens Browser           │
│    └─> checkHealth() runs                          │
│    └─> loadUserModelIfExists() runs                │
│    └─> Reads 'finetuned_model_id' from localStorage│
│    └─> POST /api/load-user-model                   │
│    └─> Server loads user_abc123.pth                │
│    └─> active_model = user_abc123                  │
│    └─> UI shows "Custom Model"                     │
└─────────────────────────────────────────────────────┘
                        ↓
┌─────────────────────────────────────────────────────┐
│ 4. User Clicks "Reset to Base Model"               │
│    └─> POST /api/reset-to-base-model               │
│    └─> localStorage cleared                        │
│    └─> active_model = mit_b3                       │
│    └─> UI shows "MiT-B3"                           │
└─────────────────────────────────────────────────────┘
```

## Key Features

### ✅ Automatic Activation
After fine-tuning, the custom model is **immediately loaded and set as active**.

### ✅ Persistent Across Reloads
User's model preference survives:
- Page refreshes
- Browser restarts
- Tab closes/reopens

### ✅ Seamless Switching
- Custom model → All inference automatically uses it
- No manual selection required
- Works across all features (Analyze, Video, etc.)

### ✅ Easy Reset
Users can switch back to base model anytime via "Reset to Base Model" button.

### ✅ Graceful Degradation
If model file is deleted:
- localStorage is cleared
- Falls back to base model
- No errors shown to user

## Storage Details

**localStorage** (Browser):
```javascript
Key: 'finetuned_model_id'
Value: 'abc12345'  // 8-character UUID
Scope: Per-domain (persists across sessions)
```

**File System** (Server):
```
models/
├── user_abc12345.pth  (Custom model 1)
├── user_def67890.pth  (Custom model 2)
└── ...
```

**Runtime Memory** (Server):
```python
registry.models = {
    'mit_b3': <base_model>,
    'user_abc12345': <custom_model>  # ← Loaded on demand
}
registry.active_model_key = 'user_abc12345'
```

## Testing the Flow

1. **Fine-tune a model**:
   - Upload 5 images + masks
   - Click "Start Fine-Tuning"
   - Wait ~10-20 seconds
   - See "Custom Model" in green status

2. **Verify persistence**:
   - Click "Analyze" tab
   - Upload an image → runs inference with custom model
   - Refresh page → custom model still active
   - Close tab, reopen → custom model still active

3. **Reset to base**:
   - Click "Reset to Base Model"
   - Status changes to "MiT-B3"
   - Inference uses base model

## Code Locations

| Component | File | Lines |
|-----------|------|-------|
| Fine-tuning pipeline | `backend/finetune.py` | All |
| Model loading | `backend/models.py` | 137-168 |
| API endpoints | `backend/app.py` | 220-251, 540-590 |
| Auto-load on start | `frontend/app.js` | 41, 288-309 |
| UI indicators | `frontend/app.js` | 256-286 |
| Results UI | `frontend/index.html` | 650-670 |

## Summary

The system implements a **complete model lifecycle**:
1. ✅ Train → Save → Load → Activate (automatic)
2. ✅ Persist across sessions (localStorage)  
3. ✅ Visual indicators (green "Custom Model")
4. ✅ Reset option (back to base)
5. ✅ Graceful error handling

**Result**: Users get a personalized model that "just works" everywhere, without manual configuration.
