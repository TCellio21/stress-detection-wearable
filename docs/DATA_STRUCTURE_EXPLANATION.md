# WESAD Data Structure and Data Loader Explanation

## Your Questions Answered

### 1. Does each subject have a pickle file that includes all of the data?

**YES!** Each subject has one main `.pkl` file (e.g., `S3.pkl`) that contains **synchronized, pre-processed sensor data** from both devices.

### File Structure per Subject:

```
S3/
├── S3.pkl                  ← Main file: ALL synchronized sensor data + labels
├── S3_readme.txt          ← Metadata: age, gender, height, weight, study notes
├── S3_quest.csv           ← Questionnaire responses (PANAS, STAI, SAM)
├── S3_respiban.txt        ← RespiBAN device calibration info
└── S3_E4_Data/            ← Raw CSV files from E4 (alternative format)
    ├── ACC.csv
    ├── BVP.csv
    ├── EDA.csv
    ├── TEMP.csv
    ├── HR.csv
    ├── IBI.csv
    ├── tags.csv
    └── info.txt
```

---

## 2. Does the PKL file include both chest and wrist data?

**YES!** The `.pkl` file contains data from **BOTH devices**:

### PKL File Internal Structure:

```python
data = {
    'subject': 'S3',           # Subject ID

    'signal': {
        'chest': {             # RespiBAN chest device (700 Hz)
            'ACC': array,      # 3-axis accelerometer (N × 3)
            'ECG': array,      # Electrocardiogram (N × 1)
            'EDA': array,      # Electrodermal activity (N × 1)
            'EMG': array,      # Electromyogram (N × 1)
            'Temp': array,     # Temperature (N × 1)
            'Resp': array      # Respiration (N × 1)
        },

        'wrist': {             # Empatica E4 wrist device (variable rates)
            'ACC': array,      # 3-axis accelerometer at 32 Hz (M × 3)
            'BVP': array,      # Blood volume pulse at 64 Hz (P × 1)
            'EDA': array,      # Skin conductance at 4 Hz (Q × 1)
            'TEMP': array      # Skin temperature at 4 Hz (R × 1)
        }
    },

    'label': array             # Ground truth labels at 700 Hz (N × 1)
}
```

**Key observations:**
- Chest signals: All sampled at **700 Hz**
- Wrist signals: Different sampling rates (32, 64, 4 Hz)
- Labels: Sampled at **700 Hz** (matches chest device)

---

## 3. How are different sampling rates accounted for?

This is a **critical** question! Here's how it works:

### The Sampling Rate Problem:

Different sensors sample at different frequencies:

| Device | Signal | Sampling Rate | Samples in 1 minute |
|--------|--------|---------------|---------------------|
| **Chest** | All signals | 700 Hz | 42,000 |
| **Chest** | Labels | 700 Hz | 42,000 |
| **Wrist** | BVP | 64 Hz | 3,840 |
| **Wrist** | ACC | 32 Hz | 1,920 |
| **Wrist** | EDA | 4 Hz | 240 |
| **Wrist** | TEMP | 4 Hz | 240 |

### How WESAD Handles This:

The WESAD dataset creators **pre-synchronized** all signals and stored them at their **native sampling rates** in the PKL file. This means:

1. **Labels are at 700 Hz** (reference clock = chest device)
2. **Each signal is stored at its own rate**
3. **All signals are time-aligned** (start at same moment)

### How YOU Handle This (3 Approaches):

#### **Approach 1: Index Mapping** (Current approach in your code)

When you need to match a label at 700 Hz to a wrist signal:

```python
# If you want data from label index 42000 to 84000 (1 minute at 700 Hz)
label_start = 42000
label_end = 84000

# Convert to BVP indices (64 Hz)
bvp_start = int(label_start * 64 / 700)  # = 3840
bvp_end = int(label_end * 64 / 700)      # = 7680
bvp_window = bvp_data[bvp_start:bvp_end]

# Convert to EDA indices (4 Hz)
eda_start = int(label_start * 4 / 700)   # = 240
eda_end = int(label_end * 4 / 700)       # = 480
eda_window = eda_data[eda_start:eda_end]
```

**This is what your notebook does!** See the windowing code in Part 5-7.

#### **Approach 2: Resampling** (Alternative)

You could resample everything to the same rate:

```python
from scipy.signal import resample

# Downsample labels from 700 Hz to 4 Hz
labels_4hz = resample(labels, len(labels) * 4 // 700)

# Now labels_4hz aligns directly with EDA
```

**Pros:** Easier indexing
**Cons:** Loss of temporal precision, computational cost

#### **Approach 3: Windowing with Time-based Indexing** (Most robust)

Your feature extraction uses **fixed time windows** (e.g., 60 seconds):

```python
window_duration = 60  # seconds

# For each signal, calculate how many samples = 60 seconds
bvp_samples = 60 * 64   # = 3840 samples
eda_samples = 60 * 4    # = 240 samples
acc_samples = 60 * 32   # = 1920 samples
label_samples = 60 * 700 # = 42000 samples

# All windows represent the SAME 60-second time period
# Just at different resolutions
```

**This is your current approach!** It's the best because:
- ✅ No data loss (preserves native resolution)
- ✅ No resampling artifacts
- ✅ Features are extracted at appropriate rates for each signal

---

## 4. How was the ground truth obtained?

Excellent question! Ground truth = the **labels** in the dataset. Here's how they were created:

### The Experimental Protocol:

Each subject went through a **controlled experiment** with distinct conditions:

```
Timeline (~2 hours):
┌─────────────┬─────────────┬─────────────┬─────────────┬─────────────┐
│  Baseline   │   Stress    │  Amusement  │ Meditation  │  Baseline   │
│  (Neutral)  │   (TSST)    │   (Funny    │  (Guided    │  (Neutral)  │
│             │             │   videos)   │  meditation)│             │
└─────────────┴─────────────┴─────────────┴─────────────┴─────────────┘
   Label: 1       Label: 2      Label: 3      Label: 4      Label: 1

(Label 0 = transitions between conditions)
```

### How Labels Were Assigned:

**Ground truth labels were assigned by the experimenters based on the experimental protocol**, NOT by analyzing the physiological data.

#### **Label 1: Baseline**
- **What subjects did:** Sat quietly, read neutral magazines
- **Duration:** ~20 minutes at start, ~20 minutes at end
- **Label assignment:** Experimenter marks start/end times

#### **Label 2: Stress (TSST - Trier Social Stress Test)**
- **What subjects did:**
  1. Prepare a 5-minute speech for a fake job interview
  2. Present speech in front of panel of evaluators
  3. Perform mental arithmetic (count backwards by 17 from 2043)
  4. Evaluators provide negative feedback
- **Duration:** ~10-15 minutes
- **Label assignment:** Experimenter marks start when instructions begin, end when task finishes
- **Why this works:** TSST is a **validated psychological stressor** - proven to reliably induce stress

#### **Label 3: Amusement**
- **What subjects did:** Watched funny video clips
- **Duration:** ~392 seconds (~6.5 minutes)
- **Label assignment:** Video start/stop times are known precisely

#### **Label 4: Meditation**
- **What subjects did:** Guided meditation exercise (audio)
- **Duration:** ~402 seconds (~6.7 minutes)
- **Label assignment:** Audio start/stop times are known precisely

#### **Label 0: Transient**
- **What subjects did:** Transitions between conditions (walking to different room, receiving instructions, etc.)
- **Label assignment:** Any time not in an active condition
- **Note:** These are **excluded from analysis** because physiological state is unclear

### Why This Ground Truth is Reliable:

1. **Experimenter-controlled:** Researchers knew exactly when each condition started/stopped
2. **Validated protocols:** TSST, meditation, and video stimuli are well-established in psychology
3. **Time-synchronized:** Labels were recorded in sync with sensor data (700 Hz sampling)
4. **Objective timing:** Video/audio conditions have precise start/stop times

### Important Limitations:

⚠️ **Label ≠ immediate physiological state**

- Stress response takes time to develop (30-60 seconds)
- Relaxation after stress is gradual
- Individual differences in response magnitude
- This is why **windowing** (60-second windows) helps smooth out transient effects

---

## 5. How the Data Loader Works

Let's trace through what happens when you run:

```python
loader = WESADDataLoader(data_dir)
data = loader.load_subject_pkl('S3')
```

### Step-by-Step Execution:

#### **Step 1: Initialization**
```python
def __init__(self, dataset_path: str):
    self.dataset_path = Path(dataset_path)  # Store base path
    self.subjects = self._get_available_subjects()  # Find all S* folders
```

**Result:** `loader.subjects = ['S2', 'S3', 'S4', ..., 'S17']`

#### **Step 2: Load PKL File**
```python
def load_subject_pkl(self, subject_id: str) -> Dict:
    pkl_file = self.dataset_path / subject_id / f"{subject_id}.pkl"
    # Example: "data/raw/S3/S3.pkl"

    with open(pkl_file, 'rb') as f:
        data = pickle.load(f, encoding='latin1')  # Python 2 → 3 compatibility

    return data
```

**Returns:** The nested dictionary structure shown in Section 2

#### **Step 3: Extract Signals (if requested)**

```python
# Extract wrist signals
wrist_signals = loader.extract_wrist_signals(data)
# Returns: {'ACC': array, 'BVP': array, 'EDA': array, 'TEMP': array}

# Extract labels
labels = loader.get_labels(data)
# Returns: 1D array at 700 Hz
```

**Key code in `extract_wrist_signals`:**
```python
def extract_wrist_signals(self, pkl_data: Dict) -> Dict[str, np.ndarray]:
    wrist_data = pkl_data['signal']['wrist']

    signals = {}

    # Handle 3D accelerometer
    if 'ACC' in wrist_data:
        signals['ACC'] = wrist_data['ACC']

    # Handle 1D signals (squeeze removes extra dimensions)
    if 'BVP' in wrist_data:
        signals['BVP'] = wrist_data['BVP'].squeeze()
    if 'EDA' in wrist_data:
        signals['EDA'] = wrist_data['EDA'].squeeze()
    if 'TEMP' in wrist_data:
        signals['TEMP'] = wrist_data['TEMP'].squeeze()

    return signals
```

**Why `.squeeze()`?**
- Some signals are stored as `(N, 1)` instead of `(N,)`
- `.squeeze()` converts `(N, 1)` → `(N)` for easier processing

---

## 6. Alternative: Raw E4 CSV Files

The `S3_E4_Data/` folder contains **raw, unprocessed** CSV files from the E4 device:

### CSV File Format:

**BVP.csv:**
```
1.572537216E9    ← Unix timestamp (first line)
64.0             ← Sampling rate (second line)
18.0             ← Data starts (third line onward)
20.5
22.3
...
```

**ACC.csv:**
```
1.572537216E9    ← Unix timestamp
32.0             ← Sampling rate
-0.015625, -0.015625, 0.984375    ← X, Y, Z (comma-separated)
-0.015625, -0.015625, 0.984375
...
```

### Why Use PKL Instead of CSV?

| Aspect | PKL File | CSV Files |
|--------|----------|-----------|
| **Synchronization** | ✅ Pre-aligned | ❌ Must align manually |
| **Labels** | ✅ Included | ❌ Separate file needed |
| **Chest data** | ✅ Included | ❌ Not available |
| **Loading speed** | ✅ Fast (binary) | ❌ Slower (text parsing) |
| **Use case** | ML/analysis | Debugging/inspection |

**Recommendation:** Use PKL files for all ML work. Only use CSVs if you need to debug sensor issues.

---

## 7. Summary: Data Flow Diagram

```
┌─────────────────────────────────────────────────────────────┐
│  EXPERIMENT (2 hours)                                       │
│  Subject wears:                                             │
│    - Chest device (RespiBAN): ECG, EDA, EMG, Resp, Temp, ACC│
│    - Wrist device (E4): BVP, EDA, Temp, ACC                │
│  Experimenter marks condition changes → Ground truth labels  │
└────────────────────────┬────────────────────────────────────┘
                         ↓
┌─────────────────────────────────────────────────────────────┐
│  DATA COLLECTION & SYNCHRONIZATION (by WESAD creators)      │
│  - Chest signals sampled at 700 Hz                          │
│  - Wrist signals sampled at native rates (4-64 Hz)         │
│  - All signals time-aligned to common start time           │
│  - Labels encoded at 700 Hz (matches chest clock)          │
└────────────────────────┬────────────────────────────────────┘
                         ↓
┌─────────────────────────────────────────────────────────────┐
│  S3.PKL FILE                                                │
│  {                                                          │
│    'subject': 'S3',                                         │
│    'signal': {'chest': {...}, 'wrist': {...}},             │
│    'label': [1,1,1,...,2,2,2,...,3,3,3,...,4,4,4,...]      │
│  }                                                          │
└────────────────────────┬────────────────────────────────────┘
                         ↓
┌─────────────────────────────────────────────────────────────┐
│  YOUR DATA LOADER (data_loader.py)                         │
│  - Loads PKL file                                           │
│  - Extracts wrist signals at native rates                  │
│  - Extracts labels at 700 Hz                               │
└────────────────────────┬────────────────────────────────────┘
                         ↓
┌─────────────────────────────────────────────────────────────┐
│  WINDOWING (preprocessing.py)                              │
│  - Create 60-second windows                                │
│  - For each window, extract:                               │
│    • BVP: 3840 samples (60s × 64 Hz)                       │
│    • EDA: 240 samples (60s × 4 Hz)                         │
│    • ACC: 1920 samples (60s × 32 Hz)                       │
│    • Labels: 42000 samples (60s × 700 Hz)                  │
│  - Assign label via majority vote                          │
└────────────────────────┬────────────────────────────────────┘
                         ↓
┌─────────────────────────────────────────────────────────────┐
│  FEATURE EXTRACTION (feature_extraction.py)                │
│  - Process each 60s window → 57 features                   │
│  - Each window = one training sample                       │
└────────────────────────┬────────────────────────────────────┘
                         ↓
┌─────────────────────────────────────────────────────────────┐
│  ML-READY DATASET                                           │
│  DataFrame with 57 feature columns + 1 label column        │
└─────────────────────────────────────────────────────────────┘
```

---

## 8. Key Takeaways

✅ **One PKL per subject** with everything synchronized
✅ **Both chest and wrist** data included
✅ **Different sampling rates** preserved at native resolution
✅ **Index mapping** converts between sampling rates
✅ **Ground truth** from experimental protocol (time-based, not data-derived)
✅ **Labels at 700 Hz** align with chest device (reference clock)
✅ **Windowing approach** handles multi-rate signals elegantly

---

## 9. Code Example: Complete Pipeline

```python
from pathlib import Path
from src.data_loader import WESADDataLoader
from src.preprocessing import SignalPreprocessor, E4_Converter
from src.feature_extraction import MultimodalFeatureExtractor
import numpy as np

# 1. Load data
data_dir = Path('data/raw')
loader = WESADDataLoader(data_dir)
data = loader.load_subject_pkl('S3')

# 2. Extract signals and labels
wrist = loader.extract_wrist_signals(data)
labels = loader.get_labels(data)

# 3. Convert to physical units
converter = E4_Converter()
wrist['ACC'] = converter.convert_acc(wrist['ACC'])

# 4. Create 60-second windows
window_duration = 60  # seconds
label_rate = 700

# Find stress period
stress_indices = np.where(labels == 2)[0]
start_idx = stress_indices[0]

# Get window at each sensor's rate
bvp_start = int(start_idx * 64 / 700)
bvp_window = wrist['BVP'][bvp_start : bvp_start + 60*64]

eda_start = int(start_idx * 4 / 700)
eda_window = wrist['EDA'][eda_start : eda_start + 60*4]

# 5. Extract features
from src.feature_extraction import HRVFeatures, EDAFeatures

hrv_features = HRVFeatures(64).extract_features(bvp_window)
eda_features = EDAFeatures(4).extract_features(eda_window)

# 6. Get window label (majority vote)
label_window = labels[start_idx : start_idx + 60*700]
window_label = np.bincount(label_window.astype(int)).argmax()

print(f"Extracted {len(hrv_features) + len(eda_features)} features")
print(f"Window label: {window_label} (Stress)")
```

---

**End of Explanation**
