# Graph Placement Guide for COMPREHENSIVE_REPORT.md

## Location: Section 6.1 - Training Loss Dynamics

### Where the Graph Goes:

The training loss curve image `training_loss.png` is referenced in **Section 6.1** of the report under "Training Loss Curve Visualization".

### File Location:

```
/Users/tsheringwangpodorji/Documents/Year3 Sem II/DAM304/DAM304_AS2026_Practicals/practical_4/training_loss.png
```

### Exact Position in Document:

```
┌─ Section 6.1: Training Loss Dynamics
│
├─ 📊 [INSERT GRAPH HERE] ← training_loss.png
│   - Position: After "Loss reduction per epoch" metrics table
│   - Before: "Figure 1: Training Loss Curve..." caption
│
│   Title: Figure 1: Training Loss Curve - MiniLM
│   Caption: "The graph shows smooth convergence from epoch 1 to 100, with three
│            distinct phases: rapid decrease (epochs 1-30), gradual decrease
│            (epochs 30-70), and plateau (epochs 70-100). The exponential decay
│            pattern is characteristic of deep learning optimization on closed datasets."
│
├─ ⚠️ **Where to Place This Graph in Your Document** (highlighted box)
│   Special instructions for converting to DOCX/PDF
│
└─ #### Analysis of Training Dynamics
    (Analysis text continues...)
```

## When Converting to DOCX/PDF:

### Step-by-Step Instructions:

1. **Open** the Markdown file in your preferred editor (Word, Google Docs, etc.)

2. **Locate** the line that says:

   ```
   ![Training Loss Curve](training_loss.png)
   ```

3. **Insert Image**:
   - Position cursor right before this line
   - Go to Insert → Image
   - Navigate to: `/practical_4/training_loss.png`
   - Select and insert

4. **Format Image**:
   - **Size**: 6 inches wide × 4 inches tall (maintains aspect ratio)
   - **Alignment**: Centered
   - **Caption**: "Figure 1: Training Loss Curve - MiniLM (100 epochs)"

5. **Spacing**:
   - Space above image: 12pt
   - Space below image: 12pt
   - Keep the explanatory text after the graph

## Content of the Graph:

The `training_loss.png` shows:

- **X-axis**: Epoch (0 to 100)
- **Y-axis**: Cross-Entropy Loss (0 to 3.5)
- **Line**: Smooth blue curve showing loss decrease
- **Pattern**:
  - Steep drop: Epochs 1-30
  - Gradual decline: Epochs 30-70
  - Plateau: Epochs 70-100
- **Final value**: 0.1292

## Summary of Actual Values Now in Report:

| Metric                         | Value         | Location in Report  |
| ------------------------------ | ------------- | ------------------- |
| **Model Parameters**           | 289,792       | Section 6.1 (table) |
| **Initial Loss**               | 3.0888        | Section 6.1 (table) |
| **Final Loss**                 | 0.1292        | Section 6.1 (table) |
| **Vocab Size**                 | 25 characters | Section 3.2, 6.1    |
| **Training Examples**          | 138           | Section 6.1, 6.2    |
| **Parameter-to-Example Ratio** | 2,100:1       | Section 6.2         |

---

**Report is now calibrated with actual experimental results!** 
