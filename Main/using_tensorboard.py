from torch.utils.tensorboard import SummaryWriter
from tensorboard.backend.event_processing import event_accumulator
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
ea = event_accumulator.EventAccumulator("runs/exp3")  # path to your event folder
ea.Reload()

# List available tags
print(ea.Tags())

data = {}
steps = None
for t in ea.Tags().get("scalars", []):
    events = ea.Scalars(t)
    vals = np.array([e.value for e in ea.Scalars(t)])
    if steps is None:
        steps = [e.step for e in events]  # assume all tags have same steps
        data["step"] = steps
    data[t] = np.round(vals, 5)   # <-- add values for this tag
    print(f"{t}: {np.round(vals, 5)}")

# build DataFrame
df = pd.DataFrame(data)
# save to CSV
out_path = "all_scalars.csv"
df.to_csv(out_path, index=False)

# Extract scalars
train_loss = ea.Scalars("loss/train_norm")
val_loss = ea.Scalars("loss/val_norm")

#print("Train loss entries:", train_loss)
#print("Val loss entries:", val_loss)

# Convert to lists of (step, value)
steps = [x.step for x in train_loss]
train_vals = [x.value for x in train_loss]
val_vals   = [x.value for x in val_loss]

# Plot
plt.figure(figsize=(8,6))
plt.plot(steps, train_vals, label="Train Loss")
plt.plot(steps, val_vals, label="Val Loss")
plt.xlabel("Epoch")
plt.ylabel("Loss (MSE)")
plt.title("Training vs Validation Loss")
plt.legend()
plt.grid(True)
plt.show()
plt.savefig("res.jpg")