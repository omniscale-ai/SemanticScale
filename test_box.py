import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch

fig, ax = plt.subplots(figsize=(6, 6))
ax.set_xlim(0, 10)
ax.set_ylim(0, 10)

# Test 1: my current code
box1 = FancyBboxPatch((1, 7), 5.2, 1.6, boxstyle="round,pad=0.2",
                      linewidth=1.2, edgecolor="black", facecolor="#ebf4ff")
ax.add_patch(box1)
ax.text(1 + 2.6, 7 + 0.8, "Test 1: pad=0.2", ha="center", va="center")

# Test 2: pad=0.0, rounding_size=0.2
box2 = FancyBboxPatch((1, 4), 5.2, 1.6, boxstyle="round,pad=0.0,rounding_size=0.2",
                      linewidth=1.2, edgecolor="black", facecolor="#ebf4ff")
ax.add_patch(box2)
ax.text(1 + 2.6, 4 + 0.8, "Test 2: pad=0.0, rnd=0.2", ha="center", va="center")

# Test 3: original code
box3 = FancyBboxPatch((1, 1), 3.0, 1.6, boxstyle="round,pad=0.04",
                      linewidth=0.8, edgecolor="black", facecolor="#ebf4ff")
ax.add_patch(box3)
ax.text(1 + 1.5, 1 + 0.8, "Test 3: original", ha="center", va="center")

plt.savefig("test_box.png")
