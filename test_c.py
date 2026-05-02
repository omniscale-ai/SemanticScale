import matplotlib.pyplot as plt
fig, ax = plt.subplots(figsize=(4, 4))
ax.set_ylim(0, 100)
labels = ["Random\n(per-attempt avg)", "LightGBM\nSLoD scorer", "Pass@5\noracle"]
values = [58.2, 68.0, 83.0]
colors = ["#a0aec0", "blue", "#2d3748"]
bars = ax.bar(labels, values, color=colors, edgecolor="white", linewidth=1.2, width=0.75)
for bar, v, c in zip(bars, values, colors):
    ax.text(bar.get_x() + bar.get_width() / 2, v + 2,
            f"{v:.1f}%", ha="center", fontsize=12, fontweight="bold",
            color=c)

# Let's plot grid to see coordinates
ax.grid(True)
plt.savefig("test_c_base.png")
