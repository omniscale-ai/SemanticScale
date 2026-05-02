import matplotlib.pyplot as plt

fig, ax = plt.subplots(figsize=(4, 4))
ax.set_xlim(-0.5, 2.5)
ax.set_ylim(0, 100)

labels = ["Random", "LightGBM", "Oracle"]
values = [58.2, 68.0, 83.0]
ax.bar([0, 1, 2], values, width=0.75)
for i, v in enumerate(values):
    ax.text(i, v + 2, f"{v:.1f}%", ha="center", fontsize=12)

ax.annotate("", xy=(1, 75), xytext=(0.2, 62),
            arrowprops=dict(arrowstyle="-|>", color="blue", lw=2.0, mutation_scale=15,
                            connectionstyle="angle,angleA=0,angleB=90,rad=0"))
plt.savefig("test_arrow.png")
