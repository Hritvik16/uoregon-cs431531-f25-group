import matplotlib.pyplot as plt

labels = ["CPU 1 thread", "CPU 32 threads", "GPU"]
times = [13.245625, 0.488025, 0.114257]

plt.figure()
plt.bar(labels, times)
plt.ylabel("Time (s)")
plt.title("GoL Performance: CPU vs GPU (N=4096, steps=100)")
for i, t in enumerate(times):
    plt.text(i, t, f"{t:.3f}s", ha="center", va="bottom")
plt.savefig("cpu_gpu_comparison.png", dpi=200)
