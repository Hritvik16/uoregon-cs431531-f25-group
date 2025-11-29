import matplotlib.pyplot as plt

threads = [1, 2, 4, 8, 16, 32]
times = [13.245625, 7.011376, 3.415661, 1.717138, 0.882485, 0.488025]
speedups = [times[0] / t for t in times]

# plt.figure()
# plt.plot(threads, times, marker="o")
# plt.xlabel("Threads")
# plt.ylabel("Time (s)")
# plt.title("GoL OpenMP Strong Scaling (N=4096, steps=100)")
# plt.xscale("log", base=2)
# plt.yscale("log", base=10)
# plt.grid(True, which="both")
# plt.savefig("openmp_time.png", dpi=200)

# plt.figure()
# plt.plot(threads, speedups, marker="o")
# plt.xlabel("Threads")
# plt.ylabel("Speedup (vs 1 thread)")
# plt.title("GoL OpenMP Speedup")
# plt.xscale("log", base=2)
# plt.grid(True, which="both")
# plt.savefig("openmp_speedup.png", dpi=200)

plt.figure()
plt.plot(threads, speedups, marker="o")
plt.xlabel("Threads")
plt.ylabel("Speedup (vs 1 thread)")
plt.title("GoL OpenMP Speedup")
plt.grid(True)
plt.savefig("openmp_speedu1.png", dpi=200)



plt.figure()
plt.plot(threads, times, marker="o")
plt.xlabel("Threads")
plt.ylabel("Time (s)")
plt.title("GoL OpenMP Strong Scaling (N=4096, steps=100)")
plt.grid(True)
plt.savefig("openmp_time1.png", dpi=200)