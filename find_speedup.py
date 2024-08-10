import subprocess
import re
import os

gpu_report = "./reports/fractal_details_GPU.txt"
cpu_report = "./reports/fractal_details_CPU.txt"

def find_time(file_path):
    with open(file_path, 'r') as file:
        content = file.read()
    
    # Regular expression to find "Time to generate" values
    time_pattern = re.compile(r"Time to generate:\s*([\d\.]+)\s*ms")
    
    # Find all occurrences of the pattern
    times = time_pattern.findall(content)
    
    # Convert the extracted times to float
    times = [float(time) for time in times]
    
    return times

N_TRIALS = 10
subprocess.run(["make", "clean"])
subprocess.run(["make", "prepare"])
subprocess.run(["make", "build"])
subprocess.run(["make", "build-gpu"])

cpu_times = []
gpu_times = []
speedups = []

for i in range(N_TRIALS):
    print(f"\nITERATION {i}\n---------------------")
    subprocess.run(["make", "execution"])
    # subprocess.run(["make", "run-gpu"])
    
    cpu_time = find_time(cpu_report)
    gpu_time = find_time(gpu_report)

    # Debug print statements to check the values
    # print(f"CPU times: {cpu_time}")
    # print(f"GPU times: {gpu_time}")
    
    if len(cpu_time) == 2 and len(gpu_time) == 2:  # Ensure there are exactly two times
        cpu_times.append(cpu_time)
        gpu_times.append(gpu_time)
        speedups.append([cpu / gpu for cpu, gpu in zip(cpu_time, gpu_time)])
        print(f"Speedups:\tMandelbrot: {speedups[-1][0]}\tTriple Mandelbrot: {speedups[-1][1]}")
    else:
        print("Error: Mismatch in expected number of timing values.")
    os.remove(gpu_report)
    os.remove(cpu_report)
    print("---------------------")

# Calculate average speedup
if speedups:  # Ensure there is data in speedups before processing
    average_speedup = [sum(x)/len(x) for x in zip(*speedups)]
else:
    average_speedup = [0, 0]  # Default to 0 if no data

# Calculate average times for CPU and GPU
if cpu_times and gpu_times:  # Ensure there is data in cpu_times and gpu_times before processing
    average_cpu_times = [sum(x)/len(x) for x in zip(*cpu_times)]
    average_gpu_times = [sum(x)/len(x) for x in zip(*gpu_times)]
else:
    average_cpu_times = [0, 0]  # Default to 0 if no data
    average_gpu_times = [0, 0]  # Default to 0 if no data

# Prepare output
output_lines = []
output_lines.append("Individual Speedups per trial (Mandelbrot, Triple Mandelbrot):\n")
for i, speedup in enumerate(speedups):
    output_lines.append(f"Trial {i + 1}: Mandelbrot: {speedup[0]:.2f}, Triple Mandelbrot: {speedup[1]:.2f}\n")

output_lines.append("\nAverage Speedups:\n")
output_lines.append(f"Mandelbrot: {average_speedup[0]:.2f}\n")
output_lines.append(f"Triple Mandelbrot: {average_speedup[1]:.2f}\n")

output_lines.append("\nAverage Times (ms):\n")
output_lines.append(f"CPU Mandelbrot: {average_cpu_times[0]:.2f}\n")
output_lines.append(f"CPU Triple Mandelbrot: {average_cpu_times[1]:.2f}\n")
output_lines.append(f"GPU Mandelbrot: {average_gpu_times[0]:.2f}\n")
output_lines.append(f"GPU Triple Mandelbrot: {average_gpu_times[1]:.2f}\n")

# Print output to console
print("".join(output_lines))

# Write output to a text file
output_file = "./reports/speedup_report.txt"
with open(output_file, "w") as file:
    file.writelines(output_lines)

print(f"Speedup report saved to {output_file}")
