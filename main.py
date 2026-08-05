import os
import subprocess
import sys
import time

def run_code(code_name):
	os.makedirs("logs", exist_ok=True)
	name_without_py = os.path.splitext(code_name)[0]
	path = os.path.join("logs", f"{name_without_py}.log")

	with open(path, 'w', encoding="utf-8") as file_log:
		did_run = subprocess.run([sys.executable, code_name], 
						   stdout=file_log, stderr=subprocess.STDOUT)

	if did_run.returncode != 0:
		print("Something went wrong")
		sys.exit(did_run.returncode)
	else:
		print(f"{code_name} has run succesfully")
	

if __name__ == "__main__":
	time_start = time.time()
	print("Starting program...")

	pipeline = ["database.py", "model.py"]
	for code in pipeline:
		run_code(code)

	duration = time.time() - time_start
	print(f"The runtime lasted {duration:.2f} seconds")