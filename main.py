import os
import subprocess
import sys
import time

# Running the code and saving logs
def run_code(code_name):
	"""
		Automating the process of running each separate code file,
		saving the output along the way for easier verification.
	"""
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
	

# Start of the runtime
if __name__ == "__main__":
	time_start = time.time()
	# If the model has not been optimized yet,
	# the database is created and the model is trained on it
	if not os.path.exists("models/best_resnet18.pt"):
		print("The model has not been trained")
		print("Initiate training:")
		pipeline = ["database.py", "train_ai.py"]
		for code in pipeline:
			run_code(code)
	# Testing part of the model
	print("The model is ready; testing begins")
	pipeline=["test_and_predict_model.py", "plot_from_saved_model.py", "generate_scatter_plot.py"]
	for code in pipeline:
		run_code(code)

	duration = time.time() - time_start
	print(f"The runtime lasted {duration:.2f} seconds")