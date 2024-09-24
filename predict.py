import matplotlib.pyplot as plt  # For plotting
import pandas as pd  # For handling dataframes (CSV files)
import argparse  # For handling command-line arguments
import numpy as np  # For numerical operations
import sys  # For system-specific parameters and functions

def error_exit(string):
	"""
	Print an error message and exit the program.
	"""
	print(string)
	sys.exit(0)  # Terminates the program

def file_check(file):
	"""
	Check if the given file contains valid data.
	Each line (after the header) must contain two numeric values.
	Returns 1 if valid, 0 if invalid.
	"""
	try:
		with open(file, 'r') as f:
			fl = f.readlines()  # Read all lines from the file
			for l in fl[1:]:  # Skip the first line (header)
				L = l[:-1].split(',')  # Split each line by commas
				if not L[0].isnumeric() or not L[1].isnumeric():  # Check if both fields are numeric
					return 0  # Invalid format
			return 1  # Valid file
	except:
		error_exit('No data')  # Handle file read error

def check_number(f, n):
	"""
	Check if the input 'n' can be converted using function 'f'.
	If conversion fails, exit the program with an error.
	"""
	try:
		n = f(n)  # Attempt to convert 'n' using the provided function 'f'
	except:
		error_exit("Value error")  # Handle conversion error
	return n  # Return the converted value

if __name__ == '__main__':
	# Argument parsing for command-line inputs
	parser = argparse.ArgumentParser()
	parser.add_argument('mileage', type=int, help='mileage to predict')  # Expected mileage input
	parser.add_argument('file', type=str, help='text file for input', default=None)  # Input file for theta values
	parser.add_argument('-sc', '--scatter', help='data scatter plot', type=str)  # Optional argument for scatter plot file
	args = parser.parse_args()

	# Check if theta (model parameters) are present in the input file
	try:
		with open(args.file, 'r') as f:
			if f.mode == 'r':  # Ensure the file is open in read mode
				theta = []  # List to hold theta values
				fl = f.readlines()  # Read the file lines
				for x in fl:  # Extract theta values
					theta.append(check_number(float, x.split(':')[1]))  # Convert and append theta values
	except:
		theta = [0, 0]  # Default theta values if file read fails

	# Display the theta values
	print('Theta0: ', theta[0])
	print('Theta1: ', theta[1])

	# Mileage price calculation using the linear model: price = theta0 + theta1 * mileage
	d = args.mileage  # Get the mileage from the command line
	d = check_number(float, d)  # Convert mileage to float
	p = theta[0] + theta[1] * d  # Calculate the predicted price
	print('The price of this car is :', p, 'euros')  # Output the predicted price

	# If scatter plot data is provided, check the file and read it into a DataFrame
	if args.scatter:
		if not file_check(args.scatter):  # Check if the scatter file is valid
			error_exit('Bad File Format')  # Exit if file is invalid
		df = pd.read_csv(args.scatter)  # Load the file into a pandas DataFrame
		km = df.columns[0]  # First column is the mileage (km)
		price = df.columns[1]  # Second column is the price
		plt.scatter(df[km], df[price])  # Plot the scatter plot with mileage vs price
		X, Y = df[km], df[price]  # Store mileage and price for further use
	else:
		X = np.linspace(0, 250000, num=25)  # Generate a range of mileage values if scatter data is not provided
		price = None  # No price data available
		km = None  # No mileage data available

	# Plot the predicted price point in red
	plt.plot(d, p, '*', markersize=12, color='red')

	# Plot the linear prediction line
	if d > max(X):  # If the given mileage exceeds the dataset's range
		plt.plot(
			pd.DataFrame([i for i in range(int(d) + 10000)], columns=['KM']),
			theta[0] + theta[1] * pd.DataFrame([i for i in range(int(d) + 10000)], columns=['KM']),
			color='green'  # Prediction line for larger mileage
		)
	else:
		plt.plot(X, theta[0] + theta[1] * X, color='green')  # Prediction line within the dataset range

	# Add labels and title to the plot if scatter data is available
	if price and km:
		plt.ylabel(price)  # Y-axis label as price
		plt.xlabel(km)  # X-axis label as mileage
		plt.title(price + ' = f(' + km + ')')  # Title showing the relationship

	# Save the plot as an image
	plt.savefig('PredictGraph.png')

