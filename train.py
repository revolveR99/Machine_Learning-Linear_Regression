from util import *
from reprint import output
import matplotlib.pyplot as plt
import argparse
import imageio
import signal
import os


class ft_linear_regression:
    def __init__(self, rate, iter, input, output, po, pn, history, live):
        # Initialize the figure and axis for plotting the data
        self.figure, self.axis = plt.subplots(2, 2, figsize=(10, 10))

        # Import data sets and split them into normalized (X, Y) and original (x, y) coordinates
        self.data = getData(input)
        self.x = self.data[0]  # Original x values
        self.y = self.data[1]  # Original y values
        self.X = normalisation(self.x)  # Normalized x values
        self.Y = normalisation(self.y)  # Normalized y values

        # Initialize theta values: normalized (_T) and denormalized (T)
        self._T0 = 0  # Normalized intercept
        self._T1 = 0  # Normalized slope
        self.T0 = 1   # Denormalized intercept
        self.T1 = 1   # Denormalized slope

        # Initialize metrics for data processing
        self.M = len(self.x)  # Length of the datasets
        self.C = []           # List to hold the history of cost values
        self.images = []      # List to hold images of each iteration
        self.RMSE = None      # Root Mean Square Error value
        self.MSE = None       # Mean Square Error value

        # Initialize variables to track progress of accuracy
        self.prev_mse = 0.0   # Previous MSE value for comparison
        self.cur_mse = self.cost()  # Current MSE value calculated
        self.delta_mse = self.cur_mse  # Difference in MSE for accuracy assessment

        # Set learning rate and maximum iterations for the training model
        self.learning_rate = rate  # Learning rate (alpha)
        self.iterations = 0        # Current iteration count
        self.max_iterations = iter  # Maximum number of iterations for training
        self.output = output       # Output file name for final theta values

        # Flags for visualizing different aspects of the training
        self.po = po              # Flag to plot original data
        self.pn = pn              # Flag to plot normalized data
        self.history = history     # Flag to plot history of cost
        self.live = live          # Flag to enable live watching of training

    def RMSE_percent(self):
        # Calculate and return the Root Mean Square Error as a percentage
        self.RMSE = 100 * (1 - self.cost() ** 0.5)
        return self.RMSE

    def MSE_percent(self):
        # Calculate and return the Mean Square Error as a percentage
        self.MSE = 100 * (1 - self.cost())
        return self.MSE

    def cost(self):
        """
        Calculate Mean Square Error (MSE) for the current model
        """
        dfX = DataFrame(self.X, columns=['X'])  # Create DataFrame for normalized X
        dfY = DataFrame(self.Y, columns=['Y'])  # Create DataFrame for normalized Y
        # Calculate and return the mean squared error
        return ((self.T1 * dfX['X'] + self.T0 - dfY['Y']) ** 2).sum() / self.M


        def estimatePrice(self, t0, t1, mileage):
        # Estimate the price based on the given parameters: t0 (intercept), t1 (slope), and mileage
        return ((t0 + (t1 * float(mileage))))

    def live_update(self, output_lines):
        # Calculate the range of x and y values for normalization
        deltaX = max(self.x) - min(self.x)  # Range of x values
        deltaY = max(self.y) - min(self.y)  # Range of y values
        # Update the normalized theta values
        self._T1 = deltaY * self.T1 / deltaX  # Normalize slope
        self._T0 = ((deltaY * self.T0) + min(self.y) - self.T1 * (deltaY / deltaX) * min(self.x))  # Normalize intercept
        # Update output lines with current values of theta, RMSE, MSE, delta MSE, and iterations
        output_lines[prCyan('    Theta0           ')] = str(self.T0)
        output_lines[prCyan('    Theta1           ')] = str(self.T1)
        output_lines[prCyan('    RMSE             ')] = f'{round(self.RMSE_percent(), 2)} %'
        output_lines[prCyan('    MSE              ')] = f'{round(self.MSE_percent(), 2)} %'
        output_lines[prCyan('    Delta MSE        ')] = str(self.delta_mse)
        output_lines[prCyan('    Iterations       ')] = str(self.iterations)

    def condition_to_stop_training(self):
        # Determine whether to stop training based on maximum iterations or delta MSE threshold
        if self.max_iterations == 0:
            return self.delta_mse > 0.0000001 or self.delta_mse < -0.0000001  # Check if delta MSE is within a small range
        else:
            return self.iterations < self.max_iterations  # Check if iterations are less than the maximum

    def gradient_descent(self):
        # Print the start of the training process
        print("\033[33m{:s}\033[0m".format('TRAINING MODEL :'))
        self.iterations = 0  # Initialize iteration count
        # Create a live output display for training metrics
        with output(output_type='dict', sort_key=lambda x: 1) as output_lines:
            while self.condition_to_stop_training():  # Continue training while conditions are met
                sum1 = 0  # Accumulator for sum of errors
                sum2 = 0  # Accumulator for sum of errors weighted by x
                for i in range(self.M):  # Loop through all data points
                    T = self.T0 + self.T1 * self.X[i] - self.Y[i]  # Calculate the error
                    sum1 += T  # Accumulate error
                    sum2 += T * self.X[i]  # Accumulate weighted error

                # Update theta values using gradient descent
                self.T0 = self.T0 - self.learning_rate * (sum1 / self.M)  # Update intercept
                self.T1 = self.T1 - self.learning_rate * (sum2 / self.M)  # Update slope

                self.C.append(self.cost())  # Append current cost to history

                # Update MSE tracking
                self.prev_mse = self.cur_mse  # Store previous MSE
                self.cur_mse = self.cost()  # Calculate current MSE
                self.delta_mse = self.cur_mse - self.prev_mse  # Calculate change in MSE

                self.iterations += 1  # Increment iteration count

                # Update live output every 100 iterations or on the first iteration
                if self.iterations % 100 == 0 or self.iterations == 1:
                    self.live_update(output_lines)  # Update output display
                    if self.live == True:  # If live mode is enabled, plot current state
                        self.plot_all(self.po, self.pn, self.history)

            self.live_update(output_lines)  # Final update of output lines after training

        # Calculate final RMSE and MSE percentages after training is complete
        self.RMSE_percent()
        self.MSE_percent()


              # Print success message indicating the model has been applied to the data
        print(prYellow('SUCCESS :'))
        print(prGreen("    Applied model to data"))
        print(prYellow('RESULTS (Normalized)  :'))
        # Print the normalized theta values (intercept and slope)
        print(f'    {prCyan("Theta0           :")} {self.T0}\n    {prCyan("Theta1           :")} {self.T1}')
        print(prYellow('RESULTS (DeNormalized):'))
        # Print the denormalized theta values
        print(f'    {prCyan("Theta0           :")} {self._T0}\n    {prCyan("Theta1           :")} {self._T1}')
        print("\033[33m{:s}\033[0m".format('ALGORITHM ACCURACY:'))
        # Print the RMSE and MSE values as percentages
        print(f'    {prCyan("RMSE             : ")}{round(ftlr.RMSE, 2)} % ≈ ({ftlr.RMSE} %)')
        print(f'    {prCyan("MSE              : ")}{round(ftlr.MSE, 2)} % ≈ ({ftlr.MSE} %)')
        print(f'    {prCyan("ΔMSE             : ")}{ftlr.delta_mse}')  # Print the change in MSE
        print(prYellow('Storing Theta0 && Theta1:'))
        # Store the theta values in a CSV file
        set_gradient_csv(self.output, self._T0, self._T1)
        print(prGreen("    Theta0 && Theta1 has been stored in file , open : ") + self.output)

        # If plotting options are enabled, plot the data
        if self.po or self.pn or self.history:
            print(prYellow('Plotting Data:'))
            self.plot_all(self.po, self.pn, self.history, final=True)  # Plot the data and save the graph
            print(prGreen("    Data plotted successfully , open : ") + 'LR-Graph.png')

        # If live mode is enabled, create a GIF of the training progress
        if self.live == True:
            print(prYellow('Creating GIF image of progress:'))
            self.gifit()  # Generate GIF from images
            print(prGreen("    Live progress GIF created , open : ") + 'LR-Live.gif')

    def gifit(self):
        # Remove existing GIF if it exists
        if os.path.exists('./LR-Live.gif'):
            os.remove('./LR-Live.gif')
        def sorted_ls(path):
            # Sort files in the specified directory based on modification time
            mtime = lambda f: os.stat(os.path.join(path, f)).st_mtime
            return list(sorted(os.listdir(path), key=mtime))

        filenames = sorted_ls('./gif')  # Get sorted list of filenames in the './gif' directory
        with imageio.get_writer('./LR-Live.gif', mode='I') as writer:
            # Append each image file to the GIF writer
            for filename in filenames:
                image = imageio.imread('./gif/' + filename)  # Read image
                writer.append_data(image)  # Add image to GIF

    def plot_original(self):
        # Create a plot for the original data and the estimation
        p1 = self.axis[0, 0]  # Select the subplot for original data
        p1.plot(self.x, self.y, 'ro', label='data')  # Plot original data points
        x_estim = self.x  # Use original x values for estimation
        # Calculate estimated y values based on the model
        y_estim = [denormalizeElem(self.y, self.estimatePrice(self.T0, self.T1, normalizeElem(self.x, _))) for _ in x_estim]
        p1.plot(x_estim, y_estim, 'g-', label='Estimation')  # Plot estimation line
        p1.set_ylabel('Price (in euro)')  # Set y-axis label
        p1.set_xlabel('Mileage (in km)')  # Set x-axis label
        p1.set_title('Price = f(Mileage) | Original')  # Set title for the plot

    def plot_normalized(self):
        # Create a plot for the normalized data and the estimation
        p2 = self.axis[0, 1]  # Select the subplot for normalized data
        p2.plot(self.X, self.Y, 'ro', label='data')  # Plot normalized data points
        x_estim = self.X  # Use normalized x values for estimation
        # Calculate estimated y values based on the model
        y_estim = [self.estimatePrice(self.T0, self.T1, _) for _ in x_estim]
        p2.plot(x_estim, y_estim, 'g-', label='Estimation')  # Plot estimation line

        p2.set_title('Price = f(Mileage) | Normalized')  # Set title for the plot

    def plot_history(self):
        # Create a plot for the cost over iterations
        p4 = self.axis[1, 1]  # Select the subplot for cost history
        p4.set_ylabel('Cost')  # Set y-axis label
        p4.set_xlabel('Iterations')  # Set x-axis label
        p4.set_title(f'Cost = f(iteration) | L.Rate = {self.learning_rate}')  # Set title for the plot
        p4.plot([i for i in range(self.iterations)], self.C)  # Plot cost history against iterations

    def plot_show(self, p1, p2, p4, final):
        # Show or save plots based on input parameters
        if p1 != False or p2 != False or p4 != False:
            if p1 == False:  # Turn off the first plot if not needed
                self.axis[0, 0].axis('off')

            if p2 == False:  # Turn off the second plot if not needed
                self.axis[0, 1].axis('off')

            if p4 == False:  # Turn off the history plot if not needed
                self.axis[1, 1].axis('off')

            self.axis[1, 0].axis('off')  # Turn off the last subplot

            # plt.show() # Uncomment to display the plot window in IDEs

            imgname = f'./gif/LR-Graph-{self.iterations}.png'  # Set image name based on iterations
            if final == True:  # If this is the final image, change the filename
                imgname = f'./LR-Graph.png'

            plt.savefig(imgname)  # Save the current plot as an image
            plt.close()  # Close the plot to free up memory


def plot_all(self, p1, p2, p4, final=False):
    # Create a 2x2 subplot for plotting various graphs
    self.figure, self.axis = plt.subplots(2, 2, figsize=(10, 10))

    # Plot original data if the corresponding flag is set
    if p1:
        self.plot_original()
    # Plot normalized data if the corresponding flag is set
    if p2:
        self.plot_normalized()
    # Plot cost history if the corresponding flag is set
    if p4:
        self.plot_history()

    # Display the plots and save/show them based on the final parameter
    self.plot_show(p1, p2, p4, final)


def optparse():
    """
        Parse command line arguments for the program
    """
    parser = argparse.ArgumentParser()
    # Argument for input data file, with a default value
    parser.add_argument('--input', '-in', action="store", dest="input", type=str, default='data.csv',
                        help='source of data file')

    # Argument for output file to store theta values, with a default value
    parser.add_argument('--output', '-o', action="store", dest="output", type=str, default='thetas.txt',
                        help='source of data file')

    # Argument to set a limit on the number of iterations
    parser.add_argument('--iteration', '-it', action="store", dest="iter", type=int, default=0,
                        help='Change number of iteration. (default is Uncapped)')

    # Argument to enable saving history for future display
    parser.add_argument('--history', '-hs', action="store_true", dest="history", default=False,
                        help='save history to future display')

    # Argument to enable plotting of original data sets
    parser.add_argument('--plotOriginal', '-po', action="store_true", dest="plot_original", default=False,
                        help="Enable to plot the original data sets")

    # Argument to enable plotting of normalized data sets
    parser.add_argument('--plotNormalized', '-pn', action="store_true", dest="plot_normalized", default=False,
                        help="Enable to plot the normalized data sets")

    # Argument to set the learning rate for the algorithm
    parser.add_argument('--learningRate', '-l', action="store", dest="rate", type=float, default=0.1,
                        help='Change learning coefficient. (default is 0.1)')

    # Argument to enable live updates during training, saving them to a GIF
    parser.add_argument('--live', '-lv', action="store_true", dest="live", default=False,
                        help='Store live changes on gif graph')
    return parser.parse_args()  # Return parsed arguments


def signal_handler(sig, frame):
    # Handle interrupt signal (e.g., Ctrl+C) to exit gracefully
    sys.exit(0)


if __name__ == '__main__':
    # Register the signal handler for SIGINT
    signal.signal(signal.SIGINT, signal_handler)

    # Welcome message with ASCII art
    welcome = """
████████ ████████         ██       ████ ██    ██ ████████    ███    ████████          ████████  ████████  ██████   ████████  ████████  ██████   ██████  ████  ███████  ██    ██ 
██          ██            ██        ██  ███   ██ ██         ██ ██   ██     ██         ██     ██ ██       ██    ██  ██     ██ ██       ██    ██ ██    ██  ██  ██     ██ ███   ██ 
██          ██            ██        ██  ████  ██ ██        ██   ██  ██     ██         ██     ██ ██       ██        ██     ██ ██       ██       ██        ██  ██     ██ ████  ██ 
██████      ██            ██        ██  ██ ██ ██ ██████   ██     ██ ████████          ████████  ██████   ██   ████ ████████  ██████    ██████   ██████   ██  ██     ██ ██ ██ ██ 
██          ██            ██        ██  ██  ████ ██       █████████ ██   ██           ██   ██   ██       ██    ██  ██   ██   ██             ██       ██  ██  ██     ██ ██  ████ 
██          ██            ██        ██  ██   ███ ██       ██     ██ ██    ██          ██    ██  ██       ██    ██  ██    ██  ██       ██    ██ ██    ██  ██  ██     ██ ██   ███ 
██          ██            ████████ ████ ██    ██ ████████ ██     ██ ██     ██         ██     ██ ████████  ██████   ██     ██ ████████  ██████   ██████  ████  ███████  ██    ██ 

   """
    # Print welcome message
    print(welcome)

    # Create a directory for GIF images if it doesn't exist
    if not os.path.exists('./gif'):
        os.makedirs('./gif')

    options = optparse()  # Parse command line arguments
    # Validate learning rate, setting default if out of range
    if (options.rate < 0.0000001 or options.rate > 1):
        options.rate = 0.1
    # Print initial parameters for training
    print("\033[33m{:s}\033[0m".format('Initial Params for training model:'))
    print(prCyan('    Learning Rate    : ') + str(options.rate))
    print(prCyan('    Max iterations   : ') + "Uncapped" if str(options.iter) == "0" else "0")
    print(prCyan('    Plot Original    : ') + ('Enabled' if options.plot_original else 'Disabled'))
    print(prCyan('    Plot Normalized  : ') + ('Enabled' if options.plot_normalized else 'Disabled'))
    print(prCyan('    Plot History     : ') + ('Enabled' if options.history else 'Disabled'))
    print(prCyan('    DataSets File    : ') + options.input)
    print(prCyan('    Output File      : ') + options.output)

    # Initialize the linear regression model with parsed options
    ftlr = ft_linear_regression(rate=options.rate,
                                iter=options.iter,
                                input=options.input,
                                output=options.output,
                                po=options.plot_original,
                                pn=options.plot_normalized,
                                history=options.history,
                                live=options.live)
    ftlr.gradient_descent()  # Start the gradient descent training process

