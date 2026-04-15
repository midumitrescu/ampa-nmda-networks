import unittest


import cv2
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit


def gaussian(x, A, mu, sigma):
    return A * np.exp(-(x - mu) ** 2 / (2 * sigma ** 2))

def plot_gaussian_on_top_of_image(mean, sigma, amp, figure = "img.png" ):
    img = cv2.imread(figure)
    x_fit = np.linspace(-80, -40, 500)
    y_fit = gaussian(x_fit, amp, mu=mean, sigma=sigma)

    plt.figure(figsize=(6, 4))

    # background image
    plt.imshow(img, extent=[-80, -40, 0, 8], aspect='auto')
    plt.plot(x_fit, y_fit, 'r', linewidth=3, label=f"Gaussian μ={mean:.2f}, σ={sigma:.2f}")

    plt.legend()
    plt.xlabel("x")
    plt.ylabel("density")

    plt.show()

class MyTestCase(unittest.TestCase):
    def test_something(self):
        import numpy as np
        import matplotlib.pyplot as plt
        from scipy.optimize import curve_fit
        from PIL import Image

        def gaussian(x, mu, sigma, A):
            return A * np.exp(-0.5 * ((x - mu) / sigma) ** 2)

        # Step 1: Load image
        img_path = 'fig_1_c.png'  # Your saved image path
        img = Image.open(img_path).convert('L')  # Convert to grayscale
        img_np = np.array(img)

        # Step 2: Crop region of interest (ROI) manually if needed
        # For now, use entire image
        height, width = img_np.shape

        # Step 3: Extract data from image by summing pixel intensities vertically to get fraction of time curve
        # Invert image so peaks are bright (white)
        img_inverted = 255 - img_np

        # Sum pixel values vertically (along y axis) to get intensity vs x (mV axis)
        intensity_profile = img_inverted.sum(axis=0)

        # Normalize intensity profile to fraction of time scale (~0 to 6%)
        intensity_profile = intensity_profile / intensity_profile.max() * 6

        # Step 4: Map pixel x positions to membrane potential (mV)
        # From figure: x-axis from -80 mV (left) to -40 mV (right)
        V_min, V_max = -80, -40
        x_vals = np.linspace(V_min, V_max, width)

        # Step 5: Select only data for the Up-state peak (~right peak, between -65 and -50 mV)
        mask = (x_vals >= -65) & (x_vals <= -50)
        x_fit = x_vals[mask]
        y_fit = intensity_profile[mask]

        # Step 6: Fit Gaussian to selected peak
        # Initial guess: mean near -58, sigma 3, amplitude max of y_fit
        p0 = [-58, 3, max(y_fit)]

        params, cov = curve_fit(gaussian, x_fit, y_fit, p0=p0)
        mu_fit, sigma_fit, A_fit = params

        print(
            f"Fitted Gaussian parameters:\n Mean (mu): {mu_fit:.2f} mV\n Std dev (sigma): {sigma_fit:.2f} mV\n Amplitude: {A_fit:.2f} %")

        # Step 7: Plot original image and overlay Gaussian fit

        fig, ax = plt.subplots(figsize=(8, 5))

        # Show original image with extent to map pixels to mV and fraction %
        ax.imshow(img_np, cmap='gray_r', aspect='auto',
                  extent=[V_min, V_max, 0, 6])

        # Plot extracted intensity profile as dots
        ax.plot(x_vals, intensity_profile, 'r.', label='Extracted data')

        # Plot Gaussian fit curve
        x_smooth = np.linspace(-65, -50, 300)
        y_smooth = gaussian(x_smooth, *params)
        ax.plot(x_smooth, y_smooth, 'b-', label='Gaussian fit')

        ax.set_xlabel('Membrane potential (mV)')
        ax.set_ylabel('Fraction of time (%)')
        ax.set_title('Gaussian fit to Up-state membrane potential distribution')
        ax.legend()
        ax.set_xlim(V_min, V_max)
        ax.set_ylim(0, 6)

        plt.show()

    def test_fit_petersen_2013_fig_1_c(self, figure="fig_1_c.png", fitted_amp = 1.2):


        # ---- Gaussian function


        # ---- Load image
        img = cv2.imread(figure)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        # ---- Detect curve using threshold
        _, thresh = cv2.threshold(gray, 200, 255, cv2.THRESH_BINARY_INV)

        # ---- Extract curve points
        y_idx, x_idx = np.where(thresh > 0)

        # invert y-axis (image coordinates)
        y_idx = img.shape[0] - y_idx

        # ---- Sort by x
        order = np.argsort(x_idx)
        x = x_idx[order]
        y = y_idx[order]

        # ---- Normalize (approximate scaling)
        x_data = -80 + (x - x.min()) / (x.max() - x.min()) * 40
        y_data = y / y.max() * 8

        # ---- Select right peak
        mask = x_data > -65
        x_right = x_data[mask]
        y_right = y_data[mask]

        # ---- Fit gaussian
        popt, _ = curve_fit(gaussian, x_right, y_right, p0=[5, -60, 3])

        A, mu, sigma = popt

        print("Mean:", mu)
        print("Sigma:", sigma)

        # ---- Plot
        plt.figure(figsize=(6, 4))

        # background image
        plt.imshow(img, extent=[-80, -40, 0, 8], aspect='auto')

        # gaussian curve
        x_fit = np.linspace(-80, -40, 500)
        y_fit =  fitted_amp * gaussian(x_fit, A * fitted_amp, mu, sigma)

        plt.plot(x_fit, y_fit, 'r', linewidth=3, label=f"Gaussian μ={mu:.2f}, σ={sigma:.2f}")

        plt.legend()
        plt.xlabel("x")
        plt.ylabel("density")

        plt.show()

    def test_fit_petersen_2013_fig_1_a(self):
        self.test_fit_petersen_2013_fig_1_c(figure="img.png", fitted_amp = 1)

    def test_try_manual_fit(self):
        plot_gaussian_on_top_of_image(mean=-52.2, sigma=2.9, amp=1.9, figure="img.png")





if __name__ == '__main__':
    unittest.main()
