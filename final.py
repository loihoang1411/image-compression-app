import customtkinter as ctk
import tkinter as tk
from customtkinter import filedialog
from PIL import Image, ImageTk
import numpy as np
import pywt
import os
import time
from sklearn.decomposition import PCA
import skimage
from sklearn.metrics import mean_squared_error

# Define global variables for the original and compressed images and their dimensions
original_image = None
original_image_width = 0
original_image_height = 0
original_image_size_kb = 0
compressed_image = None
compressed_image_width = 0
compressed_image_height = 0
compressed_image_size_kb = 0
compression_time = 0

def open_image():
    global original_image, original_image_width, original_image_height, original_image_size_kb, file_extension, image_orig
    file_path = filedialog.askopenfilename(filetypes=[("Image Files", "*.png *.jpg *.jpeg *.gif *.bmp *.ppm *.pgm *.dng")])
    if file_path:
        file_extension = os.path.splitext(file_path)[1].lower()
        original_image = Image.open(file_path)
        original_image_gray = original_image.convert("L")
        original_image_width, original_image_height = original_image.size
        original_image_size_kb = os.path.getsize(file_path) / 1024
        original_image_display(original_image_gray)
        update_info_labels_original()
        image_orig = skimage.io.imread(file_path)  # Load image
        # image_orig_gray = skimage.color.rgb2gray(image_orig_first)
        # image_orig = skimage.transform.resize(image_orig_gray, image_orig_first.shape)

def calculate_image_size_SVD(image_array):
    global compressed_size_kb_SVD
    # Convert the image array to PIL Image
    compressed_image = Image.fromarray(image_array.astype(np.uint8))
    # Save the image to a temporary file
    temp_file_name = "compressed_temp" + file_extension
    # Save the image to a temporary file
    compressed_image.save(temp_file_name)
    # Get the size of the saved image file
    compressed_size_kb_SVD = os.path.getsize(temp_file_name) / 1024
    # Delete the temporary file
    os.remove(temp_file_name)
    return compressed_size_kb_SVD

def calculate_image_size_FFT(image_array):
    global compressed_size_kb_FFT
    # Convert the image array to PIL Image
    compressed_image = Image.fromarray(image_array.astype(np.uint8))
    # Save the image to a temporary file
    temp_file_name = "compressed_temp" + file_extension
    # Save the image to a temporary file
    compressed_image.save(temp_file_name)
    # Get the size of the saved image file
    compressed_size_kb_FFT = os.path.getsize(temp_file_name) / 1024
    # Delete the temporary file
    os.remove(temp_file_name)
    return compressed_size_kb_FFT

def calculate_image_size_Wavelet(image_array):
    global compressed_size_kb_Wavelet
    # Convert the image array to PIL Image
    compressed_image = Image.fromarray(image_array.astype(np.uint8))
    # Save the image to a temporary file
    temp_file_name = "compressed_temp" + file_extension
    # Save the image to a temporary file
    compressed_image.save(temp_file_name)
    # Get the size of the saved image file
    compressed_size_kb_Wavelet = os.path.getsize(temp_file_name) / 1024
    # Delete the temporary file
    os.remove(temp_file_name)
    return compressed_size_kb_Wavelet

def calculate_image_size_PCA(image_array):
    global compressed_size_kb_PCA
    # Convert the image array to PIL Image
    compressed_image = Image.fromarray(image_array.astype(np.uint8))
    # Save the image to a temporary file
    temp_file_name = "compressed_temp" + file_extension
    # Save the image to a temporary file
    compressed_image.save(temp_file_name)
    # Get the size of the saved image file
    compressed_size_kb_PCA = os.path.getsize(temp_file_name) / 1024
    # Delete the temporary file
    os.remove(temp_file_name)
    return compressed_size_kb_PCA

# Get the Mean Squared Error
def get_mse(original_img_arr, decoded_img_arr):
    # the 'Mean Squared Error' between the two images is the
    # sum of the squared difference between the two images;
    # Note: the two images must have the same dimension
    #mse = np.sum((original_img_arr.astype("float") - decoded_img_arr.astype("float")) ** 2)
    #mse /= float(original_img_arr.shape[0] * decoded_img_arr.shape[1])
    mse = mean_squared_error(original_img_arr, decoded_img_arr)
    # Return the mean-squared-error
    return mse

# Get the PSNR value
def get_psnr(original_img_arr, decoded_img_arr, ignore=None):
    # Calculate the maximum data value
    maximumDataValue = np.maximum(np.amax(original_img_arr), np.amax(decoded_img_arr))
    # Make sure that the provided data sets are numpy ndarrays, if not
    # convert them and flatten the data sets for analysis
    if type(original_img_arr).__module__ != np.__name__:
        d1 = np.asarray(original_img_arr).flatten()
    else:
        d1 = original_img_arr.flatten()
    if type(decoded_img_arr).__module__ != np.__name__:
        d2 = np.asarray(decoded_img_arr).flatten()
    else:
        d2 = decoded_img_arr.flatten()
    # Make sure that the provided data sets are the same size
    if d1.size != d2.size:
        raise ValueError('Provided datasets must have the same size/shape')
    # Check if the provided data sets are identical, and if so, return an
    # infinite peak-signal-to-noise ratio
    if np.array_equal(d1, d2):
        return float('inf')
    # If specified, remove the values to ignore from the analysis and compute
    # the element-wise difference between the data sets
    if ignore is not None:
        index = np.intersect1d(np.where(d1 != ignore)[0],
                               np.where(d2 != ignore)[0])
        error = d1[index].astype(np.float64) - d2[index].astype(np.float64)
    else:
        error = d1.astype(np.float64) - d2.astype(np.float64)
    # Compute the mean-squared error
    meanSquaredError = np.sum(error ** 2) / error.size
    # Return the peak-signal-to-noise ratio
    return 10.0 * np.log10(maximumDataValue ** 2 / meanSquaredError)

def original_image_display(image):
    display_width = 275
    display_height = display_width * original_image_height / original_image_width
    image.thumbnail((display_width, display_height))
    photo = ImageTk.PhotoImage(image)
    original_image_label.configure(image=photo, width=display_width, height=display_height)
    original_image_label.image = photo

def compressed_image_display_SVD(image):
    if image is not None:
        display_width = 275
        display_height = display_width * original_image_height / original_image_width
        image.thumbnail((display_width, display_height))
        photo = ImageTk.PhotoImage(image)
        compressed_image_label_SVD.configure(image=photo, width=display_width, height=display_height)
        compressed_image_label_SVD.image = photo
        # Calculate compression ratio and compression factor SVD
        global compression_ratio_SVD 
        global compression_factor_SVD
        compression_ratio_SVD = compressed_size_kb_SVD / original_image_size_kb
        compression_factor_SVD = original_image_size_kb / compressed_size_kb_SVD
    else:
        compressed_image_label_SVD.configure(image=None)

def compressed_image_display_FFT(image):
    if image is not None:
        display_width = 275
        display_height = display_width * original_image_height / original_image_width
        image.thumbnail((display_width, display_height))
        photo = ImageTk.PhotoImage(image)
        compressed_image_label_FFT.configure(image=photo, width=display_width, height=display_height)
        compressed_image_label_FFT.image = photo
        # Calculate compression ratio and compression factor FFT
        global compression_ratio_FFT 
        global compression_factor_FFT
        compression_ratio_FFT = compressed_size_kb_FFT / original_image_size_kb
        compression_factor_FFT = original_image_size_kb / compressed_size_kb_FFT
    else:
        compressed_image_label_FFT.configure(image=None)

def compressed_image_display_Wavelet(image):
    if image is not None:
        display_width = 275
        display_height = display_width * original_image_height / original_image_width
        image.thumbnail((display_width, display_height))
        photo = ImageTk.PhotoImage(image)
        compressed_image_label_Wavelet.configure(image=photo, width=display_width, height=display_height)
        compressed_image_label_Wavelet.image = photo
        # Calculate compression ratio and compression factor Wavelet
        global compression_ratio_Wavelet 
        global compression_factor_Wavelet
        compression_ratio_Wavelet = compressed_size_kb_Wavelet / original_image_size_kb
        compression_factor_Wavelet = original_image_size_kb / compressed_size_kb_Wavelet
    else:
        compressed_image_label_Wavelet.configure(image=None)

def compressed_image_display_PCA(image):
    if image is not None:
        display_width = 275
        display_height = display_width * original_image_height / original_image_width
        image.thumbnail((display_width, display_height))
        photo = ImageTk.PhotoImage(image)
        compressed_image_label_PCA.configure(image=photo, width=display_width, height=display_height)
        compressed_image_label_PCA.image = photo
        # Calculate compression ratio and compression factor PCA
        global compression_ratio_PCA 
        global compression_factor_PCA
        compression_ratio_PCA = compressed_size_kb_PCA / original_image_size_kb
        compression_factor_PCA = original_image_size_kb / compressed_size_kb_PCA
    else:
        compressed_image_label_PCA.configure(image=None)

def compress_image_SVD():
    global compressed_image, compressed_image_width, compressed_image_height, compressed_image_size_kb, compression_time, mse_SVD, psnr_SVD, rank_value
    if original_image:
        start_time = time.time()
        # Get the value from the input
        rank_value = int(f"{name_entry_SVD.get()}")
        # Convert original_image to grayscale
        original_image_gray = original_image.convert("L")
        X = np.array(original_image_gray)
        # Perform SVD compression
        U, S, VT = np.linalg.svd(X, full_matrices=False)
        S = np.diag(S)
        # Choose a rank for compression (you can adjust these values)
        r = rank_value
        # Construct approximate image with the chosen rank
        Xapprox = U[:, :r] @ S[0:r, :r] @ VT[:r, :]
        # Calculate the size of the compressed image
        compressed_image_size_kb = calculate_image_size_SVD(Xapprox)
        # Convert the compressed array back to PIL Image
        compressed_image = Image.fromarray((Xapprox).astype(np.uint8))
        # Display the compressed image
        compressed_image_display_SVD(compressed_image)
        # Calculate MSE for SVD method:
        mse_SVD = get_mse (X, Xapprox)
        # Calclulate PSNR for SVD method
        psnr_SVD = get_psnr (X, Xapprox)
        # Update information labels
        compressed_image_width, compressed_image_height = compressed_image.size
        compression_time = time.time() - start_time
        update_info_labels_SVD()

def compress_image_FFT():
    global compressed_image, compressed_image_width, compressed_image_height, compressed_image_size_kb, compression_time, mse_FFT, psnr_FFT, keep_FFT
    if original_image:
        start_time = time.time()
        # Get the value from the input
        keep_FFT = float(f"{name_entry_FFT.get()}")
        keep_FFT = keep_FFT/100
        # Convert original_image to grayscale
        original_image_gray = original_image.convert("L")
        B = np.array(original_image_gray)
        # FFT-based compression
        Bt = np.fft.fft2(B)
        Btsort = np.sort(np.abs(Bt.reshape(-1)))  # sort by magnitude
        #for keep_FFT in (0.1, 0.05, 0.01, 0.002):
        thresh = Btsort[int(np.floor((1 - keep_FFT) * len(Btsort)))]
        ind = np.abs(Bt) > thresh
        Btlow = Bt * ind
        Alow = np.fft.ifft2(Btlow).real  # Compressed image
            # Calculate the size of the compressed image
        compressed_image_size_kb = calculate_image_size_FFT(Alow)
            # Display the compressed image
        compressed_image = Image.fromarray((Alow).astype(np.uint8))    
        compressed_image_display_FFT(compressed_image)
        # Calculate MSE for FFT method:
        mse_FFT = get_mse (B, Alow)
        # Calclulate PSNR for FFT method
        psnr_FFT = get_psnr (B, Alow)
            # Update information labels
        compressed_image_width, compressed_image_height = compressed_image.size
        compression_time = time.time() - start_time
        update_info_labels_FFT()

def compress_image_Wavelet():
    global compressed_image, compressed_image_width, compressed_image_height, compressed_image_size_kb, compression_time, mse_Wavelet, psnr_Wavelet, keep_Wavelet
    if original_image:
        start_time = time.time()
        # Get the value from the input
        keep_Wavelet = float(f"{name_entry_Wavelet.get()}")
        keep_Wavelet = keep_Wavelet/100
        # Convert original_image to grayscale
        original_image_gray = original_image.convert("L")
        B = np.array(original_image_gray)
        # Wavelet-based compression
        n = 4
        w = 'db1'
        coeffs = pywt.wavedec2(B, wavelet=w, level=n)
        coeffs_arr, coeffs_slices = pywt.coeffs_to_array(coeffs)
        Csort = np.sort(np.abs(coeffs_arr.reshape(-1)))
        # keep_Wavelet
        thresh = Csort[int(np.floor((1 - keep_Wavelet) * len(Csort)))]
        ind = np.abs(coeffs_arr) > thresh
        Cfilt = coeffs_arr * ind
        coeffs_filt = pywt.array_to_coeffs(Cfilt, coeffs_slices, output_format='wavedec2')
        # Reconstruction
        Arecon = pywt.waverec2(coeffs_filt, wavelet=w)
        # Calculate the size of the compressed image
        compressed_image_size_kb = calculate_image_size_Wavelet(Arecon)
        # Display the compressed image
        compressed_image = Image.fromarray(Arecon.astype('uint8'))
        compressed_image_display_Wavelet(compressed_image)
        # Calculate MSE for Wavelet method:
        # Ensure that the size of Arecon matches the size of the original image
        Arecon = Arecon[:B.shape[0], :B.shape[1]]
        mse_Wavelet = get_mse (B, Arecon)
        # Calclulate PSNR for Wavelet method
        psnr_Wavelet = get_psnr (B, Arecon)
        # Update information labels
        compressed_image_width, compressed_image_height = compressed_image.size
        compression_time = time.time() - start_time
        update_info_labels_Wavelet()

def compress_image_PCA():
    global compressed_image, compressed_image_width, compressed_image_height, compressed_image_size_kb, compression_time, mse_PCA, psnr_PCA, n_components_pca
    if original_image:
        start_time = time.time()
        n_components_pca = int(f"{name_entry_PCA.get()}")
        image = skimage.color.rgb2gray(image_orig)
        image = skimage.img_as_ubyte(image)
        # Perform PCA compression
        pca = PCA(n_components=n_components_pca)
        Xapprox = pca.fit_transform(image)
        Xapprox = pca.inverse_transform(Xapprox)
        # Calculate the size of the compressed image
        compressed_image_size_kb = calculate_image_size_PCA(Xapprox)
        # Xapprox_size = np.prod(Xapprox.shape) * np.dtype(Xapprox.dtype).itemsize / 1024
        # print (Xapprox_size)
        # Convert the compressed array back to PIL Image
        compressed_image = Image.fromarray((Xapprox).astype(np.uint8))
        # Display the compressed image
        compressed_image_display_PCA(compressed_image)
        # Calculate MSE for PCA method
        mse_PCA = get_mse(image, Xapprox)
        # Calculate PSNR for PCA method
        psnr_PCA = get_psnr(image, Xapprox)
        # Update information labels
        compressed_image_width, compressed_image_height = compressed_image.size
        compression_time = time.time() - start_time
        update_info_labels_PCA()

def update_info_labels_original():
    original_width_label.configure(text=f"Width: {original_image_width}", font=("Helvetica", 14))
    original_width_label.grid(row=3, column=0, padx=10, pady=5, sticky="w")
    original_height_label.configure(text=f"Height: {original_image_height}", font=("Helvetica", 14))
    original_height_label.grid(row=4, column=0, padx=10, pady=5, sticky="w")
    original_image_size_label.configure(text=f"Size on Disk: {original_image_size_kb:.2f} KB", font=("Helvetica", 14))
    original_image_size_label.grid(row=5, column=0, padx=10, pady=5, sticky="w")

def update_info_labels_SVD():
    compressed_image_size_label_SVD.configure(text=f"Size on Disk: {compressed_size_kb_SVD:.2f} KB", font=("Helvetica", 14))
    compressed_image_size_label_SVD.grid(row=5, column=0, padx=10, pady=5, sticky="w")
    compressed_compression_ratio_label_SVD.configure(text=f"Compression Ratio: {compression_ratio_SVD:.2f}", font=("Helvetica", 14))
    compressed_compression_ratio_label_SVD.grid(row=6, column=0, padx=10, pady=5, sticky="w")
    compressed_compression_factor_label_SVD.configure(text=f"Compression Factor: {compression_factor_SVD:.2f}", font=("Helvetica", 14))
    compressed_compression_factor_label_SVD.grid(row=7, column=0, padx=10, pady=5, sticky="w")
    compressed_mse_label_SVD.configure(text=f"MSE: {mse_SVD:.2f}", font=("Helvetica", 14))
    compressed_mse_label_SVD.grid(row=8, column=0, padx=10, pady=5, sticky="w")
    compressed_psnr_label_SVD.configure(text=f"PSNR: {psnr_SVD:.2f}", font=("Helvetica", 14))
    compressed_psnr_label_SVD.grid(row=9, column=0, padx=10, pady=5, sticky="w")
    compression_time_label_SVD.configure(text=f"Compression Time: {compression_time:.2f} seconds", font=("Helvetica", 14))
    compression_time_label_SVD.grid(row=10, column=0, padx=10, pady=5, sticky="w")

def update_info_labels_FFT():
    compressed_image_size_label_FFT.configure(text=f"Size on Disk: {compressed_size_kb_FFT:.2f} KB", font=("Helvetica", 14))
    compressed_image_size_label_FFT.grid(row=5, column=0, padx=10, pady=5, sticky="w")
    compressed_compression_ratio_label_FFT.configure(text=f"Compression Ratio: {compression_ratio_FFT:.2f}", font=("Helvetica", 14))
    compressed_compression_ratio_label_FFT.grid(row=6, column=0, padx=10, pady=5, sticky="w")
    compressed_compression_factor_label_FFT.configure(text=f"Compression Factor: {compression_factor_FFT:.2f}", font=("Helvetica", 14))
    compressed_compression_factor_label_FFT.grid(row=7, column=0, padx=10, pady=5, sticky="w")
    compressed_mse_label_FFT.configure(text=f"MSE: {mse_FFT:.2f}", font=("Helvetica", 14))
    compressed_mse_label_FFT.grid(row=8, column=0, padx=10, pady=5, sticky="w")
    compressed_psnr_label_FFT.configure(text=f"PSNR: {psnr_FFT:.2f}", font=("Helvetica", 14))
    compressed_psnr_label_FFT.grid(row=9, column=0, padx=10, pady=5, sticky="w")
    compression_time_label_FFT.configure(text=f"Compression Time: {compression_time:.2f} seconds", font=("Helvetica", 14))
    compression_time_label_FFT.grid(row=10, column=0, padx=10, pady=5, sticky="w")

def update_info_labels_Wavelet():
    compressed_image_size_label_Wavelet.configure(text=f"Size on Disk: {compressed_size_kb_Wavelet:.2f} KB", font=("Helvetica", 14))
    compressed_image_size_label_Wavelet.grid(row=5, column=0, padx=10, pady=5, sticky="w")
    compressed_compression_ratio_label_Wavelet.configure(text=f"Compression Ratio: {compression_ratio_Wavelet:.2f}", font=("Helvetica", 14))
    compressed_compression_ratio_label_Wavelet.grid(row=6, column=0, padx=10, pady=5, sticky="w")
    compressed_compression_factor_label_Wavelet.configure(text=f"Compression Factor: {compression_factor_Wavelet:.2f}", font=("Helvetica", 14))
    compressed_compression_factor_label_Wavelet.grid(row=7, column=0, padx=10, pady=5, sticky="w")
    compressed_mse_label_Wavelet.configure(text=f"MSE: {mse_Wavelet:.2f}", font=("Helvetica", 14))
    compressed_mse_label_Wavelet.grid(row=8, column=0, padx=10, pady=5, sticky="w")
    compressed_psnr_label_Wavelet.configure(text=f"PSNR: {psnr_Wavelet:.2f}", font=("Helvetica", 14))
    compressed_psnr_label_Wavelet.grid(row=9, column=0, padx=10, pady=5, sticky="w")
    compression_time_label_Wavelet.configure(text=f"Compression Time: {compression_time:.2f} seconds", font=("Helvetica", 14))
    compression_time_label_Wavelet.grid(row=10, column=0, padx=10, pady=5, sticky="w")

def update_info_labels_PCA():
    compressed_image_size_label_PCA.configure(text=f"Size on Disk: {compressed_size_kb_PCA:.2f} KB", font=("Helvetica", 14))
    compressed_image_size_label_PCA.grid(row=5, column=0, padx=10, pady=5, sticky="w")
    compressed_compression_ratio_label_PCA.configure(text=f"Compression Ratio: {compression_ratio_PCA:.2f}", font=("Helvetica", 14))
    compressed_compression_ratio_label_PCA.grid(row=6, column=0, padx=10, pady=5, sticky="w")
    compressed_compression_factor_label_PCA.configure(text=f"Compression Factor: {compression_factor_PCA:.2f}", font=("Helvetica", 14))
    compressed_compression_factor_label_PCA.grid(row=7, column=0, padx=10, pady=5, sticky="w")
    compressed_mse_label_PCA.configure(text=f"MSE: {mse_PCA:.2f}", font=("Helvetica", 14))
    compressed_mse_label_PCA.grid(row=8, column=0, padx=10, pady=5, sticky="w")
    compressed_psnr_label_PCA.configure(text=f"PSNR: {psnr_PCA:.2f}", font=("Helvetica", 14))
    compressed_psnr_label_PCA.grid(row=9, column=0, padx=10, pady=5, sticky="w")
    compression_time_label_PCA.configure(text=f"Compression Time: {compression_time:.2f} seconds", font=("Helvetica", 14))
    compression_time_label_PCA.grid(row=10, column=0, padx=10, pady=5, sticky="w")

ctk.set_appearance_mode("Light")
# Create the main window
root = ctk.CTk()
root.resizable(height=False, width=False)
root.title("Image Compression App")
root.geometry("1750x700")


# Create a frame for the original image
original_frame = ctk.CTkFrame(root, fg_color="lightgreen")
original_frame.grid(row=1, column=0, padx=10, pady=10)
original_label = ctk.CTkLabel(original_frame, text="Original Image", font=("Helvetica", 20))
original_label.grid(row=0, column=0)
original_image_label = ctk.CTkLabel(original_frame, text="Original image")
original_image_label.grid(row=1, column=0)

# Create a frame for the compressed image_SVD
compressed_frame_SVD = ctk.CTkFrame(root, fg_color="lightgreen")
compressed_frame_SVD.grid(row=1, column=1, padx=10, pady=10)
compressed_label_SVD = ctk.CTkLabel(compressed_frame_SVD, text="SVD", font=("Helvetica", 20))
compressed_label_SVD.grid(row=0, column=0)
    # Create a name_entry for SVD:
name_entry_SVD = ctk.CTkEntry(compressed_frame_SVD,
                              placeholder_text="Enter SVD rank: ",
                              width=150,
                              height=25,
                              border_width=2,
                              corner_radius=10)
name_entry_SVD.grid(row=1, column=0)
    # Create buttons to add an image and compress it using SVD method
compress_button_SVD = ctk.CTkButton(compressed_frame_SVD, text="Compress", command=compress_image_SVD)
compress_button_SVD.grid(row=2, column=0)
compressed_image_label_SVD = ctk.CTkLabel(compressed_frame_SVD, text="SVD image")
compressed_image_label_SVD.grid(row=3, column=0)

# Create a frame for the compressed image_FFT
compressed_frame_FFT = ctk.CTkFrame(root, fg_color="lightgreen")
compressed_frame_FFT.grid(row=1, column=2, padx=10, pady=10)
compressed_label = ctk.CTkLabel(compressed_frame_FFT, text="FFT", font=("Helvetica", 20))
compressed_label.grid(row=0, column=0)
    # Create a name_entry for FFT:
name_entry_FFT = ctk.CTkEntry(compressed_frame_FFT,
                              placeholder_text="Enter FFT keep(%): ",
                              width=150,
                              height=25,
                              border_width=2,
                              corner_radius=10)
name_entry_FFT.grid(row=1, column=0)
    # Create buttons to add an image and compress it using FFT method
compress_button_FFT = ctk.CTkButton(compressed_frame_FFT, text="Compress", command=compress_image_FFT)
compress_button_FFT.grid(row=2, column=0)
compressed_image_label_FFT = ctk.CTkLabel(compressed_frame_FFT, text="FFT image")
compressed_image_label_FFT.grid(row=3, column=0)


# Create a frame for the compressed image_Wavelet
compressed_frame_Wavelet = ctk.CTkFrame(root, fg_color="lightgreen")
compressed_frame_Wavelet.grid(row=1, column=3, padx=10, pady=10)
compressed_label_Wavelet = ctk.CTkLabel(compressed_frame_Wavelet, text="Wavelet", font=("Helvetica", 20))
compressed_label_Wavelet.grid(row=0, column=0)
    # Create a name_entry for FFT:
name_entry_Wavelet = ctk.CTkEntry(compressed_frame_Wavelet,
                              placeholder_text="Enter Wavelet keep (%): ",
                              width=150,
                              height=25,
                              border_width=2,
                              corner_radius=10)
name_entry_Wavelet.grid(row=1, column=0)
    # Create buttons to add an image and compress it using Wavelet method
compress_button_Wavelet = ctk.CTkButton(compressed_frame_Wavelet, text="Compress", command=compress_image_Wavelet)
compress_button_Wavelet.grid(row=2, column=0)
compressed_image_label_Wavelet = ctk.CTkLabel(compressed_frame_Wavelet, text="Wavelet image")
compressed_image_label_Wavelet.grid(row=3, column=0)

# Create a frame for the compressed image_PCA
compressed_frame_PCA = ctk.CTkFrame(root, fg_color="lightgreen")
compressed_frame_PCA.grid(row=1, column=4, padx=10, pady=10)
compressed_label_PCA = ctk.CTkLabel(compressed_frame_PCA, text="PCA", font=("Helvetica", 20))
compressed_label_PCA.grid(row=0, column=0)
    # Create a name_entry for FFT:
name_entry_PCA = ctk.CTkEntry(compressed_frame_PCA,
                              placeholder_text="Enter PCA components: ",
                              width=150,
                              height=25,
                              border_width=2,
                              corner_radius=10)
name_entry_PCA.grid(row=1, column=0)
    # Create buttons to add an image and compress it using PCA method
compress_button_PCA = ctk.CTkButton(compressed_frame_PCA, text="Compress", command=compress_image_PCA)
compress_button_PCA.grid(row=2, column=0)
compressed_image_label_PCA = ctk.CTkLabel(compressed_frame_PCA, text="PCA image")
compressed_image_label_PCA.grid(row=3, column=0)

# Create labels to display the image sizes
original_size_label = ctk.CTkLabel(original_frame, text="Original Size:")
original_width_label = ctk.CTkLabel(original_frame, text="Width: N/A")
original_height_label = ctk.CTkLabel(original_frame, text="Height: N/A")
original_image_size_label = ctk.CTkLabel(original_frame, text="Size on Disk: N/A")

# Create labels to display the compressed image sizes and compression time SVD:
compressed_size_label_SVD= ctk.CTkLabel(compressed_frame_SVD, text="Compressed Size:")
compressed_image_size_label_SVD = ctk.CTkLabel(compressed_frame_SVD, text="Size on Disk: N/A")
compressed_compression_ratio_label_SVD = ctk.CTkLabel(compressed_frame_SVD, text="Compression Ratio: N/A")
compressed_compression_factor_label_SVD = ctk.CTkLabel(compressed_frame_SVD, text="Compression Factor: N/A")
compressed_mse_label_SVD = ctk.CTkLabel(compressed_frame_SVD, text="Compression MSE: N/A")
compressed_psnr_label_SVD = ctk.CTkLabel(compressed_frame_SVD, text="Compression PSNR: N/A")
compression_time_label_SVD = ctk.CTkLabel(compressed_frame_SVD, text="Compression Time: N/A")

# Create labels to display the compressed image sizes and compression time FFT:
compressed_size_label_FFT = ctk.CTkLabel(compressed_frame_FFT, text="Compressed Size:")
compressed_image_size_label_FFT = ctk.CTkLabel(compressed_frame_FFT, text="Size on Disk: N/A")
compressed_compression_ratio_label_FFT = ctk.CTkLabel(compressed_frame_FFT, text="Compression Ratio: N/A")
compressed_compression_factor_label_FFT = ctk.CTkLabel(compressed_frame_FFT, text="Compression Factor: N/A")
compressed_mse_label_FFT = ctk.CTkLabel(compressed_frame_FFT, text="Compression MSE: N/A")
compressed_psnr_label_FFT = ctk.CTkLabel(compressed_frame_FFT, text="Compression PSNR: N/A")
compression_time_label_FFT = ctk.CTkLabel(compressed_frame_FFT, text="Compression Time: N/A")

# Create labels to display the compressed image sizes and compression time Wavelet:
compressed_size_label_Wavelet = ctk.CTkLabel(compressed_frame_Wavelet, text="Compressed Size:")
compressed_image_size_label_Wavelet = ctk.CTkLabel(compressed_frame_Wavelet, text="Size on Disk: N/A")
compressed_compression_ratio_label_Wavelet = ctk.CTkLabel(compressed_frame_Wavelet, text="Compression Ratio: N/A")
compressed_compression_factor_label_Wavelet = ctk.CTkLabel(compressed_frame_Wavelet, text="Compression Factor: N/A")
compressed_mse_label_Wavelet = ctk.CTkLabel(compressed_frame_Wavelet, text="Compression MSE: N/A")
compressed_psnr_label_Wavelet = ctk.CTkLabel(compressed_frame_Wavelet, text="Compression PSNR: N/A")
compression_time_label_Wavelet= ctk.CTkLabel(compressed_frame_Wavelet, text="Compression Time: N/A")

# Create labels to display the compressed image sizes and compression time PCA:
compressed_size_label_PCA = ctk.CTkLabel(compressed_frame_PCA, text="Compressed Size:")
compressed_image_size_label_PCA = ctk.CTkLabel(compressed_frame_PCA, text="Size on Disk: N/A")
compressed_compression_ratio_label_PCA = ctk.CTkLabel(compressed_frame_PCA, text="Compression Ratio: N/A")
compressed_compression_factor_label_PCA = ctk.CTkLabel(compressed_frame_PCA, text="Compression Factor: N/A")
compressed_mse_label_PCA = ctk.CTkLabel(compressed_frame_PCA, text="Compression MSE: N/A")
compressed_psnr_label_PCA = ctk.CTkLabel(compressed_frame_PCA, text="Compression PSNR: N/A")
compression_time_label_PCA = ctk.CTkLabel(compressed_frame_PCA, text="Compression Time: N/A")

# Create buttons to add an image and compress it
add_image_button = ctk.CTkButton(root, text="Add an Image", command=open_image)

# Layout the widgets
add_image_button.grid(row=0)
# Run the GUI main loop
root.mainloop()
