import os # import os module for file system operations
from PIL import Image # import Pillow module to work with images
import concurrent.futures # import concurrent.futures to perform multi-threading

def resizeAndCompress(file):
    # Get the current working directory and join it with the file name
    filepath = os.path.join(os.getcwd(), file)
    picture = Image.open(filepath) # Open the image file
    
    # Resize the image to half of its original size
    width, height = picture.size
    picture.thumbnail((width//2, height//2))

    # Save the compressed image with custom quality
    picture.save("Compressed_" + file, "JPEG", optimize=True, quality=30)

    return file

def main():
    formats = ('.jpg', '.jpeg') # tuple containing image file extensions
    files_to_compress = [file for file in os.listdir() if os.path.splitext(file)[1].lower() in formats] # list comprehension to get all image files in current directory

    # Use ThreadPoolExecutor to compress multiple files simultaneously
    with concurrent.futures.ThreadPoolExecutor(max_workers=4) as executor:
        results = list(executor.map(resizeAndCompress, files_to_compress))

    print(f"Compressed {len(results)} files.") # print the number of files that were compressed

if __name__ == "__main__":
    main()
    
