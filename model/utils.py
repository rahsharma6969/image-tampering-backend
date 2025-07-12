from pdf2image import convert_from_bytes

def convert_pdf_to_images(file_storage):
    # file_storage is already a FileStorage object, so read() once
    pdf_bytes = file_storage.read()  
    images = convert_from_bytes(pdf_bytes) 
    return images
