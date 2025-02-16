import PyPDF2

def extract_text_from_pdf(pdf_path):
    """
    Extract text from a PDF file
    Args:
        pdf_path (str): Path to the PDF file
    Returns:
        str: Extracted text from PDF
    """
    try:
        with open(pdf_path, 'rb') as file:
            pdf_reader = PyPDF2.PdfReader(file)
            
            num_pages = len(pdf_reader.pages)
            
            text = ""
            
            for page_num in range(num_pages):
                page = pdf_reader.pages[page_num]
                text += page.extract_text()
            
            return text
            
    except Exception as e:
        print(f"Error: {str(e)}")
        return None

if __name__ == "__main__":
    pdf_path = "sample.pdf"  # Replace with your PDF file path
    extracted_text = extract_text_from_pdf(pdf_path)
    
    if extracted_text:
        print(extracted_text)