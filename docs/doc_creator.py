import html2text
import requests
import argparse
import os
from pathlib import Path

def get_unique_filename(base_path):
    """
    Generate a unique filename in the format doc.md, doc1.md, doc2.md, etc.
    
    Parameters:
    base_path (Path): Directory where the file will be created
    
    Returns:
    Path: Unique file path
    """
    counter = 0
    while True:
        suffix = str(counter) if counter > 0 else ""
        filename = f"doc{suffix}.md"
        file_path = base_path / filename
        
        if not file_path.exists():
            return file_path
        counter += 1

def url_to_markdown(url, output_path=None):
    """
    Convert webpage content to markdown and save to file.
    
    Parameters:
    url (str): URL of the webpage to convert
    output_path (Path, optional): Custom output path
    
    Returns:
    Path: Path to the created markdown file
    """
    try:
        # Get webpage content
        response = requests.get(url)
        response.raise_for_status()
        
        # Convert HTML to markdown
        converter = html2text.HTML2Text()
        converter.ignore_links = False
        converter.ignore_images = False
        markdown_content = converter.handle(response.text)
        
        # Generate output path if not provided
        if output_path is None:
            base_path = Path(__file__).parent
            output_path = get_unique_filename(base_path)
        
        # Write to file
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(markdown_content)
            
        return output_path
        
    except requests.RequestException as e:
        print(f"Error fetching URL: {e}")
        return None
    except Exception as e:
        print(f"Error processing content: {e}")
        return None

def main():
    parser = argparse.ArgumentParser(description='Convert webpage content to markdown')
    parser.add_argument('url', help='URL of the webpage to convert')
    parser.add_argument('-o', '--output', help='Output file path (optional)')
    
    args = parser.parse_args()
    
    output_path = Path(args.output) if args.output else None
    result_path = url_to_markdown(args.url, output_path)
    
    if result_path:
        print(f"Successfully created markdown file: {result_path}")
    else:
        print("Failed to create markdown file")

if __name__ == "__main__":
    main()
