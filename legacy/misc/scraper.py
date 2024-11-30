import requests
from bs4 import BeautifulSoup
from urllib.parse import urljoin
import os

def scrape_agenda_pages(base_url, output_file):
    """
    Scrape content from pages containing 'agenda' in their URLs.
    
    Args:
        base_url (str): The starting URL to scrape
        output_file (str): Path to save the scraped content
    """
    visited_urls = set()
    agenda_pages = set()
    
    def get_page_content(url):
        """Helper function to get content from a single page."""
        try:
            response = requests.get(url)
            response.raise_for_status()
            return response.text
        except requests.RequestException as e:
            print(f"Error fetching {url}: {e}")
            return None

    def is_html_url(url):
        """Check if URL likely points to HTML content."""
        # List of common web page extensions
        html_extensions = ['.html', '.htm', '/']
        # List of extensions to skip
        skip_extensions = ['.ics', '.pdf', '.doc', '.docx', '.xls', '.xlsx', '.zip', '.rar']
        
        lower_url = url.lower()
        # Skip if URL ends with a non-HTML extension
        if any(lower_url.endswith(ext) for ext in skip_extensions):
            return False
        # Accept if URL ends with HTML extension or no extension
        return any(lower_url.endswith(ext) for ext in html_extensions) or '.' not in url.split('/')[-1]

    def extract_links(soup, current_url):
        """Extract all relevant links from the page."""
        links = set()
        for a in soup.find_all('a', href=True):
            href = a['href']
            full_url = urljoin(current_url, href)
            # Only include links that are part of the same domain
            if full_url.startswith(base_url):
                links.add(full_url)
                # If the URL contains 'agenda', add it to agenda_pages
                if 'agenda' in full_url.lower() and is_html_url(full_url):
                    agenda_pages.add(full_url)
        return links

    def process_page(url):
        """Process a single page and its linked pages recursively."""
        if url in visited_urls:
            return
        
        visited_urls.add(url)
        print(f"Scanning: {url}")
        
        # Only process HTML pages
        if not is_html_url(url):
            return
        
        content = get_page_content(url)
        if not content:
            return
        
        try:
            soup = BeautifulSoup(content, 'html.parser')
        except Exception as e:
            print(f"Error parsing {url}: {e}")
            return
        
        # Find and process all links
        links = extract_links(soup, url)
        for link in links:
            process_page(link)

    def save_agenda_content():
        """Save content from agenda pages to file."""
        with open(output_file, 'w', encoding='utf-8') as f:
            f.write(f"Content from agenda pages at {base_url}\n")
            f.write("=" * 50 + "\n\n")
            
            for url in sorted(agenda_pages):
                print(f"Processing agenda page: {url}")
                content = get_page_content(url)
                if content:
                    try:
                        soup = BeautifulSoup(content, 'html.parser')
                        main_content = soup.find('main') or soup.find('article') or soup.find('div', class_='content')
                        
                        if main_content:
                            f.write(f"\n\n=== {url} ===\n\n")
                            f.write(main_content.get_text(strip=True, separator='\n'))
                    except Exception as e:
                        print(f"Error processing content from {url}: {e}")

    # Start processing from the base URL
    print("Scanning website for agenda pages...")
    process_page(base_url)
    
    # Save content from agenda pages
    print(f"\nFound {len(agenda_pages)} agenda pages.")
    save_agenda_content()
    
    print(f"\nScraping completed. Scanned {len(visited_urls)} pages total.")
    print(f"Content from {len(agenda_pages)} agenda pages saved to: {output_file}")
    print("\nAgenda pages found:")
    for url in sorted(agenda_pages):
        print(f"- {url}")

if __name__ == "__main__":
    base_url = "https://hl7norway.github.io/best-practice/"
    output_file = "agenda_content.txt"
    
    scrape_agenda_pages(base_url, output_file)
