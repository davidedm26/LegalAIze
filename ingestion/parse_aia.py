"""  
This module contains functions to parse the AIA document and extract relevant information."""


import json
from pathlib import Path

from typing import Dict, Any, List, Optional


def read_html_file(filepath: str) -> str:
	"""
	Reads the content of an HTML file and returns it as a string.
	Args:
		filepath (str): Path to the HTML file.
	Returns:
		str: Content of the HTML file.
	"""
	path = Path(filepath)
	with path.open('r', encoding='utf-8') as f:
		return f.read()

def clean_section_content(content: str) -> str:
	"""
	Cleans and normalizes the content of a section: removes excessive newlines, trims whitespace,
	and strips unwanted leading/trailing characters.
	Args:
		content (str): Raw section content.
	Returns:
		str: Cleaned content.
	"""
	import re
	# Remove leading/trailing whitespace and newlines
	content = content.strip()
	# Replace multiple newlines with a single newline
	content = re.sub(r'\n{2,}', '\n', content)
	# Remove excessive spaces
	content = re.sub(r'[ \t]+', ' ', content)
	# Remove leading/trailing numbers or parenthesis if present (e.g., "(2)")
	content = re.sub(r'^\(\d+\)\s*', '', content)
	# Remove leading/trailing non-breaking spaces and similar
	content = content.strip('\u200b\xa0 ')
	return content

def parse_ai_act_html(html_content: str) -> List[Dict[str, Any]]:
	"""
	Extracts Recital, Article, and Annex sections from the AI Act HTML using the actual tag structure.
	Each object contains at least: name, type, content, and optionally title.
	Args:
		html_content (str): HTML content of the AI Act document.
	Returns:
		List[Dict[str, Any]]: List of structured objects for each section.
	"""
	from bs4 import BeautifulSoup

	soup = BeautifulSoup(html_content, 'html.parser')
	sections = []

	# --- Recitals ---
	# Recitals are in <div class="eli-subdivision" id="rct_X">, X=1,2,...
	for div in soup.find_all('div', class_='eli-subdivision'):
		if div.has_attr('id') and div['id'].startswith('rct_'):
			name = div['id'].replace('_', ' ').upper()  # e.g. RCT 1
			section_type = 'recital'
			# Try to extract the first <p> as title if it looks like a title
			ps = div.find_all('p')
			title = None
			content = div.get_text(separator='\n').strip()
			if ps and ps[0].get_text().isupper() and len(ps[0].get_text()) < 120:
				title = ps[0].get_text().strip()
				content_body = '\n'.join([p.get_text().strip() for p in ps[1:]]).strip()
			else:
				content_body = content
			section = {
				'name': name,
				'type': section_type,
				'content': clean_section_content(content_body),
			}
			if title:
				section['title'] = title
			sections.append(section)


	# --- Articles ---
	# Articles are in <div class="eli-subdivision" id="art_X">, X=1,2,...
	for div in soup.find_all('div', class_='eli-subdivision'):
		if div.has_attr('id') and div['id'].startswith('art_'):
			name = div['id']  # e.g. art_1 (keep original, with underscore)
			section_type = 'article'
			# Try to extract the title from a child div with class 'eli-title' and matching id
			title = None
			title_div = div.find('div', class_='eli-title')
			if title_div:
				# Get the first <p> inside the title div
				p_title = title_div.find('p')
				if p_title:
					title = p_title.get_text().strip()
			# Extract all text content except the title
			# Remove the title div from a copy of the article div to avoid duplication
			div_copy = BeautifulSoup(str(div), 'html.parser')
			for tdiv in div_copy.find_all('div', class_='eli-title'):
				tdiv.decompose()
			content = div_copy.get_text(separator='\n').strip()
			section = {
				'name': name,
				'type': section_type,
				'content': clean_section_content(content),
			}
			if title:
				section['title'] = title
			sections.append(section)

	# --- Annexes ---
	# Annexes are in <div class="eli-container" id="anx_X">, X=I,II,...
	for div in soup.find_all('div', class_='eli-container'):
		if div.has_attr('id') and div['id'].startswith('anx_'):
			name = div['id']
			section_type = 'annex'
			# Extract all <p> and <span> text (for section headers, etc.)
			content_lines = []
			for p in div.find_all('p'):
				txt = p.get_text().strip()
				if txt:
					content_lines.append(txt)
			# Extract all tables as numbered lists
			for table in div.find_all('table'):
				for tr in table.find_all('tr'):
					tds = tr.find_all(['td', 'th'])
					if len(tds) >= 3:
						num = tds[1].get_text().strip()
						desc = tds[2].get_text().strip()
						if num and desc:
							content_lines.append(f"{num}. {desc}")
			# Remove duplicate lines and join
			content_body = '\n'.join(dict.fromkeys(content_lines))
			# Title: first uppercase <p> or <span> (not just first <p>)
			title = None
			for tag in div.find_all(['p', 'span']):
				t = tag.get_text().strip()
				if t.isupper() and len(t) < 120:
					title = t
					break
			# Remove title from content if present as first line
			if title and content_lines and content_lines[0] == title:
				content_body = '\n'.join(content_lines[1:])
			section = {
				'name': name,
				'type': section_type,
				'content': clean_section_content(content_body),
			}
			if title:
				section['title'] = title
			sections.append(section)

	return sections



def parse_ai_act_file_to_json(filepath: str, output_path: Optional[str] = None) -> List[Dict[str, Any]]:
	"""
	Main function: reads the HTML file, extracts sections, and optionally saves to JSON.
	Args:
		filepath (str): Path to the HTML file.
		output_path (Optional[str]): Path to save the JSON file (if provided).
	Returns:
		List[Dict[str, Any]]: List of extracted section objects.
	"""

	html_content = read_html_file(filepath)
	sections = parse_ai_act_html(html_content)
	if output_path:
		out_path = Path(output_path)
		with out_path.open('w', encoding='utf-8') as f:
			json.dump(sections, f, ensure_ascii=False, indent=2)
	return sections


