auto = """{{page_code}}

Here is the JSON template for automobile attribute extraction:
{"webpage title": "xx", "automobile model (year)": "xx", "price": "xx", "engine type": "xx", "fuel economy": "xx"}

Please extract the automobile' attributes from the HTML code above following the JSON template. For any unanswerable attributes in the template, set their value to the placeholder "<NULL>".
{{prompt_prefix}}"""

book = """{{page_code}}

Here is the JSON template for book attribute extraction:
{"webpage title": "xx", "book title": "xx", "author": "xx", "isbn_13": "xx", "publisher": "xx", "publication date": "xx"}

Please extract the book' attributes from the HTML code above following the JSON template. For any unanswerable attributes in the template, set their value to the placeholder "<NULL>".
{{prompt_prefix}}"""

camera = """{{page_code}}

Here is the JSON template for camera attribute extraction:
{"webpage title": "xx", "camera model (full)": "xx", "price": "xx", "manufacturer": "xx"}

Please extract the camera' attributes from the HTML code above following the JSON template. For any unanswerable attributes in the template, set their value to the placeholder "<NULL>".
{{prompt_prefix}}"""

job = """{{page_code}}

Here is the JSON template for job attribute extraction:
{"webpage title": "xx", "job title": "xx", "company": "xx", "location": "xx", "date posted": "xx"}

Please extract the job' attributes from the HTML code above following the JSON template. For any unanswerable attributes in the template, set their value to the placeholder "<NULL>".
{{prompt_prefix}}"""

movie = """{{page_code}}

Here is the JSON template for movie attribute extraction:
{"webpage title": "xx", "movie title": "xx", "director": "xx", "mpaa rating": "xx", "genre": "xx"}

Please extract the movie' attributes from the HTML code above following the JSON template. For any unanswerable attributes in the template, set their value to the placeholder "<NULL>".
{{prompt_prefix}}"""

nbaplayer = """{{page_code}}

Here is the JSON template for nba player attribute extraction:
{"webpage title": "xx", "player name": "xx", "team": "xx", "height": "xx", "weight": "xx"}

Please extract the nba player' attributes from the HTML code above following the JSON template. For any unanswerable attributes in the template, set their value to the placeholder "<NULL>".
{{prompt_prefix}}"""

restaurant = """{{page_code}}

Here is the JSON template for restaurant attribute extraction:
{"webpage title": "xx", "restaurant name": "xx", "address": "xx", "phone": "xx", "cuisine type": "xx"}

Please extract the restaurant' attributes from the HTML code above following the JSON template. For any unanswerable attributes in the template, set their value to the placeholder "<NULL>".
{{prompt_prefix}}"""

university = """{{page_code}}

Here is the JSON template for university attribute extraction:
{"webpage title": "xx", "university name": "xx", "university type (by fund source)": "xx", "website": "xx", "phone number": "xx"}

Please extract the university' attributes from the HTML code above following the JSON template. For any unanswerable attributes in the template, set their value to the placeholder "<NULL>".
{{prompt_prefix}}"""