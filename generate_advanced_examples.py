#!/usr/bin/env python3
"""
Advanced WorldModel Training Examples Generator
==============================================

Generates comprehensive training examples for system tasks, data analysis,
advanced math, string processing, and file operations.
"""

import random
import json
from datetime import datetime, timedelta
from pathlib import Path

def generate_system_environment_examples():
    """Generate system and environment task examples."""
    examples = []
    
    # Date and time examples
    date_time_tasks = [
        ("What's today's date?", 
         "I need to get today's date.",
         "from datetime import datetime\ntoday = datetime.now()\nprint(f\"Today's date is: {today.strftime('%Y-%m-%d')}\")",
         "python:datetime"),
        
        ("What time is it right now?",
         "I need to get the current time.",
         "from datetime import datetime\nnow = datetime.now()\nprint(f\"Current time: {now.strftime('%H:%M:%S')}\")",
         "python:datetime"),
         
        ("How many days until New Year?",
         "I need to calculate the days between now and New Year's Day.",
         "from datetime import datetime, date\ntoday = date.today()\nnew_year = date(today.year + 1, 1, 1)\ndays_left = (new_year - today).days\nprint(f\"Days until New Year: {days_left}\")",
         "python:datetime"),
         
        ("What day of the week is it?",
         "I need to get the current day of the week.",
         "from datetime import datetime\ntoday = datetime.now()\nday_name = today.strftime('%A')\nprint(f\"Today is: {day_name}\")",
         "python:datetime"),
         
        ("What's the timestamp for right now?",
         "I need to get the current Unix timestamp.",
         "import time\ntimestamp = time.time()\nprint(f\"Current timestamp: {timestamp}\")",
         "python:time")
    ]
    
    # System information examples
    system_tasks = [
        ("What's the current working directory?",
         "I need to get the current working directory path.",
         "import os\ncwd = os.getcwd()\nprint(f\"Current directory: {cwd}\")",
         "python:os"),
         
        ("What's my username?",
         "I need to get the current user's username.",
         "import os\nusername = os.getlogin()\nprint(f\"Username: {username}\")",
         "python:os"),
         
        ("What operating system am I on?",
         "I need to identify the operating system.",
         "import platform\nos_name = platform.system()\nprint(f\"Operating System: {os_name}\")",
         "python:platform"),
         
        ("How much memory is available?",
         "I need to check the system's memory usage.",
         "import psutil\nmemory = psutil.virtual_memory()\navailable_gb = memory.available / (1024**3)\nprint(f\"Available memory: {available_gb:.2f} GB\")",
         "python:psutil"),
         
        ("Get the current Python version",
         "I need to check what Python version is running.",
         "import sys\nversion = sys.version\nprint(f\"Python version: {version}\")",
         "python:sys")
    ]
    
    # Environment variables
    env_tasks = [
        ("Check if DEBUG environment variable is set",
         "I need to check for a specific environment variable.",
         "import os\ndebug = os.environ.get('DEBUG', 'Not set')\nprint(f\"DEBUG environment variable: {debug}\")",
         "python:os"),
         
        ("Get the PATH environment variable",
         "I need to get the PATH environment variable.",
         "import os\npath = os.environ.get('PATH', '')\nprint(f\"PATH: {path[:100]}...\")  # Show first 100 chars",
         "python:os")
    ]
    
    # Combine all system examples
    all_system_tasks = date_time_tasks + system_tasks + env_tasks
    
    for task, thinking, code, requires in all_system_tasks:
        examples.append({
            "user": task,
            "thinking": thinking,
            "code": code,
            "requires": requires,
            "explanation": f"The system information has been retrieved using Python's built-in modules."
        })
    
    return examples

def generate_data_analysis_examples():
    """Generate data analysis and processing examples."""
    examples = []
    
    # Statistical calculations
    stats_tasks = [
        ("Calculate the standard deviation of [1, 2, 3, 4, 5]",
         "I need to calculate the standard deviation of this dataset.",
         "import statistics\ndata = [1, 2, 3, 4, 5]\nstd_dev = statistics.stdev(data)\nprint(f\"Standard deviation: {std_dev:.4f}\")",
         "python:statistics"),
         
        ("Find the median of [7, 3, 9, 1, 5, 8]",
         "I need to find the median value of this list.",
         "import statistics\ndata = [7, 3, 9, 1, 5, 8]\nmedian_val = statistics.median(data)\nprint(f\"Median: {median_val}\")",
         "python:statistics"),
         
        ("Calculate variance of [10, 20, 30, 40, 50]",
         "I need to calculate the variance of this dataset.",
         "import statistics\ndata = [10, 20, 30, 40, 50]\nvariance = statistics.variance(data)\nprint(f\"Variance: {variance}\")",
         "python:statistics"),
         
        ("Find the mode of [1, 2, 2, 3, 3, 3, 4]",
         "I need to find the most frequently occurring value.",
         "import statistics\ndata = [1, 2, 2, 3, 3, 3, 4]\nmode_val = statistics.mode(data)\nprint(f\"Mode: {mode_val}\")",
         "python:statistics")
    ]
    
    # Data processing
    data_tasks = [
        ("Parse JSON string and extract the 'name' field",
         "I need to parse JSON data and extract a specific field.",
         "import json\njson_str = '{\"name\": \"Alice\", \"age\": 30, \"city\": \"New York\"}'\ndata = json.loads(json_str)\nname = data['name']\nprint(f\"Name: {name}\")",
         "python:json"),
         
        ("Convert CSV data to list of dictionaries",
         "I need to process CSV data into a structured format.",
         "import csv\nimport io\ncsv_data = \"name,age,city\\nAlice,30,NYC\\nBob,25,LA\"\nreader = csv.DictReader(io.StringIO(csv_data))\nrecords = list(reader)\nprint(f\"Records: {records}\")",
         "python:csv"),
         
        ("Calculate moving average of [1,2,3,4,5,6,7,8,9,10]",
         "I need to calculate a 3-period moving average.",
         "data = [1,2,3,4,5,6,7,8,9,10]\nwindow = 3\nmoving_avg = []\nfor i in range(len(data) - window + 1):\n    avg = sum(data[i:i+window]) / window\n    moving_avg.append(avg)\nprint(f\"Moving averages: {moving_avg}\")",
         "python:math")
    ]
    
    # Combine all data analysis examples
    all_data_tasks = stats_tasks + data_tasks
    
    for task, thinking, code, requires in all_data_tasks:
        examples.append({
            "user": task,
            "thinking": thinking,
            "code": code,
            "requires": requires,
            "explanation": f"The data analysis has been completed using appropriate Python libraries."
        })
    
    return examples

def generate_string_processing_examples():
    """Generate advanced string and text processing examples."""
    examples = []
    
    # Regular expressions
    regex_tasks = [
        ("Find all email addresses in this text: 'Contact john@example.com or jane@test.org'",
         "I need to use regular expressions to find email addresses.",
         "import re\ntext = 'Contact john@example.com or jane@test.org'\nemails = re.findall(r'[\\w\\.-]+@[\\w\\.-]+\\.\\w+', text)\nprint(f\"Found emails: {emails}\")",
         "python:re"),
         
        ("Extract all phone numbers from: 'Call 555-123-4567 or (555) 987-6543'",
         "I need to find phone numbers using pattern matching.",
         "import re\ntext = 'Call 555-123-4567 or (555) 987-6543'\nphones = re.findall(r'(?:\\(?\\d{3}\\)?[\\s.-]?)?\\d{3}[\\s.-]?\\d{4}', text)\nprint(f\"Found phones: {phones}\")",
         "python:re"),
         
        ("Find all words starting with 'un' in: 'The unusual unicorn was unfriendly'",
         "I need to find words with a specific prefix.",
         "import re\ntext = 'The unusual unicorn was unfriendly'\nwords = re.findall(r'\\bun\\w*', text, re.IGNORECASE)\nprint(f\"Words starting with 'un': {words}\")",
         "python:re")
    ]
    
    # Text analysis
    text_tasks = [
        ("Count the number of sentences in this text",
         "I need to count sentences by finding sentence-ending punctuation.",
         "import re\ntext = \"Hello world. How are you? I'm fine! Thanks.\"\nsentences = re.split(r'[.!?]+', text)\nsentences = [s.strip() for s in sentences if s.strip()]\ncount = len(sentences)\nprint(f\"Number of sentences: {count}\")",
         "python:re"),
         
        ("Find the longest word in: 'The quick brown fox jumps'",
         "I need to split the text and find the longest word.",
         "text = 'The quick brown fox jumps'\nwords = text.split()\nlongest = max(words, key=len)\nprint(f\"Longest word: '{longest}' ({len(longest)} characters)\")",
         "python:str"),
         
        ("Count word frequency in: 'the cat and the dog and the bird'",
         "I need to count how often each word appears.",
         "from collections import Counter\ntext = 'the cat and the dog and the bird'\nwords = text.lower().split()\nword_freq = Counter(words)\nprint(f\"Word frequencies: {dict(word_freq)}\")",
         "python:collections"),
         
        ("Convert text to title case: 'hello world from python'",
         "I need to convert the text to title case.",
         "text = 'hello world from python'\ntitle_case = text.title()\nprint(f\"Title case: {title_case}\")",
         "python:str")
    ]
    
    # Encoding/decoding
    encoding_tasks = [
        ("Encode 'Hello World' to Base64",
         "I need to encode text to Base64 format.",
         "import base64\ntext = 'Hello World'\nencoded = base64.b64encode(text.encode()).decode()\nprint(f\"Base64 encoded: {encoded}\")",
         "python:base64"),
         
        ("Convert text to ASCII values: 'ABC'",
         "I need to get the ASCII values of each character.",
         "text = 'ABC'\nascii_values = [ord(char) for char in text]\nprint(f\"ASCII values: {ascii_values}\")",
         "python:ord"),
         
        ("Decode Base64: 'SGVsbG8gV29ybGQ='",
         "I need to decode this Base64 string.",
         "import base64\nencoded = 'SGVsbG8gV29ybGQ='\ndecoded = base64.b64decode(encoded).decode()\nprint(f\"Decoded text: {decoded}\")",
         "python:base64")
    ]
    
    # Combine all string processing examples
    all_string_tasks = regex_tasks + text_tasks + encoding_tasks
    
    for task, thinking, code, requires in all_string_tasks:
        examples.append({
            "user": task,
            "thinking": thinking,
            "code": code,
            "requires": requires,
            "explanation": f"The text processing has been completed using appropriate string manipulation techniques."
        })
    
    return examples

def generate_math_examples():
    """Generate advanced mathematical operation examples."""
    examples = []
    
    # Trigonometry
    trig_tasks = [
        ("Calculate sine of 45 degrees",
         "I need to calculate the sine of 45 degrees, converting to radians first.",
         "import math\nangle_degrees = 45\nangle_radians = math.radians(angle_degrees)\nsine_value = math.sin(angle_radians)\nprint(f\"sin({angle_degrees}°) = {sine_value:.6f}\")",
         "python:math"),
         
        ("Find the hypotenuse of a right triangle with sides 3 and 4",
         "I need to use the Pythagorean theorem to find the hypotenuse.",
         "import math\na, b = 3, 4\nhypotenuse = math.sqrt(a**2 + b**2)\nprint(f\"Hypotenuse = {hypotenuse}\")",
         "python:math"),
         
        ("Convert 30 degrees to radians",
         "I need to convert degrees to radians.",
         "import math\ndegrees = 30\nradians = math.radians(degrees)\nprint(f\"{degrees} degrees = {radians:.6f} radians\")",
         "python:math")
    ]
    
    # Number theory
    number_tasks = [
        ("Find the GCD of 48 and 18",
         "I need to find the greatest common divisor of these numbers.",
         "import math\na, b = 48, 18\ngcd = math.gcd(a, b)\nprint(f\"GCD of {a} and {b} is {gcd}\")",
         "python:math"),
         
        ("Generate the first 10 Fibonacci numbers",
         "I need to generate the Fibonacci sequence.",
         "def fibonacci(n):\n    fib = [0, 1]\n    for i in range(2, n):\n        fib.append(fib[i-1] + fib[i-2])\n    return fib[:n]\n\nfib_sequence = fibonacci(10)\nprint(f\"First 10 Fibonacci numbers: {fib_sequence}\")",
         "python:math"),
         
        ("Check if 97 is a prime number",
         "I need to check if 97 is prime by testing for divisors.",
         "def is_prime(n):\n    if n < 2:\n        return False\n    for i in range(2, int(n**0.5) + 1):\n        if n % i == 0:\n            return False\n    return True\n\nnum = 97\nresult = is_prime(num)\nprint(f\"{num} is {'prime' if result else 'not prime'}\")",
         "python:math"),
         
        ("Calculate factorial of 7",
         "I need to calculate 7 factorial (7!).",
         "import math\nn = 7\nfactorial = math.factorial(n)\nprint(f\"{n}! = {factorial}\")",
         "python:math")
    ]
    
    # Linear algebra
    linear_tasks = [
        ("Calculate dot product of vectors [1,2,3] and [4,5,6]",
         "I need to calculate the dot product of these two vectors.",
         "def dot_product(v1, v2):\n    return sum(a * b for a, b in zip(v1, v2))\n\nvec1 = [1, 2, 3]\nvec2 = [4, 5, 6]\ndot_prod = dot_product(vec1, vec2)\nprint(f\"Dot product: {dot_prod}\")",
         "python:math"),
         
        ("Add matrices [[1,2],[3,4]] and [[5,6],[7,8]]",
         "I need to perform matrix addition.",
         "def add_matrices(m1, m2):\n    return [[m1[i][j] + m2[i][j] for j in range(len(m1[0]))] for i in range(len(m1))]\n\nmat1 = [[1, 2], [3, 4]]\nmat2 = [[5, 6], [7, 8]]\nresult = add_matrices(mat1, mat2)\nprint(f\"Matrix sum: {result}\")",
         "python:math")
    ]
    
    # Combine all math examples
    all_math_tasks = trig_tasks + number_tasks + linear_tasks
    
    for task, thinking, code, requires in all_math_tasks:
        examples.append({
            "user": task,
            "thinking": thinking,
            "code": code,
            "requires": requires,
            "explanation": f"The mathematical calculation has been completed using appropriate algorithms."
        })
    
    return examples

def generate_file_examples():
    """Generate file and directory operation examples."""
    examples = []
    
    # File operations
    file_tasks = [
        ("Create a temporary file with some text",
         "I need to create a temporary file and write text to it.",
         "import tempfile\nimport os\n\nwith tempfile.NamedTemporaryFile(mode='w', delete=False) as f:\n    f.write('Hello, World!')\n    temp_path = f.name\n\nprint(f\"Created temporary file: {temp_path}\")\nos.unlink(temp_path)  # Clean up\nprint(\"File cleaned up\")",
         "python:tempfile"),
         
        ("Get file extension from 'document.pdf'",
         "I need to extract the file extension from a filename.",
         "import os\nfilename = 'document.pdf'\nname, ext = os.path.splitext(filename)\nprint(f\"Filename: {name}\")\nprint(f\"Extension: {ext}\")",
         "python:os"),
         
        ("Join path components: '/home', 'user', 'documents', 'file.txt'",
         "I need to properly join path components.",
         "import os\npath_parts = ['home', 'user', 'documents', 'file.txt']\nfull_path = os.path.join('/', *path_parts)\nprint(f\"Full path: {full_path}\")",
         "python:os"),
         
        ("Get the size of the current Python file",
         "I need to get the file size of the current script.",
         "import os\nimport sys\n\nscript_path = sys.argv[0] if sys.argv[0] else __file__\nif os.path.exists(script_path):\n    size = os.path.getsize(script_path)\n    print(f\"Script size: {size} bytes\")\nelse:\n    print(\"Current script path not found\")",
         "python:os")
    ]
    
    # Directory operations  
    dir_tasks = [
        ("List all Python files in current directory",
         "I need to find all .py files in the current directory.",
         "import os\nimport glob\n\npython_files = glob.glob('*.py')\nprint(f\"Python files found: {len(python_files)}\")\nfor file in python_files[:5]:  # Show first 5\n    print(f\"  {file}\")",
         "python:glob"),
         
        ("Get the parent directory of current path",
         "I need to find the parent directory.",
         "import os\ncurrent_dir = os.getcwd()\nparent_dir = os.path.dirname(current_dir)\nprint(f\"Current: {current_dir}\")\nprint(f\"Parent: {parent_dir}\")",
         "python:os"),
         
        ("Check if a directory exists",
         "I need to check if a specific directory exists.",
         "import os\ndir_path = '/tmp'\nif os.path.isdir(dir_path):\n    print(f\"Directory '{dir_path}' exists\")\nelse:\n    print(f\"Directory '{dir_path}' does not exist\")",
         "python:os")
    ]
    
    # Path operations
    path_tasks = [
        ("Get absolute path of current directory",
         "I need to get the absolute path of the current directory.",
         "import os\nrelative_path = '.'\nabsolute_path = os.path.abspath(relative_path)\nprint(f\"Absolute path: {absolute_path}\")",
         "python:os"),
         
        ("Extract directory name from path '/home/user/file.txt'",
         "I need to extract the directory portion from a file path.",
         "import os\nfile_path = '/home/user/file.txt'\ndirectory = os.path.dirname(file_path)\nbasename = os.path.basename(file_path)\nprint(f\"Directory: {directory}\")\nprint(f\"Filename: {basename}\")",
         "python:os")
    ]
    
    # Combine all file examples
    all_file_tasks = file_tasks + dir_tasks + path_tasks
    
    for task, thinking, code, requires in all_file_tasks:
        examples.append({
            "user": task,
            "thinking": thinking,
            "code": code,
            "requires": requires,
            "explanation": f"The file operation has been completed using Python's file system utilities."
        })
    
    return examples

def format_for_training(examples):
    """Convert examples to WorldModel training format."""
    training_text = ""
    
    for example in examples:
        training_text += f"User: {example['user']}\n"
        training_text += f"Assistant: <think>{example['thinking']}</think>\n"
        training_text += f"<model>\n{example['code']}\n</model>\n"
        training_text += f"<requires>{example['requires']}</requires>\n\n"
        training_text += f"{example['explanation']}\n\n"
    
    return training_text

def main():
    """Generate all advanced training examples."""
    print("🔧 Generating Advanced WorldModel Training Examples")
    print("=" * 55)
    
    # Generate all categories
    categories = {
        "System & Environment": generate_system_environment_examples(),
        "Data Analysis": generate_data_analysis_examples(), 
        "String Processing": generate_string_processing_examples(),
        "Advanced Math": generate_math_examples(),
        "File Operations": generate_file_examples()
    }
    
    # Combine all examples
    all_examples = []
    for category, examples in categories.items():
        all_examples.extend(examples)
        print(f"✅ {category}: {len(examples)} examples")
    
    print(f"\n📊 Total examples generated: {len(all_examples)}")
    
    # Format for training
    training_text = format_for_training(all_examples)
    
    # Save to file
    output_file = Path("data/worldmodel_advanced_training.txt")
    output_file.parent.mkdir(exist_ok=True)
    
    with open(output_file, 'w') as f:
        f.write(training_text)
    
    print(f"💾 Saved to: {output_file}")
    print(f"📏 File size: {len(training_text):,} characters")
    
    # Also save as JSON for flexibility
    json_file = Path("data/worldmodel_advanced_training.json")
    with open(json_file, 'w') as f:
        json.dump(all_examples, f, indent=2)
    
    print(f"💾 JSON version: {json_file}")

if __name__ == "__main__":
    main()