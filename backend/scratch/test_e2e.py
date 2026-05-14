import sys
from pathlib import Path
import os
from unittest.mock import patch

# Add backend to sys.path
sys.path.append(str(Path(__file__).parent.parent))

import primitive_pipeline_v2

# Mock inputs for the pipeline
inputs = iter([
    "civil",  # Folder type
    "n",      # Process all PDFs?
    "2001",   # Enter folder names
    "1",      # Select MPNet
    "n"       # Run another model?
])

def mock_input(prompt):
    print(prompt, end="")
    val = next(inputs)
    print(val)
    return val

@patch('builtins.input', side_effect=mock_input)
def test_pipeline(mock_in):
    # This will run the logic at the bottom of primitive_pipeline_v2.py
    # but since it's not wrapped in a function, I have to be careful.
    # Actually, the file has code in the global scope for folder selection.
    pass

if __name__ == "__main__":
    print("Running E2E Test of Primitive Pipeline V2...")
    # Since primitive_pipeline_v2.py has a lot of top-level code,
    # importing it already runs some of it. I should have wrapped it.
    # Let me wrap it or run it via subprocess with expect.
