from pathlib import Path

from ..utilities import *

# os.path.abspath(os.path.dirname(__file__))
base_directory = str(Path(__file__).parent.absolute())

# Enable debug mode.
DEBUG = True

# Secret key for session management.
SECRET_KEY = 'my precious'