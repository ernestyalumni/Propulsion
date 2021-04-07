from . import configurations
from .WebApp.create_application import create_application

app = create_application(configurations.DevelopmentConfiguration())

@app.shell_context_processor
def make_shell_context():
