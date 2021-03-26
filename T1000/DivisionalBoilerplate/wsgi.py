# Application entry point.
from .DivisionalBoilerplate.create_application import create_app
from . import configure_flask_application

import os


# cf. pp. 44, Gaspar and Stouffer (2018)
# https://flask.palletsprojects.com/en/1.1.x/config/#development-production
app = create_app(configure_flask_application.DevelopmentConfiguration())


if __name__ == "__main__":

    # "0.0.0.0" is for localhost.
    app.run(host="0.0.0.0", debug=True)


# Or specify port manually:
'''
if __name__ == '__main__':

    # os.environ - mapping object representing the string environment. This
    # mapping is captured the first time os module is imported, typically
    # during Python startup.
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port)
'''