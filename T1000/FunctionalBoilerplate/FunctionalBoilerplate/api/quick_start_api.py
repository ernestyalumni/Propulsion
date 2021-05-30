"""
@ref https://flask-restful.readthedocs.io/en/latest/quickstart.html

@details

EXAMPLE USAGE:
==============

$ python -i quick_start_api.py

curl http://127.0.0.1:5000/

from requests import put, get
put('http://localhost:5000/todo1', data={'data': 'Remember the milk'})
get('http://localhost:5000/todo1').json()
"""

from flask import Flask, request
from flask_restful import Api, Resource

app = Flask(__name__)

api = Api(app)

class HelloWorld(Resource):
    def get(self):
        return {'hello' : 'world'}


# Endpoints
# Many times in an API, your resource will have multiple URLs. You can pass
# multiple URLs to the add_resource() method on the Api object. Each one will
# be routed to your Resource.
api.add_resource(HelloWorld, '/', '/hello')

# Resourceful Routing.

"""
@details The main building block provided by Flask-RESTful are resources.
Resources are built on top of Flask pluggable views, giving you easy access to
multiple HTTP methods just by defining methods on your resource.
"""

todos = {}

class TodoSimple(Resource):
    def get(self, todo_id):
        return {todo_id: todos[todo_id]}

    def put(self, todo_id):
        todos[todo_id] = request.form['data']
        return {todo_id: todos[todo_id]}

# cf. https://flask-restful.readthedocs.io/en/latest/api.html#flask_restful.Api.add_resource
# endpoint (str) – endpoint name (defaults to Resource.__name__.lower()
# Can be used to reference this route in fields.Url fields
api.add_resource(TodoSimple, '/<string:todo_id>', endpoint='todo_ep')


# Flask-RESTful also support setting response code and response headers using
# multiple return values:

class Todo1(Resource):
    def get(self):
        # Default to 200 OK
        return {'task' : 'Hello world'}

class Todo2(Resource):
    def get(self):
        # Set the response code to 201
        return {'task' : 'Hello world'}, 201

class Todo3(Resource):
    def get(self):
        # Set the response code to 201 and return custom headers
        return {'task': 'Hello world'}, 201, {'Etag': 'some-opaque-string'}


if __name__ == '__main__':
    app.run(debug=True)    
