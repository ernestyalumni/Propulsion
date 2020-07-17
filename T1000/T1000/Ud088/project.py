"""
@file 

@ref https://github.com/udacity/Full-Stack-Foundations/blob/master/Lesson-3/01_First-Flask-App/project.py

@details Full-Stack-Foundations Lesson 3
Full-Stack-Foundations/Lesson-3/01_First-Flask-App/project.py

python3 project.py # To run it
Ctrl-C to stop server.
"""
# request for getting data from a form.
from flask import (Flask,
    flash, jsonify, render_template, request, redirect, url_for)

# cf. https://github.com/udacity/Full-Stack-Foundations/blob/master/Lesson-3/02_Adding-Database-to-Flask-Application/project.py
# Lesson-3 / 02_Adding_Database-to-Flask-Application

from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from database_setup import Base, Restaurant, MenuItem, query_as_dict

# cf. https://classroom.udacity.com/courses/ud088/lessons/3593308717/concepts/36245586050923
# 01_First-Flask-App

# Create instance of this class.
# Python sets __name__ to '__main__' when script is entry point for python
# interpreter.
app = Flask(__name__)


engine = create_engine('sqlite:///restaurantmenu.db')
Base.metadata.bind = engine

DBSession = sessionmaker(bind=engine)
session = DBSession()


# https://github.com/udacity/Full-Stack-Foundations/blob/master/Lesson-3/19_Responding-with-JSON/project.py
@app.route('/restaurants/<int:restaurant_id>/menu/JSON')
def restaurant_menu_json(restaurant_id):

    restaurant = session.query(Restaurant).filter_by(id=restaurant_id).one()
    items = session.query(MenuItem).filter_by(
        restaurant_id=restaurant_id).all()

    # i.serialize is a Python dictionary
    return jsonify(MenuItems=[i.serialize for i in items])


# cf. https://classroom.udacity.com/courses/ud088/lessons/3593308717/concepts/36245586370923

# ADD JSON ENDPOINT HERE
@app.route('/restaurants/<int:restaurant_id>/menu/<int:menu_id>/JSON')
def menu_item_json(restaurant_id, menu_id):

    menu_item = session.query(MenuItem).filter_by(id=menu_id).one()
    return jsonify(MenuItem=menu_item.serialize)


@app.route('/test_get_restaurant/', methods=['GET', 'POST'])
def test_get_restaurant():

    if request.method == 'POST':

        test_restaurant_id = request.form['restaurant_id']

        test_restaurant_query_object = session.query(Restaurant).filter_by(
            id=test_restaurant_id).one()

        test_restaurant_json = query_as_dict(test_restaurant_query_object)

        print(test_restaurant_json)

        #test_restaurant_serialized = jsonify(test_restaurant_query_object)

        # https://stackoverflow.com/questions/42098396/redirect-to-external-url-while-sending-a-json-object-or-string-from-flask-app
        # Consider making another database
        # TODO: Make another database.
        return test_restaurant_id

    else:

        return render_template('test_get_restaurant.html')

# TODO: Consider using another database.
"""
@app.route('/test_get_restaurant_json/<int:restaurant_id>',
    methods=['GET', 'POST'])
def test_get_restaurant_json(restaurant_id, restaurant_json):

    return
"""


# Decorator. Decorators can be stacked on top of each other; @app.route('/')
# calls @app.route('/hello'), and that calls HelloWorld
# Takes same page as '/hello'
@app.route('/') # Will call function that follows it.
@app.route('/hello')
def HelloWorld():

    restaurant = session.query(Restaurant).first()
    items = session.query(MenuItem).filter_by(restaurant_id=restaurant.id)

    output = ''

    output += "Hello World\n"

    for i in items:

        output += i.name
        output += '<br>'
        output += i.price
        output += '<br>'
        output += i.description
        output += '<br>'

    return output

# URLs with Variables
# https://classroom.udacity.com/courses/ud088/lessons/3593308717/concepts/36245586130923
# https://github.com/udacity/Full-Stack-Foundations/blob/master/Lesson-3/04_Routing/project.py
# 04_Routing/project.py

# To add variables to our URL, specify a rule
# "path/<type:variable_name>/path"

# go to this page by typing in the url directly into browser address
# e.g. http://0.0.0.0:5000/restaurants/2/
@app.route('/restaurants/<int:restaurant_id>/')
def restaurant_menu(restaurant_id):
    
    restaurant = session.query(Restaurant).filter_by(id=restaurant_id).one()
    items = session.query(MenuItem).filter_by(restaurant_id=restaurant.id)

    """
    # Previous answer before render_template

    output = ''
    for i in items:
        output += i.name
        output += '<br>'
        output += i.price
        output += '<br>'
        output += i.description
        output += '</br>'
        output += '</br>'
    """

    # cf. https://classroom.udacity.com/courses/ud088/lessons/3593308717/concepts/36245586170923    
    # render_template(templateName.html, Variable = keyword)
    # Variable are what you want to pass into the engine as keyword arguments.
    # Flask will look for your templates in folder called templates.
    # cf. https://stackoverflow.com/questions/53888565/flask-cannot-find-html-files-in-templates-folder-when-run-through-gunicorn

    # cf. https://flask.palletsprojects.com/en/1.1.x/api/
    # Template Rendering
    # flask.render_template(template_name_or_list, **context)
    # Parameters: * template_name_or_list - name of template to be rendered, or
    # an iterable with template names, first 1 existing will be rendered
    # * context - variables that should be available in context of template

    output = render_template(
        'menu.html',
        restaurant=restaurant,
        items=items,
        restaurant_id=restaurant_id)

    return output
    #return render_template('menu.html', restaurant=restaurant, items=items)



# Task 1: Create route for new_menu_item function here

# cf. https://classroom.udacity.com/courses/ud088/lessons/3593308717/concepts/36245586240923
# Form requests and Redirects
# Change the response to GET and POST.
# Recall that GET is "Read" and POST is "Create"
# Now can respond to POST requests.
# Now make forms.
@app.route('/restaurant/<int:restaurant_id>/new/', methods=['GET', 'POST'])
def new_menu_item(restaurant_id):

    # Looks for a POST request
    if request.method == 'POST':

        new_item = MenuItem(

            # Extract the name field from form.
            name=request.form['name'],
            description=request.form['description'],
            price=request.form['price'],
            course=request.form['course'],
            restaurant_id=restaurant_id)

        session.add(new_item)
        session.commit()

        # cf. https://classroom.udacity.com/courses/ud088/lessons/3593308717/concepts/36245586310923
        # 15. Message Flashing

        #
        flash("new menu item created!")

        # To redirect user back to main user page.
        # flask.url_for(endpoint, **values)
        # https://flask.palletsprojects.com/en/1.1.x/api/
        # Generates a URL to given endpoint with method provided.
        # Variable arguments that are unknown to the target endpoint are
        # appended to generated URL as query arguments.
        return redirect(url_for(
            'restaurant_menu',
            restaurant_id=restaurant_id))

    # If server didn't receive "POST" request, it's a "GET" request.
    else:

        return render_template(
            'new_menu_item.html',
            restaurant_id=restaurant_id)

    #return "page to create a new menu item. Task 1 complete!"


# Task 2: Create route for edit_menu_item function here

@app.route(
    '/restaurant/<int:restaurant_id>/<int:menu_id>/edit/',
    methods=['GET', 'POST'])
def edit_menu_item(restaurant_id, menu_id):

    edited_item = session.query(MenuItem).filter_by(id=menu_id).one()

    if request.method == 'POST':

        if request.form['name']:

            edited_item.name = request.form['name']

        if request.form['description']:

            edited_item.description = request.form['description']

        if request.form['price']:

            edited_item.price = request.form['price']

        if request.form['course']:

            edited_item.course = request.form['course']

        session.add(edited_item)
        session.commit()

        flash("Menu Item has been edited")

        return redirect(
            url_for('restaurant_menu', restaurant_id=restaurant_id))

    else:

      # Use the render_template function below to see the variables you
      # should use in your edit_menu_item template.

      return render_template(
          'edit_menu_item.html',
          restaurant_id=restaurant_id,
          menu_id=menu_id,
          item=edited_item)
    #return "page to edit a menu item. Task 2 complete!"


# Task 3: Create a route for delete_menu_item function here

@app.route(
    '/restaurant/<int:restaurant_id>/<int:menu_id>/delete/',
    methods=['GET', 'POST'])
def delete_menu_item(restaurant_id, menu_id):

    # menu_id flows through to here, to object Python SQLAlchemy object.
    # Go to the "bottom", to the else statement, to see how object gets used.
    item_to_delete = session.query(MenuItem).filter_by(id=menu_id).one()

    if request.method == 'POST':

        session.delete(item_to_delete)
        session.commit()

        flash("Menu Item has been deleted")

        return redirect(url_for('restaurant_menu', restaurant_id=restaurant_id))

    # if it's a GET request
    else:

        return render_template('delete_menu_item.html', item=item_to_delete)


    #return "page to delete a menu item. Task 3 complete!"


if __name__ == '__main__':

    # cf. https://classroom.udacity.com/courses/ud088/lessons/3593308717/concepts/36245586310923
    # Add a secret key for flash, for sessions. Used to create sessions for our
    # Flask application.
    # For development, set a dummy key.
    # 15. Message Flashing
    app.secret_key = "super_secret_key"

    # In debug mode, can execute any Python code.
    # In debug mode, it'll reload itself for any code change.
    app.debug = True
    # '0.0.0.0' means listen on all ports.
    app.run(host='0.0.0.0', port=5000)